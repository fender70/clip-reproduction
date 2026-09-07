"""Frozen linear probes for localizing TinyCLIP shape information."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from clip_repro.data import Combination, render_shape


SHAPES = ("circle", "triangle", "square")
SHAPE_TO_ID = {shape: index for index, shape in enumerate(SHAPES)}
DEFAULT_HELD_OUT_PAIRS = frozenset(
    {
        ("red", "triangle"),
        ("blue", "square"),
    }
)


def _validate_probe_split(
    train_combinations: Sequence[Combination],
    test_combinations: Sequence[Combination],
    held_out_pairs: frozenset[tuple[str, str]],
    probe_train_color: str,
) -> list[Combination]:
    if len(train_combinations) != len(set(train_combinations)):
        raise ValueError("Training combinations must be unique")
    if len(test_combinations) != len(set(test_combinations)):
        raise ValueError("Test combinations must be unique")
    if set(train_combinations) & set(test_combinations):
        raise ValueError("Training and test combinations must be disjoint")

    train_pairs = {combo[:2] for combo in train_combinations}
    test_pairs = {combo[:2] for combo in test_combinations}
    leaked_pairs = held_out_pairs & train_pairs
    if leaked_pairs:
        raise ValueError(f"Held-out pairs leaked into training: {sorted(leaked_pairs)}")
    if test_pairs != held_out_pairs:
        raise ValueError(
            "The test split must contain exactly the requested held-out pairs; "
            f"expected {sorted(held_out_pairs)}, found {sorted(test_pairs)}"
        )

    probe_combinations = [
        combo for combo in train_combinations if combo[0] == probe_train_color
    ]
    shapes = {combo[1] for combo in probe_combinations}
    if shapes != set(SHAPES):
        raise ValueError(
            f"Probe color {probe_train_color!r} must contain all shapes; "
            f"found {sorted(shapes)}"
        )

    # Each shape must have exactly the same size-position conditions. This
    # prevents either nuisance attribute from becoming a shape shortcut.
    conditions = {
        shape: Counter(
            (size, position)
            for _, candidate_shape, size, position in probe_combinations
            if candidate_shape == shape
        )
        for shape in SHAPES
    }
    reference = conditions[SHAPES[0]]
    if not reference or any(conditions[shape] != reference for shape in SHAPES[1:]):
        raise ValueError(
            "Green probe data must be balanced by shape for every size-position "
            "condition"
        )
    return probe_combinations


def _freeze_and_validate_model(model: nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("TinyCLIP parameters were not fully frozen")
    if not hasattr(model, "image_encoder") or not hasattr(
        model.image_encoder, "features"
    ):
        raise TypeError("Expected model.image_encoder.features")
    if not hasattr(model.image_encoder, "projection"):
        raise TypeError("Expected model.image_encoder.projection")


def _infer_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _make_feature_sets(
    model: nn.Module,
    combinations: Sequence[Combination],
    samples_per_combo: int,
    render_seed: int,
    device: torch.device,
    render_fn: Callable[[Combination], Any],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[Combination]]:
    if samples_per_combo < 1:
        raise ValueError("samples_per_combo must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(render_seed)
    torch.manual_seed(render_seed)

    try:
        images: list[torch.Tensor] = []
        labels: list[int] = []
        sampled_combinations: list[Combination] = []
        for combo in combinations:
            if combo[1] not in SHAPE_TO_ID:
                raise ValueError(f"Unknown shape in combination: {combo}")
            for _ in range(samples_per_combo):
                rendered = render_fn(combo)
                if isinstance(rendered, tuple):
                    rendered = rendered[0]
                images.append(pil_to_tensor(rendered).float().div(255.0))
                labels.append(SHAPE_TO_ID[combo[1]])
                sampled_combinations.append(combo)

        chunks = {"pre_projection": [], "projected": []}
        for start in range(0, len(images), batch_size):
            image_batch = torch.stack(images[start : start + batch_size]).to(device)
            pre_projection = model.image_encoder.features(image_batch).flatten(
                start_dim=1
            )
            projected = F.normalize(
                model.image_encoder.projection(pre_projection), dim=-1
            )
            chunks["pre_projection"].append(pre_projection.cpu())
            chunks["projected"].append(projected.cpu())

        features = {
            stage: torch.cat(stage_chunks)
            for stage, stage_chunks in chunks.items()
        }
        return features, torch.tensor(labels), sampled_combinations
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


class _LinearProbe(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.classifier = nn.Linear(mean.numel(), len(SHAPES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier((features - self.mean) / self.std)


def _train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> _LinearProbe:
    if epochs < 1:
        raise ValueError("probe_epochs must be positive")

    # Standardization is fit only on green probe-training representations.
    features = features.clone()
    labels = labels.clone()
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)

    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        probe = _LinearProbe(mean, std)
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        for _ in range(epochs):
            logits = probe(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return probe.eval()
    finally:
        torch.random.set_rng_state(torch_state)


@torch.no_grad()
def _predictions(probe: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return probe(features).argmax(dim=1)


def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return (predictions == labels).float().mean().item()


def _confusion_matrix(
    predictions: torch.Tensor, labels: torch.Tensor
) -> list[list[int]]:
    matrix = torch.zeros((len(SHAPES), len(SHAPES)), dtype=torch.long)
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1
    return matrix.tolist()


def run_shape_probe_experiment(
    model: nn.Module,
    train_combinations: Sequence[Combination],
    test_combinations: Sequence[Combination],
    *,
    device: str | torch.device | None = None,
    render_fn: Callable[[Combination], Any] = render_shape,
    held_out_pairs: Iterable[tuple[str, str]] = DEFAULT_HELD_OUT_PAIRS,
    probe_train_color: str = "green",
    train_samples_per_combo: int = 100,
    test_samples_per_combo: int = 100,
    train_render_seed: int = 111,
    test_render_seed: int = 999,
    probe_seed: int = 222,
    probe_epochs: int = 300,
    probe_learning_rate: float = 1e-2,
    probe_weight_decay: float = 0.0,
    extraction_batch_size: int = 256,
    include_confusion_matrices: bool = True,
) -> dict[str, object]:
    """Probe frozen CNN and projected image features using green shapes only."""
    held_out = frozenset(held_out_pairs)
    probe_train_combinations = _validate_probe_split(
        train_combinations,
        test_combinations,
        held_out,
        probe_train_color,
    )
    _freeze_and_validate_model(model)
    selected_device = torch.device(device) if device else _infer_device(model)
    if _infer_device(model) != selected_device:
        model.to(selected_device)
    model.eval()

    train_features, train_labels, _ = _make_feature_sets(
        model,
        probe_train_combinations,
        train_samples_per_combo,
        train_render_seed,
        selected_device,
        render_fn,
        extraction_batch_size,
    )
    test_features, test_labels, sampled_test_combinations = _make_feature_sets(
        model,
        list(test_combinations),
        test_samples_per_combo,
        test_render_seed,
        selected_device,
        render_fn,
        extraction_batch_size,
    )

    results: dict[str, object] = {
        "config": {
            "probe_train_color": probe_train_color,
            "held_out_pairs": sorted(held_out),
            "probe_train_combinations": len(probe_train_combinations),
            "train_samples_per_combo": train_samples_per_combo,
            "test_samples_per_combo": test_samples_per_combo,
            "train_render_seed": train_render_seed,
            "test_render_seed": test_render_seed,
            "probe_seed": probe_seed,
        },
        "shape_order": list(SHAPES),
        "stages": {},
    }

    stages: dict[str, object] = {}
    for stage in ("pre_projection", "projected"):
        probe = _train_probe(
            train_features[stage],
            train_labels,
            epochs=probe_epochs,
            learning_rate=probe_learning_rate,
            weight_decay=probe_weight_decay,
            seed=probe_seed,
        )
        train_predictions = _predictions(probe, train_features[stage])
        test_predictions = _predictions(probe, test_features[stage])

        pair_results: dict[tuple[str, str], object] = {}
        for pair in sorted(held_out):
            mask = torch.tensor(
                [combo[:2] == pair for combo in sampled_test_combinations],
                dtype=torch.bool,
            )
            pair_predictions = test_predictions[mask]
            pair_labels = test_labels[mask]
            pair_result: dict[str, object] = {
                "accuracy": _accuracy(pair_predictions, pair_labels),
                "prediction_counts": Counter(
                    SHAPES[int(prediction)] for prediction in pair_predictions
                ),
                "num_examples": int(mask.sum()),
            }
            if include_confusion_matrices:
                pair_result["confusion_matrix"] = _confusion_matrix(
                    pair_predictions, pair_labels
                )
            pair_results[pair] = pair_result

        stage_result: dict[str, object] = {
            "train_accuracy": _accuracy(train_predictions, train_labels),
            "pairs": pair_results,
        }
        if include_confusion_matrices:
            stage_result["train_confusion_matrix"] = _confusion_matrix(
                train_predictions, train_labels
            )
        stages[stage] = stage_result

    results["stages"] = stages
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("TinyCLIP became trainable during the probe experiment")
    return results


def probe_result_rows(results: dict[str, object]) -> list[dict[str, object]]:
    """Flatten experiment results into one notebook-friendly row per stage."""
    rows: list[dict[str, object]] = []
    for stage, stage_result in results["stages"].items():
        row: dict[str, object] = {
            "stage": stage,
            "train_accuracy": stage_result["train_accuracy"],
        }
        for pair, pair_result in stage_result["pairs"].items():
            row[f"{'_'.join(pair)}_accuracy"] = pair_result["accuracy"]
        rows.append(row)
    return rows
