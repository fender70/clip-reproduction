from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, Sequence

import torch
from torchvision.transforms.functional import pil_to_tensor

from clip_repro.data import (
    COLORS,
    SHAPES,
    ColorShapePair,
    Combination,
    SimpleTokenizer,
    make_caption,
    render_shape,
)
from clip_repro.models import TinyCLIP


@torch.no_grad()
def encode_caption_bank(
    model: TinyCLIP,
    combinations: Sequence[Combination],
    tokenizer: SimpleTokenizer,
    device: str | torch.device,
) -> tuple[list[str], torch.Tensor]:
    captions = [make_caption(combo) for combo in combinations]
    if len(captions) != len(set(captions)):
        raise ValueError("Candidate captions must be unique")
    tokens, mask = tokenizer.batch_encode(captions)
    return captions, model.encode_text(tokens.to(device), mask.to(device))


@torch.no_grad()
def evaluate_retrieval(
    model: TinyCLIP,
    query_combinations: Sequence[Combination],
    candidate_combinations: Sequence[Combination],
    tokenizer: SimpleTokenizer,
    device: str | torch.device,
    samples_per_combo: int = 20,
) -> dict[str, object]:
    if samples_per_combo < 1:
        raise ValueError("samples_per_combo must be positive")

    model.eval()
    captions, text_embeddings = encode_caption_bank(
        model, candidate_combinations, tokenizer, device
    )
    caption_to_index = {caption: index for index, caption in enumerate(captions)}
    results: list[dict[str, object]] = []

    for combo in query_combinations:
        color, shape, size, position = combo
        correct_caption = make_caption(combo)
        if correct_caption not in caption_to_index:
            raise ValueError(f"Missing target caption from candidate bank: {correct_caption}")

        for _ in range(samples_per_combo):
            image = pil_to_tensor(render_shape(combo)).float().div(255.0).unsqueeze(0).to(device)
            similarities = (model.encode_image(image) @ text_embeddings.T).squeeze(0)
            predicted_caption = captions[similarities.argmax().item()]
            correct_score = similarities[caption_to_index[correct_caption]].item()

            color_distractors = [
                make_caption((other_color, shape, size, position))
                for other_color in COLORS
                if other_color != color
            ]
            shape_distractors = [
                make_caption((color, other_shape, size, position))
                for other_shape in SHAPES
                if other_shape != shape
            ]
            missing = [
                caption
                for caption in color_distractors + shape_distractors
                if caption not in caption_to_index
            ]
            if missing:
                raise ValueError(f"Candidate bank is missing controlled distractors: {missing}")

            color_margin = correct_score - max(
                similarities[caption_to_index[caption]].item()
                for caption in color_distractors
            )
            shape_margin = correct_score - max(
                similarities[caption_to_index[caption]].item()
                for caption in shape_distractors
            )
            results.append(
                {
                    "combo": combo,
                    "correct_caption": correct_caption,
                    "predicted_caption": predicted_caption,
                    "correct_score": correct_score,
                    "color_margin": color_margin,
                    "shape_margin": shape_margin,
                }
            )

    if not results:
        raise ValueError("No query combinations were provided")
    return {
        "accuracy": mean(
            row["predicted_caption"] == row["correct_caption"] for row in results
        ),
        "color_margin": mean(float(row["color_margin"]) for row in results),
        "shape_margin": mean(float(row["shape_margin"]) for row in results),
        "results": results,
    }


def summarize_pair(
    evaluation: dict[str, object],
    pair: ColorShapePair,
) -> dict[str, float]:
    rows = [row for row in evaluation["results"] if row["combo"][:2] == pair]
    if not rows:
        raise ValueError(f"No evaluation rows found for pair {pair}")
    return {
        "accuracy": mean(row["predicted_caption"] == row["correct_caption"] for row in rows),
        "color_margin": mean(float(row["color_margin"]) for row in rows),
        "shape_margin": mean(float(row["shape_margin"]) for row in rows),
    }


def prediction_counts(
    evaluation: dict[str, object],
    pair: ColorShapePair | None = None,
) -> Counter[str]:
    rows: Iterable[dict[str, object]] = evaluation["results"]
    if pair is not None:
        rows = (row for row in rows if row["combo"][:2] == pair)
    return Counter(str(row["predicted_caption"]) for row in rows)
