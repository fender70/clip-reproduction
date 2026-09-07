import random

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from clip_repro.data import build_splits
from clip_repro.evaluation import run_shape_probe_experiment


HELD_OUT_PAIRS = {("red", "triangle"), ("blue", "square")}


class FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.ReLU(),
        )
        self.projection = nn.Linear(2 * 4 * 4, 4)

    def forward(self, images):
        features = self.features(images).flatten(start_dim=1)
        return self.projection(features)


class FakeTinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeImageEncoder()

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)


def render_test_shape(combo):
    color, shape, size, position = combo
    value = (
        20
        + 20 * ("red", "blue", "green").index(color)
        + 30 * ("circle", "triangle", "square").index(shape)
        + random.randint(0, 2)
    )
    return Image.new("RGB", (4, 4), (value, value, value))


def test_shape_probe_is_frozen_leakage_checked_and_reproducible():
    train, test = build_splits(HELD_OUT_PAIRS)
    model = FakeTinyCLIP()
    weights_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    kwargs = {
        "model": model,
        "train_combinations": train,
        "test_combinations": test,
        "render_fn": render_test_shape,
        "device": "cpu",
        "train_samples_per_combo": 2,
        "test_samples_per_combo": 2,
        "probe_epochs": 2,
        "extraction_batch_size": 64,
    }

    first = run_shape_probe_experiment(**kwargs)
    second = run_shape_probe_experiment(**kwargs)

    assert first == second
    assert first["config"]["probe_train_combinations"] == 18
    assert set(first["stages"]) == {"pre_projection", "projected"}
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        torch.equal(weights_before[name], value)
        for name, value in model.state_dict().items()
    )


def test_shape_probe_rejects_held_out_training_leakage():
    train, test = build_splits(HELD_OUT_PAIRS)
    with pytest.raises(ValueError, match="leaked"):
        run_shape_probe_experiment(
            FakeTinyCLIP(),
            train + [test[0]],
            test[1:],
            render_fn=render_test_shape,
            device="cpu",
            train_samples_per_combo=1,
            test_samples_per_combo=1,
            probe_epochs=1,
        )
