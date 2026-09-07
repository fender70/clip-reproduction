from __future__ import annotations

import math
import random
from functools import partial
from itertools import product
from typing import Iterable, Sequence, TypeAlias

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


Combination: TypeAlias = tuple[str, str, str, str]
ColorShapePair: TypeAlias = tuple[str, str]

COLORS = ("red", "blue", "green")
SHAPES = ("circle", "triangle", "square")
SIZES = ("small", "large")
POSITIONS = ("left", "center", "right")

COLOR_INTENSITY_RANGE = (160, 255)
SIZE_RANGES = {
    "small": (8, 14),
    "large": (22, 28),
}
POSITION_RANGES = {
    "left": (10, 18),
    "center": (26, 38),
    "right": (46, 54),
}


def make_caption(combo: Combination) -> str:
    color, shape, size, position = combo
    location = "in the center" if position == "center" else f"on the {position}"
    return f"a {size} {color} {shape} {location}"


def all_combinations() -> list[Combination]:
    return list(product(COLORS, SHAPES, SIZES, POSITIONS))


def build_splits(
    held_out_pairs: Iterable[ColorShapePair],
) -> tuple[list[Combination], list[Combination]]:
    held_out_pairs = set(held_out_pairs)
    valid_pairs = set(product(COLORS, SHAPES))
    unknown_pairs = held_out_pairs - valid_pairs
    if unknown_pairs:
        raise ValueError(f"Unknown held-out color-shape pairs: {sorted(unknown_pairs)}")

    train: list[Combination] = []
    test: list[Combination] = []
    for combo in all_combinations():
        color, shape, _, _ = combo
        (test if (color, shape) in held_out_pairs else train).append(combo)

    expected_test = len(held_out_pairs) * len(SIZES) * len(POSITIONS)
    assert len(test) == expected_test
    assert len(train) + len(test) == len(all_combinations())
    assert all(combo[:2] not in held_out_pairs for combo in train)
    assert all(combo[:2] in held_out_pairs for combo in test)
    return train, test


def sample_color(color: str) -> tuple[int, int, int]:
    intensity = random.randint(*COLOR_INTENSITY_RANGE)
    if color == "red":
        return intensity, 0, 0
    if color == "green":
        return 0, intensity, 0
    if color == "blue":
        return 0, 0, intensity
    raise ValueError(f"Unknown color: {color}")


def sample_size(size: str) -> int:
    try:
        size_range = SIZE_RANGES[size]
    except KeyError as exc:
        raise ValueError(f"Unknown size: {size}") from exc
    return random.randint(*size_range)


def position_range(position: str) -> tuple[int, int]:
    try:
        return POSITION_RANGES[position]
    except KeyError as exc:
        raise ValueError(f"Unknown position: {position}") from exc


def valid_x_range(position: str, size: int, image_size: int = 64) -> tuple[int, int]:
    half = math.ceil(size / 2)
    region_min, region_max = position_range(position)
    lo = max(region_min, half)
    hi = min(region_max, image_size - half)
    if lo > hi:
        raise ValueError(
            "Empty x intersection for "
            f"position={position}, size={size}, image_size={image_size}"
        )
    return lo, hi


def sample_x(position: str, size: int, image_size: int = 64) -> int:
    return random.randint(*valid_x_range(position, size, image_size))


def check_all_position_size_intersections(image_size: int = 64) -> list[tuple[str, str, int]]:
    failures: list[tuple[str, str, int]] = []
    for size_name, position in product(SIZES, POSITIONS):
        lo, hi = SIZE_RANGES[size_name]
        for size in range(lo, hi + 1):
            try:
                valid_x_range(position, size, image_size)
            except ValueError:
                failures.append((size_name, position, size))
    return failures


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    x: int,
    y: int,
    size: int,
    fill: tuple[int, int, int],
) -> None:
    half = size / 2
    if shape == "circle":
        draw.ellipse([x - half, y - half, x + half, y + half], fill=fill)
    elif shape == "square":
        draw.rectangle([x - half, y - half, x + half, y + half], fill=fill)
    elif shape == "triangle":
        draw.polygon(
            [(x, y - half), (x - half, y + half), (x + half, y + half)],
            fill=fill,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")


def render_shape(
    combo: Combination,
    image_size: int = 64,
    return_metadata: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, object]]:
    color_name, shape, size_name, position = combo
    fill = sample_color(color_name)
    size = sample_size(size_name)
    x = sample_x(position, size, image_size)
    y = image_size // 2

    image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    _draw_shape(ImageDraw.Draw(image), shape, x, y, size, fill)

    if not return_metadata:
        return image
    return image, {
        "combo": combo,
        "caption": make_caption(combo),
        "rgb": fill,
        "size": size,
        "x": x,
        "y": y,
    }


class ShapeDataset(Dataset):
    """One stochastic rendering per semantic combination and retrieval."""

    def __init__(self, combinations: Sequence[Combination], image_size: int = 64):
        self.combinations = list(combinations)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.combinations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, Combination]:
        combo = self.combinations[index]
        image = pil_to_tensor(render_shape(combo, self.image_size)).float().div(255.0)
        return image, make_caption(combo), combo


class SimpleTokenizer:
    """Deterministic whitespace tokenizer for the controlled caption language."""

    def __init__(self, captions: Iterable[str]):
        words = sorted({word for caption in captions for word in caption.lower().split()})
        self.word_to_id = {"<pad>": 0, **{word: i + 1 for i, word in enumerate(words)}}
        self.id_to_word = {index: word for word, index in self.word_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def encode(self, caption: str) -> list[int]:
        return [self.word_to_id[word] for word in caption.lower().split()]

    def batch_encode(self, captions: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(caption) for caption in captions]
        if not encoded:
            raise ValueError("Cannot encode an empty caption batch")

        tokens = torch.zeros((len(encoded), max(map(len, encoded))), dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        for row, sequence in enumerate(encoded):
            length = len(sequence)
            tokens[row, :length] = torch.tensor(sequence)
            mask[row, :length] = 1.0
        return tokens, mask


def build_tokenizer(combinations: Iterable[Combination]) -> SimpleTokenizer:
    return SimpleTokenizer([make_caption(combo) for combo in combinations])


def collate_shape_batch(
    batch: Sequence[tuple[torch.Tensor, str, Combination]],
    *,
    tokenizer: SimpleTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[Combination]]:
    images, captions, combos = zip(*batch)
    tokens, mask = tokenizer.batch_encode(captions)
    return torch.stack(images), tokens, mask, list(captions), list(combos)


def make_collate_fn(tokenizer: SimpleTokenizer):
    return partial(collate_shape_batch, tokenizer=tokenizer)
