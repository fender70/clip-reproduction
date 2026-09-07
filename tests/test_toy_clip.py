import torch
from torch.utils.data import DataLoader

from clip_repro.data import (
    ShapeDataset,
    all_combinations,
    build_splits,
    build_tokenizer,
    check_all_position_size_intersections,
    make_collate_fn,
    render_shape,
)
from clip_repro.losses import symmetric_clip_loss
from clip_repro.models import TinyCLIP
from clip_repro.utils import set_seed


HELD_OUT_PAIRS = {("red", "triangle"), ("blue", "square")}


def test_compositional_split_has_expected_counts_and_no_leakage():
    train, test = build_splits(HELD_OUT_PAIRS)
    assert len(train) == 42
    assert len(test) == 12
    assert all(combo[:2] not in HELD_OUT_PAIRS for combo in train)
    assert all(combo[:2] in HELD_OUT_PAIRS for combo in test)


def test_renderer_respects_geometry_and_is_seed_reproducible():
    assert check_all_position_size_intersections() == []
    combo = ("blue", "circle", "large", "center")
    set_seed(7)
    first_image, first_metadata = render_shape(combo, return_metadata=True)
    set_seed(7)
    second_image, second_metadata = render_shape(combo, return_metadata=True)
    assert first_metadata == second_metadata
    assert first_image.tobytes() == second_image.tobytes()


def test_model_and_symmetric_loss_smoke():
    combinations = all_combinations()
    train, _ = build_splits(HELD_OUT_PAIRS)
    tokenizer = build_tokenizer(combinations)
    loader = DataLoader(
        ShapeDataset(train),
        batch_size=len(train),
        collate_fn=make_collate_fn(tokenizer),
    )
    images, tokens, mask, captions, combos = next(iter(loader))
    model = TinyCLIP(tokenizer.vocab_size)
    logits = model(images, tokens, mask)
    loss = symmetric_clip_loss(logits)

    assert logits.shape == (42, 42)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert len(captions) == len(combos) == 42
