import torch

from clip_repro.losses.clip_loss import clip_loss


def test_aligned_pairs_have_lower_loss_than_shuffled_pairs():
    image_features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    text_features = torch.tensor([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.8, 0.8, 0.1],
    ])

    logit_scale = torch.tensor(2.6592)

    aligned_loss = clip_loss(
        image_features,
        text_features,
        logit_scale,
    )

    shuffled_text_features = text_features[[2, 0, 3, 1]]

    shuffled_loss = clip_loss(
        image_features,
        shuffled_text_features,
        logit_scale,
    )

    assert aligned_loss < shuffled_loss
