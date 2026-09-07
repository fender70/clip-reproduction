import torch
import torch.nn.functional as F


def symmetric_clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric image-to-text and text-to-image cross-entropy."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square [batch, batch] logits, got {tuple(logits.shape)}")

    targets = torch.arange(logits.shape[0], device=logits.device)
    image_to_text = F.cross_entropy(logits, targets)
    text_to_image = F.cross_entropy(logits.T, targets)
    return (image_to_text + text_to_image) / 2


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible feature-level CLIP loss used by the objective notebook."""
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * (image_features @ text_features.T)
    return symmetric_clip_loss(logits)
