import torch
import torch.nn.functional as F

def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:

    # 1. normalize image features
    image_features_norm = torch.nn.functional.normalize(image_features)
    
    # 2. normalize text features
    text_features_norm = torch.nn.functional.normalize(text_features)

    # 3. construct B x B similarity matrix
    similarity_it = image_features_norm @ text_features_norm.T
    similarity_ti = similarity_it.T

    # 4. construct targets
    batch_size = image_features.shape[0]
    targets = torch.arange(batch_size, device=image_features.device)

    # 5. image -> text cross entropy
    scale = logit_scale.exp()
    logits_it = scale * similarity_it
    ce_it = F.cross_entropy(logits_it, targets)
    
    # 6. text -> image cross entropy
    logits_ti = scale * similarity_ti
    ce_ti = F.cross_entropy(logits_ti, targets)

    # 7. symmetric average
    loss = (ce_it + ce_ti) / 2

    return loss
