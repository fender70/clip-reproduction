from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Keep the 8 x 8 feature map so horizontal position remains available.
        self.projection = nn.Linear(64 * 8 * 8, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(images).flatten(start_dim=1))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, token_dim: int = 32, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.projection = nn.Linear(token_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(tokens)
        mask = mask.unsqueeze(-1)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.projection(pooled)


class TinyCLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        token_dim: int = 32,
        initial_temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            token_dim=token_dim,
            embed_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))

    @property
    def temperature(self) -> float:
        return 1.0 / self.logit_scale.exp().item()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(tokens, mask), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(tokens, mask)
        return self.logit_scale.exp() * (image_embeddings @ text_embeddings.T)
