from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from clip_repro.losses import symmetric_clip_loss
from clip_repro.models import TinyCLIP
from clip_repro.utils import set_seed


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embed_dim: int = 64
    token_dim: int = 32
    initial_temperature: float = 0.07


def retrieval_accuracy(logits: torch.Tensor) -> tuple[float, float]:
    targets = torch.arange(logits.shape[0], device=logits.device)
    image_to_text = (logits.argmax(dim=1) == targets).float().mean().item()
    text_to_image = (logits.argmax(dim=0) == targets).float().mean().item()
    return image_to_text, text_to_image


def train_model(
    tokenizer,
    train_loader: Iterable,
    device: str | torch.device,
    config: TrainConfig = TrainConfig(),
    *,
    seed: int | None = None,
    print_every: int | None = None,
) -> tuple[TinyCLIP, list[dict[str, float | int]]]:
    if seed is not None:
        set_seed(seed)

    model = TinyCLIP(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        token_dim=config.token_dim,
        initial_temperature=config.initial_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        model.train()
        loss_total = i2t_total = t2i_total = 0.0
        example_count = 0

        for images, tokens, mask, _, _ in train_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(images, tokens, mask)
            loss = symmetric_clip_loss(logits)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            i2t, t2i = retrieval_accuracy(logits.detach())
            loss_total += loss.item() * batch_size
            i2t_total += i2t * batch_size
            t2i_total += t2i * batch_size
            example_count += batch_size

        if example_count == 0:
            raise ValueError("The training loader produced no batches")

        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": loss_total / example_count,
            "i2t": i2t_total / example_count,
            "t2i": t2i_total / example_count,
            "temperature": model.temperature,
        }
        history.append(row)

        if print_every and (epoch == 0 or (epoch + 1) % print_every == 0):
            print(
                f"Epoch {epoch + 1:4d} | Loss {row['loss']:.4f} | "
                f"I2T {row['i2t']:.3f} | T2I {row['t2i']:.3f} | "
                f"tau {row['temperature']:.4f}"
            )

    return model, history
