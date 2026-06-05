"""
Demostración simple: memoria con FP32 contra precisión mixta.

Compara:
- codigo Deepshield/train.py
- train_orchestrator.py

Referencia:
https://docs.pytorch.org/docs/stable/amp.html
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RedPequena(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def medir_memoria(use_amp: bool) -> float:
    device = torch.device("cuda")
    model = RedPequena().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(4, 3, 512, 512, device=device)
    y = torch.rand(4, 2, 512, 512, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=use_amp):
        pred = model(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA no disponible. Esta prueba requiere GPU.")
        return

    memoria_fp32 = medir_memoria(use_amp=False)
    memoria_amp = medir_memoria(use_amp=True)

    reduccion = 100.0 * (memoria_fp32 - memoria_amp) / memoria_fp32

    print("=== FP32 vs AMP ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria FP32: {memoria_fp32:.2f} MB")
    print(f"Memoria AMP: {memoria_amp:.2f} MB")
    print(f"Reducción aproximada: {reduccion:.2f}%")
    print("")
    print("Conclusión:")
    print("La precisión mixta reduce uso de VRAM y ayuda a entrenar a 512x512.")


if __name__ == "__main__":
    main()