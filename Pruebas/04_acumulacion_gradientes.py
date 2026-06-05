"""
Demostración simple: batch grande contra acumulación de gradientes.

Compara:
- codigo Deepshield/train.py
- train_orchestrator.py

Referencia:
https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeloSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def diferencia_maxima(model_a: nn.Module, model_b: nn.Module) -> float:
    diferencias = []

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        diferencias.append((param_a - param_b).abs().max().item())

    return max(diferencias)


def main() -> None:
    torch.manual_seed(117)

    x = torch.randn(8, 8)
    y = torch.rand(8, 1)

    modelo_base = ModeloSimple()

    modelo_batch_grande = copy.deepcopy(modelo_base)
    modelo_acumulado = copy.deepcopy(modelo_base)

    opt_grande = torch.optim.SGD(modelo_batch_grande.parameters(), lr=0.1)
    opt_acum = torch.optim.SGD(modelo_acumulado.parameters(), lr=0.1)

    opt_grande.zero_grad(set_to_none=True)
    pred = modelo_batch_grande(x)
    loss_grande = F.binary_cross_entropy_with_logits(pred, y)
    loss_grande.backward()
    opt_grande.step()

    opt_acum.zero_grad(set_to_none=True)

    pasos_acumulacion = 4
    tam_micro_batch = 2

    for inicio in range(0, 8, tam_micro_batch):
        fin = inicio + tam_micro_batch
        pred_micro = modelo_acumulado(x[inicio:fin])
        loss_micro = F.binary_cross_entropy_with_logits(pred_micro, y[inicio:fin])
        loss_micro = loss_micro / pasos_acumulacion
        loss_micro.backward()

    opt_acum.step()

    diferencia = diferencia_maxima(modelo_batch_grande, modelo_acumulado)

    print("=== Batch grande vs acumulación de gradientes ===")
    print(f"Loss batch grande: {loss_grande.item():.8f}")
    print(f"Diferencia máxima entre pesos: {diferencia:.12f}")
    print("")
    print("Conclusión:")
    print("Dividir la pérdida entre pasos de acumulación permite simular un batch mayor.")
    print("Esto ayuda cuando la VRAM no permite usar batch grande directamente.")


if __name__ == "__main__":
    main()