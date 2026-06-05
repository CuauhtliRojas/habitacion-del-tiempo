"""
Demostración simple: checkpoint completo contra solo model.state_dict().

Compara:
- codigo Deepshield/train.py
- src/checkpoints.py
- train_orchestrator.py

Referencia:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeloSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def entrenar(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, pasos: int) -> list[float]:
    losses = []

    for _ in range(pasos):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def distancia(model_a: nn.Module, model_b: nn.Module) -> float:
    valores = []

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        valores.append((param_a - param_b).abs().max().item())

    return max(valores)


def main() -> None:
    torch.manual_seed(117)

    x = torch.randn(16, 4)
    y = torch.randn(16, 1)

    modelo_inicial = ModeloSimple()

    modelo_continuo = copy.deepcopy(modelo_inicial)
    opt_continuo = torch.optim.Adam(modelo_continuo.parameters(), lr=0.01)

    entrenar(modelo_continuo, opt_continuo, x, y, pasos=5)

    solo_pesos = copy.deepcopy(modelo_continuo.state_dict())

    checkpoint_completo = {
        "epoch": 5,
        "model_state": copy.deepcopy(modelo_continuo.state_dict()),
        "optimizer_state": copy.deepcopy(opt_continuo.state_dict()),
        "metrics_history": [{"epoch": 5}],
    }

    entrenar(modelo_continuo, opt_continuo, x, y, pasos=5)

    modelo_solo_pesos = ModeloSimple()
    modelo_solo_pesos.load_state_dict(solo_pesos)
    opt_solo_pesos = torch.optim.Adam(modelo_solo_pesos.parameters(), lr=0.01)
    entrenar(modelo_solo_pesos, opt_solo_pesos, x, y, pasos=5)

    modelo_resume = ModeloSimple()
    opt_resume = torch.optim.Adam(modelo_resume.parameters(), lr=0.01)
    modelo_resume.load_state_dict(checkpoint_completo["model_state"])
    opt_resume.load_state_dict(checkpoint_completo["optimizer_state"])
    entrenar(modelo_resume, opt_resume, x, y, pasos=5)

    dist_solo_pesos = distancia(modelo_continuo, modelo_solo_pesos)
    dist_resume = distancia(modelo_continuo, modelo_resume)

    print("=== Solo pesos vs checkpoint completo ===")
    print(f"Distancia usando solo model.state_dict(): {dist_solo_pesos:.12f}")
    print(f"Distancia usando checkpoint completo: {dist_resume:.12f}")
    print("")
    print("Conclusión:")
    print("Guardar solo pesos no reproduce igual la continuación del entrenamiento.")
    print("El checkpoint completo conserva el estado del optimizador y la época.")


if __name__ == "__main__":
    main()