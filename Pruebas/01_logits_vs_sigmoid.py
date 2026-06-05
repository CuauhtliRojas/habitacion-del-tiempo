"""
PRUEBA 1: Logits contra sigmoid dentro del modelo.

Objetivo:
Dejar evidencia de por qué se quitó sigmoid de la salida del modelo deepshield y se
pasó a logits crudos en src/model.py.

El bug real del código deepshield era de contrato numérico:
- codigo Deepshield/model.py devolvía probabilidades con sigmoid.
- codigo Deepshield/train.py usaba BCEWithLogitsLoss.
- BCEWithLogitsLoss espera logits crudos, no probabilidades.

Además, se muestra el riesgo matemático de trabajar con probabilidades saturadas
en FP16: si una probabilidad se redondea a 1.0 y luego se evalúa BCE como fórmula
manual, aparece log(0), es decir, infinito.

Referencia:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
"""

from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    target = torch.tensor([0.0], dtype=torch.float16)

    """
    Caso base:
    La red produce un logit alto positivo, pero el target real es 0.
    Esto representa una predicción muy segura, pero equivocada.
    """
    logit_deepshield = torch.tensor([15.0], dtype=torch.float16, requires_grad=True)
    logit_src = torch.tensor([15.0], dtype=torch.float16, requires_grad=True)

    """
    1. Caso deepshield exacto:
    El modelo aplica sigmoid antes de salir.
    Después, el entrenamiento manda esa probabilidad a BCEWithLogitsLoss.

    Esto no necesariamente produce inf, porque PyTorch protege internamente la
    operación, pero el contrato está mal: BCEWithLogitsLoss interpreta esa
    probabilidad como si todavía fuera un logit.
    """
    probabilidad_deepshield = torch.sigmoid(logit_deepshield)
    criterio = nn.BCEWithLogitsLoss()

    loss_deepshield_contract = criterio(probabilidad_deepshield, target)
    loss_deepshield_contract.backward()

    """
    2. Caso src correcto:
    El modelo devuelve el logit crudo.
    BCEWithLogitsLoss aplica sigmoid internamente de forma estable.
    """
    loss_src = criterio(logit_src, target)
    loss_src.backward()

    """
    3. Riesgo matemático:
    Si una probabilidad FP16 se redondea a 1.0 y se calcula BCE como fórmula
    manual, aparece log(1 - 1), equivalente a log(0).
    """
    probabilidad_saturada = torch.sigmoid(torch.tensor([15.0], dtype=torch.float16))
    loss_manual_inestable = -(
        target * torch.log(probabilidad_saturada)
        + (1.0 - target) * torch.log(1.0 - probabilidad_saturada)
    )

    print("\n=== PRUEBA 1: LOGITS VS SIGMOID EN EL MODELO ===")
    print(f"Target usado: {target.item()}")
    print(f"Logit crudo simulado: {logit_src.detach().item()}")
    print("")

    print("--- CASO deepshield EXACTO ---")
    print("Contrato usado: sigmoid(model_output) + BCEWithLogitsLoss")
    print(f"Salida tras sigmoid FP16: {probabilidad_deepshield.detach().item()}")
    print(f"Loss deepshield por contrato incorrecto: {loss_deepshield_contract.detach().item():.6f}")
    print(f"Gradiente que llega al logit deepshield: {logit_deepshield.grad.detach().item():.10f}")
    print("Interpretación: la pérdida no necesariamente explota, pero se alimenta con")
    print("probabilidades una función que esperaba logits crudos.")
    print("")

    print("--- CASO SRC CORRECTO ---")
    print("Contrato usado: logits crudos + BCEWithLogitsLoss")
    print(f"Loss src estable: {loss_src.detach().item():.6f}")
    print(f"Gradiente que llega al logit src: {logit_src.grad.detach().item():.10f}")
    print("Interpretación: PyTorch recibe el tipo de valor esperado y calcula la pérdida")
    print("con su forma estable interna.")
    print("")

    print("--- RIESGO MATEMÁTICO DE PROBABILIDAD SATURADA ---")
    print(f"Probabilidad saturada en FP16: {probabilidad_saturada.item()}")
    print(f"BCE manual sobre probabilidad saturada: {loss_manual_inestable.item()}")
    print("Interpretación: si se trabaja con probabilidades saturadas y se evalúa la")
    print("BCE como log-probabilidad, aparece log(0), es decir, pérdida infinita.")
    print("")

    print("=== CONCLUSIÓN ===")
    print("El cambio a logits crudos corrige el contrato entre modelo y pérdida.")
    print("La sigmoid se conserva solo donde corresponde: métricas y visualización.")
    print("Esto permite usar BCEWithLogitsLoss, AMP y entrenamiento a 512x512 con")
    print("menor riesgo numérico.")


if __name__ == "__main__":
    main()