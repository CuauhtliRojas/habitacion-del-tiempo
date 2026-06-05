import torch
import torch.nn as nn

"""

Este script justifica el cambio arquitectónico entre los archivos antiguos 
(codigo Deepshield/model.py y train.py) y el nuevo ecosistema (src/model.py y 
train_orchestrator.py)

En la version codigo Deepshield\model.py la capa final de la red aplicaba lo siguiente:

    return torch.sigmoid(final(up4))

Al pasar esa salida al codigo Deepshield/train.py en la línea 56, se usa
esta función para calcular el error:

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([65]).to(device))

El problema:
La función `BCEWithLogitsLoss` ya trae un Sigmoide integrado por diseño.
Al ponerle otro Sigmoide en el modelo, terminábamos aplicándolo DOS VECES. 
Esto aplastaba los números de una forma tan brutal que la red se quedaba "ciega", 
los gradientes se hacían cero y dejaba de aprender 
(Vanishing Gradient Problem:ocurre en redes neuronales profundas cuando los 
gradientes se vuelven extremadamente pequeños durante la retropropagación).

Solución actual:
El modelo ahora escupe "Logits" (números crudos, directos de la convolución).
Estos van directo a BCEWithLogitsLoss, que usa un truco matemático (Log-Sum-Exp) 
para fusionar las operaciones sin explotar la memoria.

https://docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html

"""
print("\n--- Simulacion de Doble Sigmoide ---")

"""
Imaginemos un píxel donde la red asigna que NO es deepfake.
El logit crudo es un número muy negativo (ej. -100.0). 
"""

logit_crudo = torch.tensor([-100.0], dtype=torch.float16, requires_grad=True)

"""(Etiqueta real): Supongamos que la red se equivocó y SÍ era deepfake."""

etiqueta_real = torch.tensor([1.0], dtype=torch.float16)

# 3. El error en el código Legacy: Aplicar Sigmoide en el modelo
#    Por culpa del float16, la computadora no puede manejar decimales tan 
#    pequeños y redondea el resultado a EXACTAMENTE 0.0
probabilidad_distorsionada = torch.sigmoid(logit_crudo)
print(f"1. Sigmoide en el modelo (Aplastado) : {probabilidad_distorsionada.item()}")

# 4. Tratar de calcular el Loss con BCELoss clásico usando ese 0.0
#    Matemáticamente intenta hacer log(0), lo cual es Infinito.
criterio_legacy = nn.BCELoss()
loss_legacy = criterio_legacy(probabilidad_distorsionada, etiqueta_real)
print(f"2. Resultado del Loss Legacy         : {loss_legacy.item()} <- ¡AQUÍ EXPLOTA EL ENTRENAMIENTO!")


print("\n--- SIMULACIÓN DEL CÓDIGO NUEVO (Logits crudos) ---")

# 1. Usamos el mismo logit crudo.
logit_crudo_nuevo = torch.tensor([-100.0], dtype=torch.float16, requires_grad=True)

# 2. BCEWithLogitsLoss recibe el número intacto (Logit).
#    Aplica el truco Log-Sum-Exp por debajo para cancelar términos y 
#    evitar el cálculo de log(0).
criterio_nuevo = nn.BCEWithLogitsLoss()
loss_nuevo = criterio_nuevo(logit_crudo_nuevo, etiqueta_real)

print(f"1. El modelo escupe el Logit directo : {logit_crudo_nuevo.item()}")
print(f"2. Resultado del Loss Nuevo          : {loss_nuevo.item():.4f} <- ¡NÚMERO VÁLIDO Y ESTABLE!")

# 3. Demostración de que la red SÍ puede aprender.
#    Calculamos el gradiente (el ajuste matemático para que la red mejore).
loss_nuevo.backward()
print(f"3. Gradiente que viaja a la red      : {logit_crudo_nuevo.grad.item():.4f} <- ¡EL ENTRENAMIENTO CONTINÚA!")
print("----------------------------------------------------\n")