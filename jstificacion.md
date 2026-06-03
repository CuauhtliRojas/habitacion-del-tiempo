# Justificación técnica de cambios del entrenamiento legacy al pipeline estable actual

## 1. Contexto general

El entrenamiento legacy partía de una implementación funcional pero poco controlada para experimentación reproducible. El flujo actual conserva la arquitectura conceptual de U-Net dual-decoder, pero corrige el contrato numérico del modelo, formaliza la configuración por YAML, documenta la distribución real de las máscaras, estabiliza el ciclo de entrenamiento y agrega artefactos de evaluación.

El cambio no se hizo por preferencia estética del código, sino por fallos observados durante el primer smoke test real: el entrenamiento inicial con `mixed_precision: true` llegó a producir `loss=nan`, lo cual invalidaba métricas, checkpoints y cualquier decisión posterior. Después de corregir el contrato de logits, la función de pérdida, la acumulación de gradientes, el clipping y el descarte de batches no finitos, el smoke estable completó una época completa con entrenamiento y validación sin batches saltados.

Resultado validado del smoke estable:

- Experimento: `faceswap_smoke_gpu_512_pw2_lr3e4_stable_e1_retry1`
- Épocas: 1
- Train batches físicos: 10503
- Val batches: 1167
- Optimizer steps: 2626
- Train skipped nonfinite batches: 0
- Val skipped nonfinite batches: 0
- Train loss total: 0.478652
- Val loss total: 0.380923
- Val Dice fake: 0.925402
- Val IoU fake: 0.864836
- Val precision fake: 0.907594
- Val recall fake: 0.949517
- Val accuracy fake: 0.950833

## 2. Cambio 1: salida del modelo de probabilidades a logits

### Legacy / estado anterior

El modelo usaba `torch.sigmoid(final(up4))` dentro de la rama decodificadora, por lo que devolvía probabilidades en rango `[0, 1]`.

Esto podía ser válido si la pérdida fuera `BCELoss` o una BCE manual sobre probabilidades, pero el legacy también usaba `BCEWithLogitsLoss`, que espera logits crudos. Por tanto, existía una inconsistencia conceptual: la red ya aplicaba sigmoid y luego la pérdida asumía que recibía logits.

### Estado actual

El modelo ahora devuelve logits crudos:

```python
return final(up4)
```

La conversión a probabilidad se hace únicamente cuando es necesaria:

Para Dice Loss.
Para IoU como métrica.
Para precision, recall, F1, accuracy.
Para generar máscaras binarias visuales.
Para overlays.
Justificación

PyTorch documenta que BCEWithLogitsLoss combina una capa Sigmoid y BCELoss en una sola operación, con mayor estabilidad numérica que usar Sigmoid seguida de BCE separada. Por ello, el contrato correcto es:

modelo -> logits
BCE -> BCEWithLogits
métricas/visualización -> sigmoid(logits)

Este cambio fue necesario para evitar inestabilidad numérica en entrenamiento con mixed precision.

3. Cambio 2: BCE manual a BCE estable con logits
   Legacy / estado anterior

La pérdida limpia anterior calculaba BCE manualmente:

pred = pred.clamp(min=EPS, max=1.0 - EPS)
positive_loss = -pos_weight _ target _ torch.log(pred)
negative_loss = -(1.0 - target) \* torch.log(1.0 - pred)

Este enfoque era frágil con mixed_precision: true, porque si pred se redondeaba a 1.0 en float16, entonces log(1.0 - pred) podía convertirse en log(0), generando inf o nan.

Estado actual

La BCE se calcula con:

F.binary_cross_entropy_with_logits(
logits,
target,
pos_weight=pos_weight_tensor,
)
Justificación

PyTorch recomienda la variante con logits por estabilidad numérica. Este cambio elimina la necesidad de aplicar log(pred) y log(1 - pred) manualmente, reduciendo el riesgo de NaN por saturación de sigmoid o precisión reducida.

4. Cambio 3: pos_weight de 65.0 a 2.1
   Legacy / estado anterior

El entrenamiento legacy usaba:

pos_weight = 65

Ese valor tendría sentido si la máscara positiva ocupara alrededor de 1.5% de la imagen:

positive_fraction ≈ 1 / (65 + 1) ≈ 0.01515
Diagnóstico real del dataset faceswap

El diagnóstico sobre las máscaras reales de faceswap mostró:

train_mean_fake_positive_ratio = 0.32529530
val_mean_fake_positive_ratio = 0.32552761
train_estimated_pos_weight = 2.074129905504
val_estimated_pos_weight = 2.071936066483

La máscara fake ocupa aproximadamente 32.5% de la imagen, no 1.5%.

Estado actual

Se usa:

pos_weight: 2.1
Justificación

La fórmula aplicada fue:

pos_weight ≈ pixeles_negativos / pixeles_positivos
pos_weight ≈ 0.675 / 0.325 ≈ 2.07

Por tanto, pos_weight: 2.1 está alineado con la distribución real de faceswap. Usar 65.0 habría sobrepenalizado los falsos negativos y empujado al modelo hacia máscaras infladas o aprendizaje inestable.

5. Cambio 4: eliminación de IoU de la pérdida
   Legacy / estado anterior

La pérdida sumaba tres términos por rama:

BCE + Dice + IoU

Esto hacía que la región de superposición recibiera doble presión porque Dice e IoU miden fenómenos muy relacionados: solapamiento entre predicción y máscara objetivo.

Estado actual

La pérdida final usa únicamente:

lambda_bce _ BCEWithLogits + lambda_dice _ Dice

IoU se conserva como métrica de evaluación, pero no forma parte de la pérdida.

Justificación

Dice e IoU son métricas de superposición altamente relacionadas. Incluir ambas en la pérdida puede sobrerrepresentar el componente regional frente al componente pixel-wise. Al dejar IoU como métrica, el entrenamiento conserva una señal regional por Dice y mantiene IoU como indicador externo de calidad.

6. Cambio 5: lambdas configurables para BCE y Dice
   Legacy / estado anterior

Los pesos eran implícitamente 1.0:

BCE + Dice + IoU

No había forma de modificar la importancia relativa de cada término desde YAML.

Estado actual

Se agregaron parámetros configurables:

lambda_bce: 1.0
lambda_dice: 2.0

La pérdida queda:

loss_fake = lambda_bce _ BCE_fake + lambda_dice _ Dice_fake
loss_authentic = lambda_bce _ BCE_authentic + lambda_dice _ Dice_authentic
loss_total = loss_fake + loss_authentic
Justificación

La segmentación de máscaras no solo requiere clasificar píxeles individualmente, sino maximizar la forma/región de la máscara. Dar mayor peso a Dice ayuda a priorizar coherencia espacial y solapamiento, especialmente con batch size pequeño y máscaras amplias.

El smoke estable confirmó que esta ponderación produjo un entrenamiento finito y métricas altas en validación.

7. Cambio 6: Gradient Accumulation
   Legacy / estado anterior

Con batch size físico 2, cada batch generaba un optimizer.step().

batch físico = 2
batch efectivo = 2
Estado actual

Se implementó:

gradient_accumulation_steps: 4

Por tanto:

batch físico = 2
batch efectivo aproximado = 2 \* 4 = 8

En la corrida estable:

train batches físicos = 10503
optimizer steps = 2626

Esto coincide con:

10503 / 4 ≈ 2626
Justificación

La GPU limita el batch físico a 2 por VRAM. La acumulación permite simular un batch efectivo mayor sin aumentar memoria, reduciendo ruido en actualizaciones y mejorando estabilidad.

8. Cambio 7: Gradient Clipping compatible con AMP
   Legacy / estado anterior

No había torch.nn.utils.clip*grad_norm*.

Estado actual

Se agregó:

max_grad_norm: 1.0

y el loop aplica clipping después de:

scaler.unscale\_(optimizer)

antes de:

scaler.step(optimizer)
Justificación

Con AMP, los gradientes están escalados antes del step. PyTorch indica que, para hacer gradient clipping con GradScaler, primero deben desescalarse los gradientes mediante scaler.unscale*(optimizer) y después aplicar clip_grad_norm*. Esto evita recortar gradientes artificialmente escalados.

El valor max_grad_norm: 1.0 es conservador y apropiado para estabilizar un entrenamiento inicial.

9. Cambio 8: descarte de batches con loss no finita
   Legacy / estado anterior

Si un batch producía loss=nan o loss=inf, ese valor podía entrar al promedio de la época y contaminar metrics.csv.

Estado actual

El loop revisa si la pérdida es finita. Si no lo es:

No hace backward.
No hace optimizer step.
Incrementa skipped_nonfinite_batches.
No agrega esa pérdida al promedio de época.
Si todos los batches de una época fallaran, el entrenamiento falla explícitamente.
Justificación

GradScaler puede saltar un step si detecta gradientes no finitos, pero el registro manual de métricas también debe protegerse. De lo contrario, metrics.csv puede quedar contaminado aunque los pesos del modelo no se actualicen con ese batch.

En el smoke estable:

train_skipped_nonfinite_batches = 0
val_skipped_nonfinite_batches = 0

Esto indica que la protección existe, pero no fue necesario activarla en la corrida estable.

10. Cambio 9: binarización estricta de máscaras
    Legacy / estado anterior

El legacy usaba transformaciones tipo ToTensor() sobre máscaras. Si una máscara tenía valores intermedios, podían entrar al entrenamiento como valores continuos.

Estado actual

El dataset limpio binariza explícitamente las máscaras:

valor > 0.5 -> 1.0
valor <= 0.5 -> 0.0

También usa interpolación apropiada para máscaras al redimensionar, evitando generar grises intermedios por interpolación bilineal.

Justificación

Para segmentación binaria, las máscaras objetivo deben contener exclusivamente 0.0 y 1.0. Esto evita que la pérdida aprenda valores ambiguos como 0.5 o 128/255.

11. Cambio 10: artefactos reproducibles del experimento
    Legacy / estado anterior

El legacy guardaba pesos, pero no dejaba suficiente trazabilidad de:

Configuración resuelta.
Historial de métricas.
Curvas.
Manifest del entorno.
Comando de reanudación.
Samples visuales por época.
Estado actual

El entrenamiento genera:

config.resolved.json
run_manifest.json
metrics/metrics.csv
plots/
samples/epoch_XXX/
checkpoints/last.pth
checkpoints/epoch_XXX.pth
README_RESUME.md
fit_report.txt
Justificación

Esto permite auditar el experimento, reanudarlo, comparar métricas, revisar samples visuales y tomar decisiones con evidencia.

12. Evaluación del smoke estable

El smoke estable de una época fue aprobado técnicamente.

Resultados principales:

train_loss_total = 0.478652
val_loss_total = 0.380923

val_dice_fake = 0.925402
val_iou_fake = 0.864836
val_precision = 0.907594
val_recall = 0.949517
val_f1 = 0.925402
val_accuracy = 0.950833

train_skipped_nonfinite_batches = 0
val_skipped_nonfinite_batches = 0
optimizer_steps = 2626

Interpretación:

La pérdida ya no colapsó a NaN.
La validación se completó.
Las métricas fake son altas.
La rama authentic quedó trivial en validación porque las máscaras authentic parecen completamente negras o fáciles para este split.
Una sola época no permite diagnosticar overfitting o underfitting.
Se requiere revisar visualmente samples/epoch_001 antes de aprobar un entrenamiento largo. 13. Riesgos pendientes
13.1 Riesgo de métricas demasiado altas en una época

Un val_dice_fake de 0.9254 en una sola época puede ser positivo, pero también obliga a revisar:

Si las máscaras faceswap son muy consistentes.
Si el modelo está aprendiendo una forma promedio demasiado general.
Si existe fuga accidental de datos.
Si las predicciones están infladas.
Si pred_fake.png se parece realmente a gt_fake.png.
13.2 Checkpoint con best_val_loss y best_val_dice en None

La inspección del checkpoint mostró:

Best val loss: None
Best val dice: None

Aunque la fila de métricas existe, esto indica que el checkpoint last.pth no está persistiendo los mejores valores en esa inspección. Antes del entrenamiento serio conviene revisar el orden en que se actualizan best_val_loss y best_val_dice contra el guardado del checkpoint.

Esto no invalida el smoke, pero sí debe corregirse antes de confiar plenamente en best_val_loss.pth y best_val_dice.pth.

14. Parámetros recomendados para el siguiente entrenamiento

Se recomienda iniciar con 15 épocas, no 20, por las siguientes razones:

El smoke ya mostró métricas altas en una sola época.
No hay suficientes épocas para ver si el modelo se sobreajusta.
Cada época tarda aproximadamente 47 minutos.
Un entrenamiento de 15 épocas tomará aproximadamente 12 horas.
Un entrenamiento de 20 épocas tomará aproximadamente 16 horas.
Con checkpoints por época, si a la época 15 todavía mejora, se puede extender a 20 mediante resume.
