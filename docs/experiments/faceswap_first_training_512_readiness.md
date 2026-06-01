# Readiness experimental: faceswap_first_training_512

## 1. Objetivo

Este documento registra el análisis previo al primer entrenamiento real del experimento `faceswap_first_training_512`.

El objetivo es dejar trazabilidad técnica sobre la configuración inicial, los diagnósticos ejecutados, los riesgos detectados y las decisiones pendientes antes de iniciar entrenamiento con GPU. Este documento no autoriza entrenamiento real por sí mismo.

## 2. Contexto del experimento

Repositorio local:

```text
C:\Users\cuauh\Desktop\camara-de-entrenamiento-100G
```

### Configuración base evaluada:

```
configs/faceswap_first_training.yaml
```

### Experimento configurado:

```text

experiment_name: faceswap_first_training_512
attacks: [faceswap]
image_size: 512
batch_size: 2
epochs: 20
learning_rate: 0.0005
pos_weight: 65.0
val_ratio: 0.1
threshold: 0.5
mixed_precision: true
checkpoint_every: 1
sample_every: 1
```

## 3. Validaciones previas realizadas

Se validó el entorno local con:

```
Python 3.12.10
PyTorch 2.6.0+cu124
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Se validó el dataset:

```
faceswap/train: 23,338 muestras completas
faceswap/test: 2,593 muestras completas
local_inpainting/train: 22,501 muestras completas
local_inpainting/test: 2,499 muestras completas
total: 50,931 muestras completas
total PNG: 152,793
```

Se ejecutó dry-run de configs/faceswap_first_training.yaml sin entrenamiento real:

```
Train samples: 21,005
Val samples: 2,333
Train batches/epoch aprox: 10,503
Val batches/epoch aprox: 1,167
Total train batches aprox: 210,060
Device: dry-run/cpu
```

## 4. Artefactos de diagnóstico generados

El script de diagnóstico pre-entrenamiento genera:

```
outputs/diagnostics/faceswap_first_training_512/mask_distribution.csv
outputs/diagnostics/faceswap_first_training_512/mask_distribution_summary.json
outputs/diagnostics/faceswap_first_training_512/training_readiness_recommendation.json
outputs/diagnostics/faceswap_first_training_512/training_readiness_report.txt
outputs/diagnostics/faceswap_first_training_512/diagnostic_config_snapshot.json
```

Estos artefactos permiten revisar distribución de máscaras, proporción de píxeles positivos, estimación de pos_weight, conteos por split y estimaciones de batches.

## 5. Hallazgo principal: desbalance real de máscara fake

El diagnóstico detectó que las máscaras fake de faceswap ocupan una proporción alta de la imagen:

```
train/faceswap fake positive ratio mean: 0.32529530
train/faceswap fake positive ratio median: 0.32559204
train/faceswap fake p05/p50/p95: 0.23140259 / 0.32559204 / 0.41926270
val/faceswap fake positive ratio mean: 0.32552761
val/faceswap fake positive ratio median: 0.32504654
```

La estimación de pos_weight con base en negativos/positivos fue:

```
train/faceswap estimated pos_weight: 2.074129905503714
val/faceswap estimated pos_weight: 2.071936066482757
```

## 6. Implicación sobre pos_weight

La configuración base usa:

```
pos_weight: 65.0
```

El diagnóstico sugiere que este valor es demasiado alto para la distribución real de faceswap.

Justificación:

```
Si la máscara fake ocupa aproximadamente 32.5% de los píxeles, entonces los negativos representan aproximadamente 67.5%.
La relación negativos/positivos es cercana a 2.07, no a 65.
```

Riesgo técnico de mantener pos_weight: 65.0:

```
- sobresegmentación;
- aumento artificial de recall con caída de precision;
- máscaras predichas demasiado grandes;
- falsos positivos sobre regiones no manipuladas;
- dificultad para interpretar accuracy;
- pérdida dominada por positivos de manera desproporcionada.
```

Decisión preliminar:

```
No iniciar el primer entrenamiento serio con pos_weight: 65.0.
Evaluar una configuración corregida cerca de pos_weight: 2.0 o 2.1.
```

## 7. Métricas de análisis

La accuracy no será métrica principal porque en segmentación binaria puede ser engañosa si existe predominio de fondo o máscaras grandes.

Métricas principales para la máscara fake:

```
val_dice_fake
val_iou_fake
val_f1_fake
val_precision_fake
val_recall_fake
```

Métrica secundaria:

```
val_accuracy_fake
```

Métricas visuales obligatorias:

```
image.png
gt_fake.png
pred_fake.png
overlay_fake.png
gt_authentic.png
pred_authentic.png
```

## 8. Registro ampliado de métricas

Se amplió el registro de metrics.csv para incluir:

```

train_precision_fake
train_recall_fake
train_f1_fake
train_accuracy_fake
train_dice_authentic
train_iou_authentic
val_accuracy_fake
val_dice_authentic
val_iou_authentic
```

Además, se ampliaron las gráficas para analizar:

```

loss_authentic_curve.png
precision_fake_curve.png
recall_fake_curve.png
f1_fake_curve.png
accuracy_fake_curve.png
dice_authentic_curve.png
iou_authentic_curve.png
```

## 9. Riesgo computacional

Con image_size: 512, batch_size: 2 y 20 épocas:

```
train batches/epoch aprox: 10,503
total train updates aprox: 210,060
```

Riesgos:

```
- tiempo largo de entrenamiento;
- posible OOM en GPU de 6 GB si el modelo o activaciones crecen;
- necesidad de checkpoints por época;
- necesidad de reanudación segura;
- necesidad de revisar muestras visuales desde las primeras épocas.
```

Plan ante OOM:

```

1. Reducir batch_size físico de 2 a 1.
2. Mantener image_size 512 si es posible.
3. No bajar a 256 salvo que batch_size 1 también falle o el tiempo sea inviable.
```

## 10. Configuraciones pendientes

Antes de entrenamiento real se recomienda crear al menos dos configuraciones separadas:

## Smoke test GPU

Objetivo: validar entrenamiento real corto, checkpoints, samples, métricas y reanudación.

Configuración sugerida:

```
experiment_name: faceswap_smoke_gpu_512_pw2_e1
image_size: 512
batch_size: 2
epochs: 1
learning_rate: 0.0003
pos_weight: 2.1
mixed_precision: true
checkpoint_every: 1
sample_every: 1
```

## Baseline serio

Objetivo: primer entrenamiento real útil de faceswap.

Configuración sugerida inicial:

```
experiment_name: faceswap_first_training_512_pw2_lr3e4
image_size: 512
batch_size: 2
epochs: 20
learning_rate: 0.0003
pos_weight: 2.1
mixed_precision: true
checkpoint_every: 1
sample_every: 1
```

## 11. Criterios de parada y revisión

Durante el entrenamiento, revisar por época:

```
- si train_loss baja pero val_loss sube;
- si train_dice_fake sube y val_dice_fake se estanca;
- si precision cae y recall sube excesivamente;
- si las muestras visuales muestran máscaras infladas;
- si las predicciones salen casi negras;
- si aparecen NaN o pérdidas inestables.
```

Candidatos a parada temprana manual:

```
- val_dice_fake no mejora durante varias épocas;
- val_loss empeora de forma sostenida mientras train_loss mejora;
- sobresegmentación visual persistente;
- métricas inconsistentes con las muestras visuales.
```

## 12. Pendientes antes de entrenar

Antes de ejecutar entrenamiento real:

```

1. Crear YAML de smoke test.
2. Crear YAML de baseline corregido.
3. Ejecutar dry-run de ambos YAML.
4. Confirmar git status limpio.
5. Confirmar output path exacto.
6. Confirmar plan de resume desde checkpoints/last.pth.
7. Confirmar plan ante OOM.
8. Autorizar explícitamente inicio de entrenamiento.
```

## 13. Conclusión

El diagnóstico pre-entrenamiento cambió la decisión inicial sobre `pos_weight`.

La configuración original `pos_weight: 65.0`
no está justificada para faceswap con máscaras fake que ocupan aproximadamente 32.5% de la imagen. La primera prueba real debe usar un valor cercano a `2.1`
, salvo que una revisión externa o una prueba controlada indique lo contrario.

No se debe iniciar entrenamiento real hasta cerrar los YAML de smoke test y baseline corregido, validar ambos con dry-run y confirmar manualmente la ejecución.
