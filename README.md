# 🐉 Habitación del Tiempo (Cámara de Entrenamiento 100G)

Guía operativa para preparar, validar, entrenar, reanudar y evaluar experimentos de segmentación semántica binaria sobre manipulaciones faciales deepfake.

Este repositorio entrena una arquitectura **U-Net Dual-Decoder** para localizar regiones manipuladas en imágenes faciales. Actualmente se trabaja con dos tipos de ataque:

- `faceswap`
- `local_inpainting`

El objetivo principal es predecir una **máscara fake** que indique la región manipulada. La arquitectura también predice una **máscara auténtica**, pero el análisis principal del proyecto se centra en la localización de la región alterada.

---

## 1. Requisitos locales

Entorno esperado:

```text
Python 3.12
uv
PyTorch con CUDA
GPU NVIDIA con CUDA disponible
Dataset local en Dataset_U-Net_dual_decoder_fs_li/
```

### Validar GPU

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Validar sintaxis general

```powershell
uv run python -m py_compile `
  .\train_orchestrator.py `
  .\src\model.py `
  .\src\dataset.py `
  .\src\metrics.py `
  .\src\checkpoints.py `
  .\src\experiment_manifest.py `
  .\src\plots.py `
  .\src\visualization.py `
  .\scripts\analyze_local_inpainting_pos_weight.py `
  .\scripts\analyze_training_readiness.py `
  .\scripts\check_dataset_counts.py `
  .\scripts\check_model_checkpoint_contract.py `
  .\scripts\inspect_checkpoint.py `
  .\scripts\recompute_experiment_artifacts.py `
  .\scripts\write_resume_notes.py
```

Si no imprime errores, la sintaxis básica está correcta.

## 2. Estructura esperada del dataset

El dataset local debe tener esta estructura:

```text
Dataset_U-Net_dual_decoder_fs_li/
├── faceswap/
│   ├── train/
│   │   ├── imagen_original/
│   │   ├── mascara_autentica/
│   │   ├── mascara_fake/
│   │   └── metadata.csv
│   └── test/
│       ├── imagen_original/
│       ├── mascara_autentica/
│       ├── mascara_fake/
│       └── metadata.csv
└── local_inpainting/
    ├── train/
    │   ├── imagen_original/
    │   ├── mascara_autentica/
    │   ├── mascara_fake/
    │   └── metadata.csv
    └── test/
        ├── imagen_original/
        ├── mascara_autentica/
        ├── mascara_fake/
        └── metadata.csv
```

El dataset no se versiona en Git. Debe permanecer ignorado por `.gitignore`.

Cada muestra completa debe tener:

```text
imagen_original
mascara_autentica
mascara_fake
```

Validar conteos:

```powershell
uv run python .\scripts\check_dataset_counts.py
```

Conteos conocidos del dataset local:

```text
faceswap/train: 23,338 muestras
faceswap/test: 2,593 muestras

local_inpainting/train: 22,501 muestras
local_inpainting/test: 2,499 muestras

Total: 50,931 muestras completas
Total PNG aproximado: 152,793 archivos
```

## 3. Componentes principales del repositorio

### `src/model.py`

Define la arquitectura DualSegmentationModel.

El modelo devuelve logits, no probabilidades. La sigmoid se aplica después en métricas y visualización. Esto permite usar BCE con logits de forma estable.

### `src/dataset.py`

Lee imágenes y máscaras desde el dataset estructurado por ataque y split.

Aplica:

- `BILINEAR` para redimensionar imágenes.
- `NEAREST` para redimensionar máscaras.
- binarización explícita de máscaras.

### `src/metrics.py`

Contiene:

- conversión de logits a probabilidades;
- Dice;
- IoU;
- precision;
- recall;
- F1;
- accuracy;
- BCE con logits;
- Dice Loss;
- pérdida total dual.

La pérdida actual combina:

```text
BCE con logits + Dice Loss
```

IoU se reporta como métrica, pero no se suma a la pérdida principal.

### `src/checkpoints.py`

Guarda y carga checkpoints completos:

```text
model_state
optimizer_state
config
metrics_history
best_val_loss
best_val_dice
scaler_state, si se usa mixed precision
```

### `src/plots.py`

Genera gráficas desde metrics.csv.

### `src/visualization.py`

Guarda muestras visuales por época:

```text
image.png
gt_fake.png
pred_fake.png
gt_authentic.png
pred_authentic.png
overlay_fake.png
```

### `train_orchestrator.py`

Orquesta el entrenamiento completo desde un archivo YAML.

## 4. Flujo básico de trabajo

El flujo recomendado siempre es:

1. Crear o elegir YAML.
2. Ejecutar dry-run.
3. Revisar preflight.
4. Entrenar con --yes.
5. Inspeccionar checkpoint.
6. Regenerar gráficas/reporte si hace falta.
7. Leer metrics.csv y fit_report.txt.
8. Revisar samples visuales.
9. Elegir best_val_dice.pth o best_val_loss.pth según el objetivo.

## 5. Convención para nombrar archivos YAML

Usar nombres descriptivos. El nombre debe decir:

```text
ataque
tipo de experimento
resolución
pos_weight
learning rate
estabilidad/configuración
épocas
batch size, si es relevante
```

Ejemplos:

```text
faceswap_training_512_pw2_lr3e4_stable_e15.yaml
faceswap_benchmark_256_pw2_lr3e4_stable_e20_bs16.yaml
local_inpainting_training_512_pw33_lr3e4_stable_e15_bs4.yaml
local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6.yaml
```

Lectura rápida:

```text
faceswap_training_512_pw2_lr3e4_stable_e15
```

Significa:

```text
ataque: faceswap
modo: training
resolución: 512
pos_weight: 2 aproximadamente
learning rate: 3e-4
configuración estable
épocas: 15
```

```text
local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6
```

Significa:

```text
ataque: local_inpainting
modo: fine-tuning
resolución: 512
origen: época 6
pos_weight: 15
learning rate: 5e-5
épocas: 6
```

## 6. Campos principales de un YAML

Ejemplo base:

```yaml
experiment_name: faceswap_training_512_pw2_lr3e4_stable_e15
dataset_root: Dataset_U-Net_dual_decoder_fs_li
output_root: outputs/experiments

attacks:
  - faceswap

train_split: train
test_split: test
image_size: 512
batch_size: 2
epochs: 15
learning_rate: 0.0003
pos_weight: 2.1
lambda_bce: 1.0
lambda_dice: 2.0
gradient_accumulation_steps: 4
max_grad_norm: 1.0
val_ratio: 0.1
seed: 117
num_workers: 0
threshold: 0.5
checkpoint_every: 1
sample_every: 1
mixed_precision: true
```

Campos importantes:

### `experiment_name`

Nombre de la carpeta que se creará dentro de outputs/experiments/.

### `dataset_root`

Ruta al dataset local.

### `attacks`

Lista de ataques a entrenar. Puede ser uno o varios.

Ejemplo individual:

```yaml
attacks:
  - faceswap
```

Ejemplo mixto:

```yaml
attacks:
  - faceswap
  - local_inpainting
```

### `image_size`

Resolución cuadrada de entrada. Valores usados:

```text
256
512
```

### `batch_size`

Cantidad de muestras por batch físico.

### `gradient_accumulation_steps`

Cantidad de batches físicos acumulados antes de actualizar pesos.

Batch efectivo:

```text
batch_size * gradient_accumulation_steps
```

Ejemplos:

```text
batch_size 2 * accumulation 4 = batch efectivo 8
batch_size 4 * accumulation 2 = batch efectivo 8
batch_size 16 * accumulation 1 = batch efectivo 16
```

### `pos_weight`

Peso para píxeles positivos en BCE. No debe elegirse al azar. Debe estimarse según el porcentaje real de área blanca en las máscaras.

Valores usados hasta ahora:

```text
faceswap: 2.1
local_inpainting: 33.4754694047
fine-tuning local_inpainting: 15.0
```

### `lambda_bce` y `lambda_dice`

Pesos de la pérdida compuesta.

Configuración estable usada:

```yaml
lambda_bce: 1.0
lambda_dice: 2.0
```

### `mixed_precision`

Debe estar en true para aprovechar mejor GPU y memoria.

### `max_grad_norm`

Límite de clipping de gradientes. Configuración estable:

```yaml
max_grad_norm: 1.0
```

## 7. Validar un YAML antes de entrenar

Siempre ejecutar --dry-run antes de un entrenamiento real:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\<archivo>.yaml `
  --dry-run
```

El dry-run debe imprimir:

```text
Experimento
Dataset
Ataques
Image size
Batch size
Epochs
Train samples
Val samples
Train batches/epoch
Val batches/epoch
Device
Advertencias, si aplica
Distribución train
Distribución val
```

El dry-run no entrena, no guarda checkpoints y no modifica pesos.

## 8. Ejecutar entrenamiento desde cero

Ejemplo:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\faceswap_training_512_pw2_lr3e4_stable_e15.yaml `
  --yes
```

Durante el entrenamiento revisar:

```text
loss
skipped
steps
```

Señales sanas:

- skipped=0
- loss finita
- val_loss finita
- val_dice_fake subiendo o estable

Señales de alerta:

- loss=nan
- loss=inf
- skipped subiendo constantemente
- CUDA out of memory
- val_loss sube mientras train_loss baja durante varias épocas
- precision muy baja con recall muy alto

## 9. Fine-tuning

Fine-tuning significa cargar pesos ya aprendidos, pero arrancar con una nueva estrategia de entrenamiento.

Usar weights, no resume.

Ejemplo:

```yaml
experiment_name: local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6
dataset_root: Dataset_U-Net_dual_decoder_fs_li
output_root: outputs/experiments

attacks:
  - local_inpainting

train_split: train
test_split: test
image_size: 512
batch_size: 4
gradient_accumulation_steps: 2
epochs: 6
learning_rate: 0.00005
pos_weight: 15.0
lambda_bce: 1.0
lambda_dice: 2.0
max_grad_norm: 1.0
val_ratio: 0.1
seed: 117
num_workers: 0
threshold: 0.5
checkpoint_every: 1
sample_every: 1
mixed_precision: true

weights: outputs/experiments/local_inpainting_training_512_pw33_lr3e4_stable_e15_bs4/checkpoints/epoch_006.pth
```

Ejecutar:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6.yaml `
  --yes
```

Usar fine-tuning cuando:

- el modelo ya aprendió a encontrar la región;
- el recall es alto;
- la precision es baja;
- el modelo sobrepinta;
- se quiere bajar learning_rate;
- se quiere reducir pos_weight;
- se quiere limpiar bordes sin empezar desde cero.

## 10. Resume

Resume significa continuar el mismo experimento desde un checkpoint completo.

Usar --resume.

Ejemplo:

```powershell
uv run python .\train_orchestrator.py `
  --config .\outputs\experiments\<experiment_name>\config.resolved.json `
  --resume .\outputs\experiments\<experiment_name>\checkpoints\last.pth
```

Diferencia importante:

- weights = carga solo pesos del modelo; sirve para fine-tuning.
- resume = carga modelo + optimizer + scaler + historial; sirve para continuar el mismo entrenamiento.

## 11. Analizar pos_weight

Para local_inpainting, calcular el área blanca real:

```powershell
uv run python .\scripts\analyze_local_inpainting_pos_weight.py `
  --mask-dir .\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\train\mascara_fake `
  --workers 4
```

Salida relevante:

```text
Global positive fraction
Global negative fraction
Suggested pos_weight negative/positive
```

Resultado obtenido para local_inpainting:

```text
Global positive fraction: 0.0290061315
Global negative fraction: 0.9709938685
Suggested pos_weight: 33.4754694047
```

Interpretación:

- solo 2.9% de los píxeles son positivos;
- la clase positiva es pequeña;
- por eso el pos_weight matemático es alto.

Sin embargo, un pos_weight muy alto puede provocar que el modelo sobrepinte. En ese caso, se puede hacer fine-tuning con un valor menor, por ejemplo:

```yaml
pos_weight: 15.0
```

## 12. Artefactos generados por experimento

Cada entrenamiento crea:

```text
outputs/experiments/<experiment_name>/
├── config.resolved.json
├── run_manifest.json
├── fit_report.txt
├── checkpoints/
│   ├── last.pth
│   ├── best_val_loss.pth
│   ├── best_val_dice.pth
│   └── epoch_XXX.pth
├── metrics/
│   └── metrics.csv
├── plots/
│   ├── loss_total_curve.png
│   ├── loss_fake_curve.png
│   ├── loss_authentic_curve.png
│   ├── dice_fake_curve.png
│   ├── iou_fake_curve.png
│   ├── precision_fake_curve.png
│   ├── recall_fake_curve.png
│   ├── f1_fake_curve.png
│   ├── accuracy_fake_curve.png
│   ├── precision_recall_f1_curve.png
│   ├── dice_authentic_curve.png
│   ├── iou_authentic_curve.png
│   └── overfit_loss_gap.png
└── samples/
    └── epoch_XXX/
        └── <sample_id>/
            ├── image.png
            ├── gt_fake.png
            ├── pred_fake.png
            ├── gt_authentic.png
            ├── pred_authentic.png
            └── overlay_fake.png
```

## 13. Inspeccionar checkpoint

```powershell
uv run python .\scripts\inspect_checkpoint.py `
  .\outputs\experiments\<experiment_name>\checkpoints\last.pth
```

Este comando muestra:

```text
epoch
best_val_loss
best_val_dice
config
última fila de métricas
comando conceptual de resume
```

## 14. Regenerar gráficas y reporte sin entrenar

```powershell
uv run python .\scripts\recompute_experiment_artifacts.py `
  .\outputs\experiments\<experiment_name>
```

Esto vuelve a generar:

```text
plots/
fit_report.txt
```

a partir de metrics.csv.

## 15. Generar notas de reanudación

```powershell
uv run python .\scripts\write_resume_notes.py `
  .\outputs\experiments\<experiment_name>
```

Genera:

```text
README_RESUME.md
```

dentro de la carpeta del experimento.

## 16. Leer metrics.csv

metrics.csv es la tabla principal del experimento. Cada fila representa una época.

Columnas principales:

```text
epoch
train_loss_total
val_loss_total
train_dice_fake
val_dice_fake
train_iou_fake
val_iou_fake
train_precision_fake
val_precision_fake
train_recall_fake
val_recall_fake
train_f1_fake
val_f1_fake
train_accuracy_fake
val_accuracy_fake
train_skipped_nonfinite_batches
val_skipped_nonfinite_batches
```

Criterios prácticos:

### `val_dice_fake`

Métrica principal para elegir el mejor modelo de segmentación.

### `val_iou_fake`

Métrica más estricta de solapamiento.

### `val_precision_fake`

Indica si el modelo está pintando de más.

### `val_recall_fake`

Indica si el modelo está encontrando la región manipulada real.

### `val_loss_total`

Sirve para ver estabilidad general, pero no siempre el menor loss produce la mejor máscara visual.

### `overfit_loss_gap`

Si train_loss baja y val_loss sube, hay señal de sobreajuste.

## 17. Elegir el mejor checkpoint

Para segmentación fake, el candidato principal suele ser:

```text
checkpoints/best_val_dice.pth
```

También revisar:

```text
checkpoints/best_val_loss.pth
checkpoints/epoch_XXX.pth
```

Criterio recomendado:

1. elegir el mayor val_dice_fake;
2. confirmar que val_iou_fake también sea alto;
3. revisar precision y recall;
4. abrir samples visuales;
5. evitar un modelo que tenga buena métrica pero sobrepinte visualmente.

## 18. Interpretar precision y recall

### Caso 1

```text
recall alto
precision baja
```

El modelo encuentra casi toda la alteración, pero pinta de más. Hay muchos falsos positivos.

### Caso 2

```text
precision alta
recall bajo
```

El modelo pinta poco y con cuidado, pero deja zonas alteradas sin detectar.

### Caso 3

```text
precision y recall equilibrados
```

El modelo suele tener mejor Dice/F1.

## 19. Experimentos relevantes actuales

### Faceswap 512 estable

```text
configs/faceswap_training_512_pw2_lr3e4_stable_e15.yaml
```

Uso:

- Entrenamiento faceswap a 512x512.
- pos_weight: 2.1.
- batch_size: 2.
- gradient_accumulation_steps: 4.

Resultado observado:

- mejor época práctica alrededor de epoch 14;
- val_dice_fake aproximado: 0.965;
- val_iou_fake aproximado: 0.934.

### Faceswap 256 benchmark

```text
configs/faceswap_benchmark_256_pw2_lr3e4_stable_e20_bs16.yaml
```

Uso:

- Benchmark/ablación a 256x256 para comparar contra modelos de menor resolución.
- batch_size: 16.
- mixed_precision: true.

### Local inpainting 512 inicial

```text
configs/local_inpainting_training_512_pw33_lr3e4_stable_e15_bs4.yaml
```

Uso:

- Entrenamiento desde cero para local_inpainting.
- pos_weight: 33.4754694047.
- batch_size: 4.

Resultado observado:

- el pos_weight alto aumentó recall, pero bajó precision;
- el modelo tendía a sobrepintar.

### Local inpainting fine-tuning

```text
configs/local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6.yaml
```

Uso:

- Fine-tuning desde época 6.
- pos_weight reducido a 15.
- learning_rate: 5e-5.

Resultado observado:

- mejoró el Dice aproximadamente 5 puntos;
- pico reportado en época 3;
- val_dice_fake aproximado: 0.777.

## 20. Limpieza de archivos temporales

No versionar archivos temporales de trabajo como:

```text
comandos.txt
cambios.diff
src_comment_context.txt
dataset_audit_full.txt
```

Los experimentos grandes en outputs/ tampoco deben versionarse salvo que se decida guardar un resumen específico.

## 21. Código legacy

El repositorio conserva código legacy para comparación:

```text
codigo Deepshield/
├── model.py
├── train.py
└── test.py

Modelo_U_Net_dual_decoder.py
Train_UDD.py
Test_UDD.py
```

Ese código sirve como referencia histórica. El flujo operativo actual debe usar:

```text
train_orchestrator.py
src/
configs/
scripts/
```

## 22. Regla final antes de dejar entrenando toda la noche

Antes de dejar un entrenamiento largo corriendo:

1. validar sintaxis con py_compile;
2. correr dry-run del YAML;
3. confirmar ataque correcto;
4. confirmar image_size;
5. confirmar batch_size y accumulation;
6. confirmar pos_weight;
7. revisar que mixed_precision esté en true;
8. revisar que checkpoint_every sea 1;
9. vigilar los primeros minutos;
10. detener si hay OOM, NaN, Inf o skipped subiendo constantemente.
