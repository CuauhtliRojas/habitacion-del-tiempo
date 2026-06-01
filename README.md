# Cámara de entrenamiento 100G

Repositorio local para preparar, validar y ejecutar experimentos de segmentación semántica binaria sobre manipulaciones faciales deepfake mediante un modelo U-Net dual-decoder.

El objetivo del orquestador es centralizar la configuración experimental, la validación previa, el control de épocas, checkpoints, métricas, gráficas, muestras visuales, reporte de ajuste y reanudación de entrenamiento. El proyecto está pensado para funcionar tanto en equipo local como en Colab, siempre que se conserve la misma estructura de dataset y se ajusten las rutas cuando sea necesario.

## Estado actual

El proyecto ya cuenta con:

- Entorno `uv` configurado con Python 3.12.
- PyTorch con CUDA instalado localmente.
- Dataset local ignorado por Git.
- Modelo base `DualSegmentationModel`.
- Orquestador base `train_orchestrator.py`.
- Validación previa mediante `--dry-run`.
- Métricas base de segmentación.
- Checkpoints completos.
- Generación futura de gráficas, muestras visuales y reporte de ajuste.
- Configuraciones YAML separadas para preflight y experimentos base.

## Regla operativa actual

No se ejecuta entrenamiento real hasta cerrar:

1. configuración final;
2. documentación local y Colab;
3. validaciones `--dry-run`;
4. revisión del flujo de checkpoints y resume;
5. autorización explícita para iniciar entrenamiento.

Los comandos de entrenamiento real no deben ejecutarse todavía. Por ahora solo se permite validar estructura, compilar scripts, revisar configuraciones, ejecutar conteos y correr `--dry-run`.

## Dataset esperado

La estructura esperada es:

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

El dataset local no se versiona en Git. Está excluido mediante `.gitignore`.

## Conteo actual del dataset

Conteos validados localmente:

```text
faceswap/train: 23,338 muestras
faceswap/test: 2,593 muestras

local_inpainting/train: 22,501 muestras
local_inpainting/test: 2,499 muestras
```

Total estimado:

```text
50,931 muestras completas
152,793 archivos PNG
```

Cada muestra completa está formada por:

```text
imagen_original
mascara_autentica
mascara_fake
```

## Validar conteos del dataset

```powershell
uv run python .\scripts\check_dataset_counts.py
```

Salida esperada:

```text
[faceswap/train]
imagen_original: 23338
mascara_autentica: 23338
mascara_fake: 23338
OK: 23338 muestras completas

[faceswap/test]
imagen_original: 2593
mascara_autentica: 2593
mascara_fake: 2593
OK: 2593 muestras completas

[local_inpainting/train]
imagen_original: 22501
mascara_autentica: 22501
mascara_fake: 22501
OK: 22501 muestras completas

[local_inpainting/test]
imagen_original: 2499
mascara_autentica: 2499
mascara_fake: 2499
OK: 2499 muestras completas

TOTAL ESTIMADO DE MUESTRAS COMPLETAS: 50931
TOTAL ESTIMADO DE PNG: 152793
```

## Validar sintaxis de scripts

```powershell
uv run python -m py_compile `
  .\train_orchestrator.py `
  .\src\dataset.py `
  .\src\metrics.py `
  .\src\checkpoints.py `
  .\src\plots.py `
  .\src\visualization.py
```

Si no imprime errores, la sintaxis básica está correcta.

## Validar el orquestador sin entrenar

Modo mixto 256:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\preflight_mixto_256.yaml `
  --dry-run
```

Modo mixto 512 reducido:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\preflight_mixto_512_lite.yaml `
  --dry-run
```

El modo `--dry-run` debe validar la configuración, contar muestras, calcular batches aproximados y mostrar advertencias, pero no debe entrenar ni guardar checkpoints reales.

## Configuraciones disponibles

### `configs/mixto_fs_li_256_base.yaml`

Configuración base mixta para `faceswap` y `local_inpainting` en 256x256.

### `configs/preflight_mixto_256.yaml`

Configuración de validación previa para dataset mixto en 256x256.

### `configs/preflight_mixto_512_lite.yaml`

Configuración reducida para validar uso de 512x512 sin entrenar todo el dataset. Incluye límites de muestras para pruebas controladas.

### `configs/faceswap_256_base.yaml`

Configuración base para entrenar únicamente con `faceswap`.

### `configs/local_inpainting_256_base.yaml`

Configuración base para entrenar únicamente con `local_inpainting`.

## Parámetros principales del orquestador

El orquestador puede recibir valores desde YAML o desde terminal. Los valores enviados por terminal sobrescriben los del YAML.

Parámetros principales:

- `experiment_name`
- `dataset_root`
- `output_root`
- `attacks`
- `image_size`
- `batch_size`
- `epochs`
- `learning_rate`
- `pos_weight`
- `val_ratio`
- `seed`
- `num_workers`
- `threshold`
- `checkpoint_every`
- `sample_every`
- `mixed_precision`
- `max_train_samples`
- `max_val_samples`
- `max_test_samples`
- `resume`
- `weights`
- `device`
- `dry_run`
- `yes`

## Sobrescritura por terminal

Ejemplo de validación 512 sin entrenamiento real:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\preflight_mixto_256.yaml `
  --experiment-name dry_512_mixto `
  --image-size 512 `
  --batch-size 2 `
  --epochs 1 `
  --max-train-samples 100 `
  --max-val-samples 40 `
  --dry-run
```

Este comando solo valida configuración. No ejecuta entrenamiento real.

## Artefactos esperados durante entrenamiento futuro

Cuando el entrenamiento real sea autorizado, cada experimento generará:

```text
outputs/experiments/<experiment_name>/
├── config.resolved.json
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
│   ├── dice_fake_curve.png
│   ├── iou_fake_curve.png
│   ├── precision_recall_f1_curve.png
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

## Checkpoints

El orquestador debe guardar checkpoints completos, no solo pesos del modelo.

Un checkpoint completo incluye:

```text
epoch
model_state
optimizer_state
config
metrics_history
best_val_loss
best_val_dice
scaler_state, si se usa mixed precision
```

Esto permite reanudar entrenamiento desde otro equipo, Colab o una sesión futura.

## Reanudación futura de entrenamiento

Cuando se autorice entrenamiento real, la reanudación se hará con `--resume` apuntando a un checkpoint completo.

Ejemplo conceptual:

```powershell
uv run python .\train_orchestrator.py `
  --config .\configs\mixto_fs_li_256_base.yaml `
  --resume .\outputs\experiments\<experiment_name>\checkpoints\last.pth
```

Este flujo todavía debe validarse antes de usarse en entrenamiento real.

## Métricas principales

El foco principal de evaluación está en la máscara manipulada o fake:

- `dice_fake`
- `iou_fake`
- `precision_fake`
- `recall_fake`
- `f1_fake`
- `accuracy_fake`

La máscara auténtica también se evalúa porque el modelo tiene doble decoder, pero el análisis principal de la tesis debe centrarse en la localización de la región manipulada.

## Reporte de ajuste

El archivo `fit_report.txt` debe resumir señales de:

- posible overfitting;
- posible underfitting;
- brecha entre entrenamiento y validación;
- comportamiento de pérdida;
- comportamiento de Dice fake.

El reporte no sustituye el análisis humano. Solo funciona como alerta heurística.

## Uso local

El entorno local usa:

```text
Python 3.12
uv
PyTorch CUDA
RTX 4050 Laptop GPU
```

Validación de CUDA:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Uso futuro en Colab

Para Colab se debe conservar la misma estructura lógica del repo y del dataset. El flujo recomendado será:

1. montar Google Drive;
2. ubicar el repo en Drive o subirlo temporalmente;
3. instalar dependencias;
4. validar conteos del dataset;
5. ejecutar `--dry-run`;
6. reanudar o iniciar entrenamiento solo cuando esté autorizado.

La documentación exacta de Colab se agregará antes del primer entrenamiento real.

## Commits locales actuales

El repositorio usa control de versiones local. No se subirá a GitHub hasta tener al menos el primer entrenamiento validado.

Commits base:

```text
13ebe62 Inicializa entorno de entrenamiento U-Net dual decoder
4fb57ff Agrega orquestador base de entrenamiento
53982d4 Completa monitoreo del orquestador
```

## Pendientes antes de entrenar

Antes de iniciar entrenamiento real falta cerrar:

1. README local/Colab final.
2. Validación de `--dry-run` sin inicializar GPU innecesariamente.
3. Revisión de `resume` y checkpoints.
4. Revisión de compatibilidad del modelo con la función de pérdida.
5. Revisión de configuración 512.
6. Validación de rutas en entorno local.
7. Decisión explícita del primer experimento real.
