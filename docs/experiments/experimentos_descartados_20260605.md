# Experimentos descartados - 2026-06-05

Estos experimentos ya fueron ejecutados y revisados. Se eliminaron sus carpetas de outputs para liberar espacio, pero se conservan los YAML fuente en configs/ cuando existen.

No se elimina el dataset, el codigo fuente, las pruebas en Pruebas/ ni el entrenamiento mixto activo.

## Carpetas eliminadas

- dummy_resume_validation
  - Tamano liberado aproximado: 0.38 MB (0 GB)
  - Motivo: Prueba de reanudación; 3 épocas; métricas bajas; sin checkpoints útiles.

- faceswap_smoke_gpu_512_pw2_lr3e4
  - Tamano liberado aproximado: 0 MB (0 GB)
  - Motivo: Smoke test; sin metrics.csv ni checkpoints.

- faceswap_smoke_gpu_512_pw2_lr3e4_logits_e1
  - Tamano liberado aproximado: 1218.42 MB (1.19 GB)
  - Motivo: Smoke test de 1 época; carpeta pesada por checkpoints duplicados; no representa resultado final.

- faceswap_smoke_gpu_512_pw2_lr3e4_stable_e1
  - Tamano liberado aproximado: 0 MB (0 GB)
  - Motivo: Smoke test; sin metrics.csv ni checkpoints.

- faceswap_smoke_gpu_512_pw2_lr3e4_stable_e1_retry1
  - Tamano liberado aproximado: 1218.41 MB (1.19 GB)
  - Motivo: Smoke test de 1 época; carpeta pesada por checkpoints duplicados; no representa resultado final.

- local_inpainting_finetune_512_from_e06_pw15_lr5e5_e6
  - Tamano liberado aproximado: 2148.82 MB (2.1 GB)
  - Motivo: Fine-tuning parcial con 4 épocas reales; resultado inferior al fine-tuning bestdice; no se conserva como modelo candidato.

## Carpetas conservadas

- faceswap_training_512_pw2_lr3e4_stable_e15: baseline 512 principal de FaceSwap.
- local_inpainting_training_512_pw33_lr3e4_stable_e15_bs4_IoU: baseline principal de Local Inpainting con IoU.
- local_inpainting_finetune_512_bestdice_lr3e5_bce05_dice2_iou1: fine-tuning desde best dice.
- mixed_faceswap_local_inpainting_512_attack_weighted_bce_dice_iou: entrenamiento mixto activo.

