from pathlib import Path

DATASET_ROOT = Path("Dataset_U-Net_dual_decoder_fs_li")

attacks = ["faceswap", "local_inpainting"]
splits = ["train", "test"]
folders = ["imagen_original", "mascara_autentica", "mascara_fake"]

total_samples = 0

for attack in attacks:
    for split in splits:
        print(f"\n[{attack}/{split}]")
        split_counts = {}

        for folder in folders:
            path = DATASET_ROOT / attack / split / folder
            files = list(path.glob("*.png")) if path.exists() else []
            split_counts[folder] = len(files)
            print(f"{folder}: {len(files)}")

        if len(set(split_counts.values())) == 1:
            samples = next(iter(split_counts.values()))
            total_samples += samples
            print(f"OK: {samples} muestras completas")
        else:
            print("ERROR: conteos no coinciden entre imagenes y mascaras")

print(f"\nTOTAL ESTIMADO DE MUESTRAS COMPLETAS: {total_samples}")
print(f"TOTAL ESTIMADO DE PNG: {total_samples * 3}")
