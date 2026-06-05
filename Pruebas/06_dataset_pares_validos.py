"""
Demostración simple: validación de pares imagen/máscaras.

Compara:
- codigo Deepshield/model.py
- codigo Deepshield/train.py
- src/dataset.py

Referencia:
https://docs.python.org/3/library/tempfile.html
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.dataset import build_samples


def crear_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=255).save(path)


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)

        base = root / "local_inpainting" / "train"
        image_dir = base / "imagen_original"
        authentic_dir = base / "mascara_autentica"
        fake_dir = base / "mascara_fake"

        crear_png(image_dir / "muestra_001.png")
        crear_png(authentic_dir / "muestra_001.png")

        print("=== Validación de pares imagen/máscaras ===")

        try:
            build_samples(
                dataset_root=root,
                attacks=["local_inpainting"],
                split="train",
            )
        except FileNotFoundError as error:
            print("Error detectado correctamente:")
            print(error)

        crear_png(fake_dir / "muestra_001.png")

        samples = build_samples(
            dataset_root=root,
            attacks=["local_inpainting"],
            split="train",
        )

        print("")
        print(f"Muestras válidas encontradas: {len(samples)}")
        print(f"Ataque: {samples[0].attack}")
        print(f"Split: {samples[0].split}")
        print("")
        print("Conclusión:")
        print("El dataset nuevo detiene muestras incompletas y permite rutas por ataque/split.")


if __name__ == "__main__":
    main()