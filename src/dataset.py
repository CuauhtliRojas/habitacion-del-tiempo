from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_DIR = "imagen_original"
AUTH_MASK_DIR = "mascara_autentica"
FAKE_MASK_DIR = "mascara_fake"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

"""
Este archivo se encarga de leer el dataset.

En el código antiguo las rutas estaban escritas directamente dentro del script
de entrenamiento. Aquí se separa esa responsabilidad: este archivo solo sabe
buscar imágenes y máscaras, validar que existan y convertirlas a tensores.

La idea es que el entrenamiento pueda cambiar de ataque, split o resolución
desde un archivo YAML, sin modificar el código Python.
"""

@dataclass(frozen=True)
class SegmentationSample:
    """
    Representa una muestra completa del dataset.

    Una muestra no es solo la imagen. También debe tener su máscara auténtica,
    su máscara manipulada, el ataque al que pertenece y el split donde vive.

    Esto evita entrenar con pares incompletos o ambiguos.
    """
    image_path: Path
    authentic_mask_path: Path
    fake_mask_path: Path
    attack: str
    split: str
    stem: str


def _list_images(path: Path) -> list[Path]:
    """
    Lista imágenes válidas dentro de una carpeta.

    Si la carpeta no existe, se detiene el proceso.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe la carpeta: {path}")

    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    )


def build_samples(
    dataset_root: str | Path,
    attacks: Sequence[str],
    split: str,
) -> list[SegmentationSample]:
    """
    Construye la lista de muestras que se van a entrenar o validar.

    En el código antiguo se usaban rutas fijas como Train_D/images. Aquí la ruta
    se arma con esta estructura:

    dataset_root / ataque / split / imagen_original
    dataset_root / ataque / split / mascara_autentica
    dataset_root / ataque / split / mascara_fake

    Gracias a esto podemos entrenar faceswap, local_inpainting o una mezcla de
    ataques sin cambiar el código.
    """
    dataset_root = Path(dataset_root)
    samples: list[SegmentationSample] = []

    for attack in attacks:
        base = dataset_root / attack / split
        image_dir = base / IMAGE_DIR
        authentic_dir = base / AUTH_MASK_DIR
        fake_dir = base / FAKE_MASK_DIR

        images = _list_images(image_dir)

        for image_path in images:
            authentic_path = authentic_dir / image_path.name
            fake_path = fake_dir / image_path.name

            if not authentic_path.exists():
                """
                Si falta la máscara auténtica, la muestra queda incompleta.

                No se rellena ni se ignora en silencio porque eso haría que el
                entrenamiento aprendiera con datos mal emparejados.
                """
                raise FileNotFoundError(
                    f"Falta mascara autentica para {image_path.name}: {authentic_path}"
                )

            if not fake_path.exists():
                """
                Si falta la máscara fake, no existe el objetivo principal que el
                modelo debe aprender para esa imagen.

                Por eso se detiene el proceso en lugar de entrenar con una
                muestra incompleta.
                """
                raise FileNotFoundError(
                    f"Falta mascara fake para {image_path.name}: {fake_path}"
                )

            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    authentic_mask_path=authentic_path,
                    fake_mask_path=fake_path,
                    attack=attack,
                    split=split,
                    stem=image_path.stem,
                )
            )

    if not samples:
        raise RuntimeError(
            f"No se encontraron muestras para split={split}, attacks={list(attacks)}"
        )

    return samples


def split_train_val(
    samples: Sequence[SegmentationSample],
    val_ratio: float,
    seed: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    """
    Divide las muestras en entrenamiento y validación.

    La semilla permite que la división sea repetible. Si dos experimentos usan
    la misma seed, comparan contra el mismo split y la comparación es más justa.

    También permite limitar muestras para pruebas rápidas sin tocar el dataset.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio debe estar en el rango [0.0, 1.0).")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    val_size = int(len(samples) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]

    if max_val_samples is not None:
        val_samples = val_samples[:max_val_samples]

    return train_samples, val_samples


def limit_samples(
    samples: Sequence[SegmentationSample],
    max_samples: int | None,
) -> list[SegmentationSample]:
    """
    Recorta una lista de muestras cuando se quiere hacer una prueba pequeña.

    Sirve para smoke tests o diagnósticos rápidos. Si max_samples es None,
    devuelve todas las muestras.
    """
    if max_samples is None:
        return list(samples)
    return list(samples[:max_samples])


class DualMaskSegmentationDataset(Dataset):
    """
    Dataset final que entrega lo que necesita el entrenamiento.

    Cada elemento devuelve:
    - image: imagen original normalizada.
    - fake_mask: máscara de la región manipulada.
    - authentic_mask: máscara de la región auténtica.
    - attack: nombre del ataque.
    - stem: nombre base del archivo.

    Esto deja el entrenamiento más claro que el código antiguo, donde se
    regresaban tuplas y el significado dependía del orden.
    """
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        image_size: int,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)

        if self.image_size <= 0:
            raise ValueError("image_size debe ser mayor que 0.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Carga una imagen RGB y la convierte a tensor.

        La imagen sí puede redimensionarse con BILINEAR porque es una fotografía:
        suavizar un poco el cambio de tamaño no rompe su significado.
        """
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def _load_mask(self, path: Path) -> torch.Tensor:
        """
        Carga una máscara y la convierte a binaria.

        Aquí usamos NEAREST para redimensionar porque una máscara no debe crear
        grises intermedios. La máscara debe seguir siendo fondo o región.

        Después se aplica un umbral:
        - mayor a 0.5 se vuelve 1;
        - menor o igual a 0.5 se vuelve 0.
        """
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        array = np.asarray(mask, dtype=np.float32) / 255.0
        array = (array > 0.5).astype(np.float32)
        array = np.expand_dims(array, axis=0)
        return torch.from_numpy(array)

    def __getitem__(self, index: int):
        """
        Devuelve una muestra lista para el entrenamiento.

        Se cargan la imagen, la máscara fake y la máscara auténtica. También se
        devuelven metadatos para poder separar métricas por ataque y guardar
        muestras visuales con nombres reconocibles.
        """
        sample = self.samples[index]

        image = self._load_image(sample.image_path)
        fake_mask = self._load_mask(sample.fake_mask_path)
        authentic_mask = self._load_mask(sample.authentic_mask_path)

        return {
            "image": image,
            "fake_mask": fake_mask,
            "authentic_mask": authentic_mask,
            "attack": sample.attack,
            "stem": sample.stem,
        }


def count_by_attack(samples: Iterable[SegmentationSample]) -> dict[str, int]:
    """
    Cuenta cuántas muestras hay por ataque.

    Se usa en el preflight para revisar rápidamente si el experimento va a
    entrenar con el ataque correcto y con la cantidad esperada de datos.
    """
    counts: dict[str, int] = {}

    for sample in samples:
        counts[sample.attack] = counts.get(sample.attack, 0) + 1

    return counts
