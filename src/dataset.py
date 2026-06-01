from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


IMAGE_DIR = "imagen_original"
AUTH_MASK_DIR = "mascara_autentica"
FAKE_MASK_DIR = "mascara_fake"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    authentic_mask_path: Path
    fake_mask_path: Path
    attack: str
    split: str
    stem: str


def _list_images(path: Path) -> list[Path]:
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
                raise FileNotFoundError(
                    f"Falta mascara autentica para {image_path.name}: {authentic_path}"
                )

            if not fake_path.exists():
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
    if max_samples is None:
        return list(samples)
    return list(samples[:max_samples])


class DualMaskSegmentationDataset(Dataset):
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
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        array = np.asarray(mask, dtype=np.float32) / 255.0
        array = (array > 0.5).astype(np.float32)
        array = np.expand_dims(array, axis=0)
        return torch.from_numpy(array)

    def __getitem__(self, index: int):
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
    counts: dict[str, int] = {}

    for sample in samples:
        counts[sample.attack] = counts.get(sample.attack, 0) + 1

    return counts
