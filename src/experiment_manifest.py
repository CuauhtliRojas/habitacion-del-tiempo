from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

"""
Este archivo construye el manifiesto del experimento.

El manifiesto es una ficha técnica del entrenamiento. Sirve para saber con qué
versión de Python, PyTorch, CUDA, GPU, configuración y commit de Git se generó
un resultado.

Esto es importante porque dos entrenamientos pueden usar el mismo código, pero
dar resultados distintos si cambian la GPU, la configuración, el commit o la
semilla.
"""

def _run_command(args: list[str]) -> str | None:
    """
    Ejecuta un comando pequeño del sistema y devuelve su salida.

    Se usa para consultar Git sin romper el entrenamiento. Si el comando falla,
    se devuelve None en lugar de detener todo el proceso.
    """
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip() or None


def get_git_commit() -> str | None:
    """
    Obtiene el commit exacto usado durante el experimento.

    Esto permite saber con qué versión del código se generó un entrenamiento.
    Si después el resultado se compara, se reproduce o se defiende, el commit
    ayuda a rastrear el estado real del repositorio.
    """
    return _run_command(["git", "rev-parse", "HEAD"])


def get_git_branch() -> str | None:
    """
    Obtiene la rama activa del repositorio.

    La rama ayuda a distinguir si el experimento salió de una versión estable,
    una prueba temporal o una línea de trabajo distinta.
    """
    return _run_command(["git", "branch", "--show-current"])


def get_git_status_short() -> str | None:
    """
    Guarda el estado corto de Git.

    Si hay archivos modificados o sin commit al momento de entrenar, el
    manifiesto lo deja registrado. Esto evita creer que un resultado salió de
    un repo limpio cuando todavía había cambios locales.
    """
    return _run_command(["git", "status", "--short"])


def build_run_manifest(
    *,
    config: dict[str, Any],
    command_line: list[str],
    device: torch.device,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Construye el manifiesto del experimento.

    El manifiesto junta la configuración, el comando usado, el entorno de Python,
    la versión de PyTorch, la información de CUDA/GPU y el estado de Git.

    La idea es que cada entrenamiento deje una ficha técnica mínima para saber
    cómo se ejecutó y con qué condiciones.
    """
    cuda_available: bool | str

    if dry_run:
        """
        En dry-run no se consulta CUDA a fondo.

        El dry-run sirve para validar configuración y conteos sin iniciar un
        entrenamiento real. Por eso aquí se deja explícito que esos datos no se
        consultaron.
        """
        cuda_available = "no consultado en dry-run"
        cuda_device_name = "no consultado en dry-run"
        cuda_device_count = "no consultado en dry-run"
    else:
        """
        En entrenamiento real sí se registra CUDA.

        Esto ayuda a explicar diferencias de rendimiento o memoria entre equipos.
        Por ejemplo, un experimento puede caber en una GPU y fallar en otra con
        menos VRAM.
        """
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

    """
    El diccionario final se guarda como JSON.

    No contiene el dataset ni archivos pesados. Solo guarda metadatos suficientes
    para auditar el experimento: cuándo se creó, con qué comando, con qué entorno,
    con qué versión de código y con qué configuración.
    """
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": command_line,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "device_selected": str(device),
        },
        "git": {
            "commit": get_git_commit(),
            "branch": get_git_branch(),
            "status_short": get_git_status_short(),
        },
        "config": config,
    }


def write_run_manifest(
    *,
    path: str | Path,
    manifest: dict[str, Any],
) -> None:
    """
    Escribe el manifiesto en disco.

    Normalmente se guarda como `run_manifest.json` dentro de la carpeta del
    experimento. Así queda junto a métricas, checkpoints, gráficas y muestras.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    """
    ensure_ascii=False permite conservar acentos si algún campo de texto los usa.
    indent=2 deja el JSON legible para revisión humana.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
