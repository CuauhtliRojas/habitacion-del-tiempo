from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

"""
Este archivo genera gráficas y reportes simples a partir de metrics.csv.

No entrena el modelo ni recalcula métricas desde imágenes. Solo toma la tabla
que dejó el entrenamiento y la convierte en artefactos fáciles de revisar.

Sirve para responder preguntas prácticas:
- si la pérdida baja o se estanca;
- si validación se separa demasiado de entrenamiento;
- si Dice e IoU mejoran;
- si precision y recall están equilibrados;
- si hay señales de sobreajuste o subajuste.
"""

def _read_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    """
    Lee metrics.csv y valida lo mínimo para poder graficar.

    Si el archivo no existe, está vacío o no tiene columna epoch, se detiene el
    proceso. Es mejor fallar con un mensaje claro que generar gráficas vacías o
    engañosas.
    """
    metrics_csv = Path(metrics_csv)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"No existe metrics.csv: {metrics_csv}")

    df = pd.read_csv(metrics_csv)

    if df.empty:
        raise ValueError(f"metrics.csv esta vacio: {metrics_csv}")

    if "epoch" not in df.columns:
        raise ValueError("metrics.csv debe contener la columna 'epoch'.")

    return df


def _plot_columns(
    *,
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """
    Grafica una o varias columnas contra epoch.

    La función ignora columnas que no existan en metrics.csv. Esto permite que
    el script siga funcionando aunque un experimento viejo tenga menos métricas
    que uno nuevo.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    """
    Solo se grafican columnas realmente disponibles. Si ninguna existe, no se
    genera archivo para evitar una imagen vacía.
    """
    available = [column for column in columns if column in df.columns]

    if not available:
        return

    plt.figure(figsize=(10, 6))

    for column in available:
        plt.plot(df["epoch"], df[column], marker="o", label=column)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_training_plots(
    *,
    metrics_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Genera las gráficas principales del entrenamiento.

    Cada gráfica se guarda como PNG dentro de la carpeta plots del experimento.
    Estas imágenes sirven para revisar rápidamente si el entrenamiento mejoró,
    se estancó o empezó a sobreajustarse.
    """
    df = _read_metrics(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    """
    Lista de gráficas actuales.

    Todas estas métricas sí se usan ahora para revisar el entrenamiento:
    - loss: estabilidad general;
    - Dice e IoU: calidad de segmentación;
    - precision/recall/F1: equilibrio entre pintar de más y detectar de menos;
    - accuracy: referencia secundaria;
    - métricas authentic: control de la segunda salida del modelo.
    """
    plot_specs = [
        (
            ["train_loss_total", "val_loss_total"],
            "Curva de perdida total",
            "Loss",
            output_dir / "loss_total_curve.png",
        ),
        (
            ["train_loss_fake", "val_loss_fake"],
            "Curva de perdida de mascara manipulada",
            "Loss fake",
            output_dir / "loss_fake_curve.png",
        ),
        (
            ["train_loss_authentic", "val_loss_authentic"],
            "Curva de perdida de mascara autentica",
            "Loss authentic",
            output_dir / "loss_authentic_curve.png",
        ),
        (
            ["train_dice_fake", "val_dice_fake"],
            "Curva Dice de mascara manipulada",
            "Dice",
            output_dir / "dice_fake_curve.png",
        ),
        (
            ["train_iou_fake", "val_iou_fake"],
            "Curva IoU de mascara manipulada",
            "IoU",
            output_dir / "iou_fake_curve.png",
        ),
        (
            ["train_precision_fake", "val_precision_fake"],
            "Precision de mascara manipulada",
            "Precision",
            output_dir / "precision_fake_curve.png",
        ),
        (
            ["train_recall_fake", "val_recall_fake"],
            "Recall de mascara manipulada",
            "Recall",
            output_dir / "recall_fake_curve.png",
        ),
        (
            ["train_f1_fake", "val_f1_fake"],
            "F1 de mascara manipulada",
            "F1",
            output_dir / "f1_fake_curve.png",
        ),
        (
            ["train_accuracy_fake", "val_accuracy_fake"],
            "Accuracy de mascara manipulada",
            "Accuracy",
            output_dir / "accuracy_fake_curve.png",
        ),
        (
            ["val_precision_fake", "val_recall_fake", "val_f1_fake"],
            "Precision, recall y F1 en validacion",
            "Score",
            output_dir / "precision_recall_f1_curve.png",
        ),
        (
            ["train_dice_authentic", "val_dice_authentic"],
            "Curva Dice de mascara autentica",
            "Dice authentic",
            output_dir / "dice_authentic_curve.png",
        ),
        (
            ["train_iou_authentic", "val_iou_authentic"],
            "Curva IoU de mascara autentica",
            "IoU authentic",
            output_dir / "iou_authentic_curve.png",
        ),
    ]

    for columns, title, ylabel, path in plot_specs:
        _plot_columns(
            df=df,
            columns=columns,
            title=title,
            ylabel=ylabel,
            output_path=path,
        )
        if path.exists():
            generated.append(path)

    if {"train_loss_total", "val_loss_total"}.issubset(df.columns):
        """
        La brecha val-train ayuda a detectar sobreajuste.

        Si la pérdida de entrenamiento baja pero la de validación se aleja, el
        modelo puede estar memorizando mejor el train que generalizando.
        """
        gap_path = output_dir / "overfit_loss_gap.png"
        gap = df["val_loss_total"] - df["train_loss_total"]

        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], gap, marker="o", label="val_loss_total - train_loss_total")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title("Brecha de perdida train/val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss gap")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(gap_path, dpi=150)
        plt.close()
        generated.append(gap_path)

    return generated


def analyze_fit_status(
    *,
    metrics_csv: str | Path,
    output_path: str | Path,
    patience: int = 4,
    dice_gap_threshold: float = 0.15,
) -> str:
    """
    Genera un reporte textual simple sobre el ajuste del entrenamiento.

    No reemplaza la revisión humana ni las muestras visuales. Solo busca señales
    rápidas:
    - posible sobreajuste si validación deja de mejorar mientras train mejora;
    - posible sobreajuste si Dice train supera demasiado a Dice val;
    - posible subajuste si Dice sigue bajo en train y validación.
    """
    df = _read_metrics(metrics_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    messages.append("=== REPORTE DE AJUSTE DEL ENTRENAMIENTO ===")
    messages.append(f"Epochs registradas: {len(df)}")

    if len(df) < 2:
        """
        Con una sola época no hay tendencia. Se puede registrar el reporte, pero
        todavía no tiene sentido diagnosticar sobreajuste o subajuste.
        """
        messages.append("Aun no hay suficientes epochs para diagnosticar overfitting o underfitting.")
        report = "\n".join(messages)
        output_path.write_text(report, encoding="utf-8")
        return report

    latest = df.iloc[-1]

    if {"train_loss_total", "val_loss_total"}.issubset(df.columns):
        train_loss = float(latest["train_loss_total"])
        val_loss = float(latest["val_loss_total"])
        loss_gap = val_loss - train_loss

        messages.append(f"Ultima train_loss_total: {train_loss:.6f}")
        messages.append(f"Ultima val_loss_total: {val_loss:.6f}")
        messages.append(f"Brecha val-train loss: {loss_gap:.6f}")

        if len(df) >= patience:
            """
            Señal práctica de sobreajuste:
            train mejora, pero validación no mejora durante varias épocas.
            """
            recent_val = df["val_loss_total"].tail(patience).to_list()
            val_non_improving = all(
                recent_val[index] >= recent_val[index - 1]
                for index in range(1, len(recent_val))
            )

            recent_train = df["train_loss_total"].tail(patience).to_list()
            train_improving = recent_train[-1] < recent_train[0]

            if val_non_improving and train_improving:
                messages.append(
                    "ALERTA: posible overfitting. La perdida de entrenamiento mejora, "
                    "pero la perdida de validacion no mejora en las ultimas epochs."
                )

    if {"train_dice_fake", "val_dice_fake"}.issubset(df.columns):
        """
        Dice fake es la métrica central para revisar la máscara manipulada.

        La brecha train-val ayuda a ver si el modelo aprendió demasiado bien el
        conjunto de entrenamiento, pero no generaliza igual en validación.
        """
        train_dice = float(latest["train_dice_fake"])
        val_dice = float(latest["val_dice_fake"])
        dice_gap = train_dice - val_dice

        messages.append(f"Ultimo train_dice_fake: {train_dice:.6f}")
        messages.append(f"Ultimo val_dice_fake: {val_dice:.6f}")
        messages.append(f"Brecha train-val Dice fake: {dice_gap:.6f}")

        if dice_gap >= dice_gap_threshold:
            messages.append(
                "ALERTA: posible overfitting. La brecha Dice entre entrenamiento "
                "y validacion supera el umbral configurado."
            )

        if train_dice < 0.35 and val_dice < 0.35 and len(df) >= patience:
            messages.append(
                "ALERTA: posible underfitting. Dice permanece bajo en entrenamiento "
                "y validacion tras varias epochs."
            )

    if len(messages) <= 4:
        messages.append("Sin alertas fuertes con las metrics disponibles.")

    report = "\n".join(messages)
    output_path.write_text(report, encoding="utf-8")
    return report
