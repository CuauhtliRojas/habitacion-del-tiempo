import argparse
import imageio.v2 as imageio
from pathlib import Path

def crear_gif_evolucion(
    nombre_experimento: str, 
    id_imagen: str, 
    tipo_imagen: str, 
    fps: int
):
    ruta_base = Path("outputs/experiments") / nombre_experimento / "samples"
    
    if not ruta_base.exists():
        print(f"Error: No se encontró la ruta {ruta_base}")
        return

    imagenes = []
    
    # Buscar todas las carpetas de época y ordenarlas numéricamente
    carpetas_epoca = sorted([d for d in ruta_base.iterdir() if d.is_dir() and "epoch_" in d.name])
    
    for epoca_dir in carpetas_epoca:
        # Armar la ruta completa a la imagen que buscamos
        ruta_imagen = epoca_dir / id_imagen / tipo_imagen
        
        if ruta_imagen.exists():
            print(f"Agregando: {ruta_imagen}")
            imagenes.append(imageio.imread(ruta_imagen))
        else:
            print(f"Advertencia: No se encontró {ruta_imagen} en {epoca_dir.name}")
    
    if imagenes:
        nombre_salida = f"{nombre_experimento}_{id_imagen}_{tipo_imagen.split('.')[0]}.gif"
        imageio.mimsave(nombre_salida, imagenes, fps=fps, loop=0)
        print(f"\n¡GIF creado con éxito!: {nombre_salida}")
    else:
        print("\nNo se encontraron imágenes para crear el GIF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de GIFs para la evolución del entrenamiento.")
    
    # Argumentos obligatorios
    parser.add_argument("-e", "--experimento", required=True, help="Nombre de la carpeta del experimento en outputs/experiments/")
    parser.add_argument("-i", "--imagen", required=True, help="ID de la subcarpeta de la imagen (ej. fs_b003_0660)")
    
    # Argumentos opcionales (tienen valores por defecto)
    parser.add_argument("-t", "--tipo", default="overlay_fake.png", help="Tipo de foto a rastrear. Por defecto: overlay_fake.png")
    parser.add_argument("--fps", type=int, default=3, help="Velocidad del GIF (cuadros por segundo). Por defecto: 3")

    args = parser.parse_args()

    crear_gif_evolucion(
        nombre_experimento=args.experimento,
        id_imagen=args.imagen,
        tipo_imagen=args.tipo,
        fps=args.fps
    )