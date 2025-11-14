import os
import cv2
import numpy as np

from ultralytics import YOLOv10

# ---------------------------------------------------------------
#  Script completo con carga automática del modelo YOLOv10
# ---------------------------------------------------------------

# Nombre del modelo base
DEFAULT_MODEL = "yolov10n.pt"

# Ruta de la imagen que se desea analizar
IMAGE_PATH = "Carretera.jpg"

# Umbral mínimo de confianza
CONFIDENCE_THRESHOLD = 0.25

# ---------------------------------------------------------------
# 1. Carga del modelo YOLOv10
# ---------------------------------------------------------------
if os.path.exists(DEFAULT_MODEL):
    print(f"🚀 Cargando modelo local: {DEFAULT_MODEL}")
    model = YOLOv10(DEFAULT_MODEL)
else:
    print(f"⚠️ No se encontró '{DEFAULT_MODEL}'. Intentando cargar el modelo por defecto desde Ultralytics...")
    model = YOLOv10()  # Carga modelo predefinido

# ---------------------------------------------------------------
# 2. Procesamiento de la imagen
# ---------------------------------------------------------------
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ No se encontró la imagen: {IMAGE_PATH}\nColoca una imagen en la misma carpeta del script.")

print(f"🔎 Procesando imagen: {IMAGE_PATH}")
results = model.predict(IMAGE_PATH, conf=CONFIDENCE_THRESHOLD, save=False)

# ---------------------------------------------------------------
# 3. Generación del mapa de calor
# ---------------------------------------------------------------
for result in results:
    img = result.orig_img.copy()
    h, w, _ = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    total_conf, count = 0.0, 0

    # Procesar detecciones
    for box in result.boxes:
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        total_conf += conf
        count += 1
        heatmap[y1:y2, x1:x2] += conf

    # ---------------------------------------------------------------
    # 4. Normalización del mapa de calor
    # ---------------------------------------------------------------
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    blended = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # ---------------------------------------------------------------
    # 5. Resultados y salida
    # ---------------------------------------------------------------
    avg_conf = total_conf / max(count, 1)
    out_path = str(IMAGE_PATH).rsplit(".", 1)[0] + "_heatmap.jpg"

    print("\n📊 RESULTADOS:")
    print(f"   🔥 Promedio de confianza: {avg_conf:.2f}")
    print(f"   🧩 Total de detecciones: {count}")
    print(f"   💾 Imagen resultante: {out_path}")

    cv2.imwrite(out_path, blended)

    cv2.imshow("YOLOv10 Heatmap", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
