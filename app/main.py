from fastapi import FastAPI, File, UploadFile
from app.inference import *
import numpy as np
from PIL import Image
from pathlib import Path
import os
import io

app = FastAPI()

# paths
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "trafficsignnet.keras"))
SIGNNAMES_PATH = Path(os.getenv("SIGNNAMES_PATH", ROOT_DIR / "signnames.csv"))

MODEL = None
CLASS_LABELS = None

@app.on_event("startup")
def load_resources():
    global MODEL, CLASS_LABELS

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    MODEL = load_model(str(MODEL_PATH))

    if not SIGNNAMES_PATH.exists():
        raise RuntimeError(f"signnames.csv not found at {SIGNNAMES_PATH}")

    CLASS_LABELS = load_labels()

@app.get("/")
def index():
    return {"details": "ready"}

@app.post("/inference")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(pil_img)

    result = predict(np_img, model=MODEL, class_labels=CLASS_LABELS)

    return {
        "filename": file.filename,
        "predicted_label": result["predicted_label"],
        "probability": float(result["probability"]),
    }