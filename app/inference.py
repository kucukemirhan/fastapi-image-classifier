from tensorflow.keras.models import load_model
from skimage import transform, exposure, io
import tensorflow as tf
import numpy as np
import pandas as pd

def preprocess_img(img, target_size=(32, 32)):
    img = transform.resize(img, target_size, anti_aliasing=True)
    img = exposure.equalize_adapthist(img, clip_limit=0.1)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def load_labels():
    df = pd.read_csv("signnames.csv")
    df = df.sort_values("ClassId")
    class_labels = df["SignName"].astype(str).tolist()

    return class_labels


def predict(image, model=None, model_path=None, class_labels=None):

    if model is None:
        if model_path is None:
            raise ValueError("Please input a model parameter or model_path.")
        model = load_model(model_path)

    x = preprocess_img(image)

    y = model(x, training=False) # forward pass
    probs = tf.squeeze(y, axis=0).numpy()

    class_id = int(np.argmax(probs))
    prob = float(probs[class_id])

    if class_labels is not None and class_id < len(class_labels):
        label = class_labels[class_id]
    else:
        label = str(class_id)

    return {"predicted_label": label,
            "probability": prob}
