import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


def _safe_load_model(model_path: str):
    """Load model dengan beberapa cara fallback untuk kompatibilitas TF versi beda."""
    # Cara 1: compile=False (paling umum)
    try:
        return load_model(model_path, compile=False)
    except Exception:
        pass

    # Cara 2: custom_objects untuk InputLayer yang tidak dikenal
    try:
        import tensorflow.keras as keras

        class CompatInputLayer(keras.layers.InputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)
                kwargs.pop("optional", None)
                super().__init__(*args, **kwargs)

        return load_model(
            model_path,
            compile=False,
            custom_objects={"InputLayer": CompatInputLayer},
        )
    except Exception:
        pass

    # Cara 3: safe_mode=False (TF 2.16+)
    try:
        return load_model(model_path, compile=False, safe_mode=False)
    except Exception:
        pass

    # Cara 4: tf.saved_model atau h5 legacy
    return tf.keras.models.load_model(model_path, compile=False)


def load_disease_model(model_path: str, json_path: str):
    """Load model penyakit + config dari file .h5 dan .json"""
    model = _safe_load_model(model_path)
    with open(json_path, "r") as f:
        config = json.load(f)
    return model, config


def load_severity_model(model_path: str, json_path: str):
    """Load model tingkat keparahan + config dari file .h5 dan .json"""
    model = _safe_load_model(model_path)
    with open(json_path, "r") as f:
        config = json.load(f)
    return model, config


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocessing citra sesuai ResNet50:
    1. Resize ke 224x224
    2. Convert ke RGB
    3. preprocess_input ResNet50 (bukan rescale=1/255)
    """
    img = image.convert("RGB").resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
    arr = preprocess_input(arr)         # ResNet50 normalisasi
    return arr


def compute_entropy_ratio(predictions: np.ndarray) -> float:
    """
    Hitung rasio entropi output softmax.
    Nilai mendekati 1.0 → distribusi seragam → model tidak yakin → kemungkinan bukan daun kentang.
    """
    n = len(predictions)
    if n <= 1:
        return 0.0
    max_entropy = np.log(n)
    entropy = -np.sum(predictions * np.log(np.clip(predictions, 1e-10, 1.0)))
    return float(entropy / max_entropy)


def is_valid_input(predictions: np.ndarray, config: dict) -> tuple:
    """
    Validasi input:
    - entropy_ratio > 0.85 → bukan daun kentang / foto tidak jelas
    - confidence < threshold → tidak cukup yakin

    Returns: (valid: bool, reason: str)
    """
    confidence    = float(np.max(predictions))
    entropy_ratio = compute_entropy_ratio(predictions)
    threshold     = config.get("threshold", 0.60)

    if entropy_ratio > 0.85:
        return (
            False,
            f"Gambar tidak dikenali sebagai daun kentang "
            f"(skor ketidakpastian: {entropy_ratio:.2f}). "
            "Gunakan foto daun yang lebih jelas."
        )
    if confidence < threshold:
        return (
            False,
            f"Confidence terlalu rendah ({confidence*100:.1f}%) — "
            f"minimum {threshold*100:.0f}%. Coba gambar yang lebih jelas."
        )
    return True, ""


def predict_disease(model, config: dict, image: Image.Image) -> dict:
    """
    Prediksi jenis penyakit dari citra daun kentang.

    Urutan kelas sesuai output model (indeks 0-5):
        0 → Bacteria
        1 → Fungi
        2 → Healthy
        3 → Pest
        4 → Phytophthora
        5 → Virus

    Returns dict:
        valid        : bool
        reason       : str (alasan penolakan jika tidak valid)
        label        : str kelas terprediksi (None jika ditolak)
        confidence   : float (0-1)
        all_probs    : dict {label: prob}
        entropy_ratio: float
    """
    # ── Ambil urutan kelas dari JSON (kunci string "0","1",... atau list "classes") ──
    if "indices" in config:
        # Gunakan mapping eksplisit dari JSON, sort by key int
        classes = [config["indices"][str(i)] for i in range(len(config["indices"]))]
    else:
        classes = config["classes"]   # fallback list biasa

    arr   = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]   # shape (num_classes,)

    # Pastikan panjang cocok
    assert len(preds) == len(classes), (
        f"Output model ({len(preds)}) ≠ jumlah kelas di JSON ({len(classes)})"
    )

    valid, reason = is_valid_input(preds, config)
    top_idx       = int(np.argmax(preds))

    return {
        "valid":         valid,
        "reason":        reason,
        "label":         classes[top_idx] if valid else None,
        "confidence":    float(preds[top_idx]),
        "all_probs":     {cls: float(prob) for cls, prob in zip(classes, preds)},
        "entropy_ratio": compute_entropy_ratio(preds),
    }


def predict_severity(model, config: dict, image: Image.Image, disease_label: str) -> dict:
    """
    Prediksi tingkat keparahan penyakit.

    Urutan kelas sesuai output model (indeks 0-2):
        0 → Ringan
        1 → Sedang
        2 → Parah

    Jika disease_label == "Healthy", skip prediksi.

    Returns dict:
        valid      : bool
        label      : str
        confidence : float
        all_probs  : dict {label: prob}
    """
    if disease_label == "Healthy":
        return {
            "valid":      True,
            "reason":     "",
            "label":      "Tidak Ada",
            "confidence": 1.0,
            "all_probs":  {},
        }

    # ── Ambil urutan kelas ──────────────────────────────────────────────────
    if "indices" in config:
        classes = [config["indices"][str(i)] for i in range(len(config["indices"]))]
    else:
        classes = config["classes"]

    arr   = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]

    assert len(preds) == len(classes), (
        f"Output model severity ({len(preds)}) ≠ jumlah kelas di JSON ({len(classes)})"
    )

    valid, reason = is_valid_input(preds, config)
    top_idx       = int(np.argmax(preds))

    return {
        "valid":      valid,
        "reason":     reason,
        "label":      classes[top_idx],  # selalu tampilkan top class, confidence rendah tetap OK
        "confidence": float(preds[top_idx]),
        "all_probs":  {cls: float(prob) for cls, prob in zip(classes, preds)},
    }