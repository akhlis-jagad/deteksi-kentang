import os
import requests

# ══════════════════════════════════════════════════════════════════════════════
# MODEL OPENROUTER
# ══════════════════════════════════════════════════════════════════════════════
OPENROUTER_MODELS = [
    'google/gemini-2.0-flash-001',
]

_DISEASE_DESC = {
    "Bacteria":     "penyakit bakteri (Bacterial Wilt / busuk bakteri)",
    "Fungi":        "penyakit jamur (Fungi) pada daun kentang",
    "Pest":         "kerusakan akibat serangan hama pada daun kentang",
    "Phytophthora": "penyakit busuk daun (Phytophthora infestans)",
    "Virus":        "penyakit virus (PVY/PVX/PLRV) ditularkan kutu kebul",
}


def init_gemini(api_key: str) -> dict:
    """Inisialisasi — menyimpan API key OpenRouter."""
    if not api_key or api_key.startswith("ISI_"):
        print("[❌ ERROR] API KEY OpenRouter belum diisi!")
        raise ValueError(
            "API KEY OpenRouter belum diisi!\n"
            "Isi di .streamlit/secrets.toml:\n"
            'GEMINI_API_KEY = "sk-or-v1-..."'
        )
    print("[✅ SUCCESS] Gemini/OpenRouter API berhasil diinisialisasi")
    return {"api_key": api_key, "configured": True}


def get_gemini_recommendation(
    config: dict,
    disease_label: str,
    disease_confidence: float,
    severity_label: str,
    severity_confidence: float,
) -> str:
    """Generate rekomendasi dari OpenRouter."""

    print(f"\n[📋 REQUEST] Generating rekomendasi untuk: {disease_label} ({severity_label})")

    if disease_label == "Healthy":
        prompt = (
            "Tanaman kentang terdeteksi dalam kondisi sehat. "
            "Berikan 3 tips singkat perawatan daun kentang agar tetap sehat "
            "dalam Bahasa Indonesia yang sederhana dan ramah petani. "
            "Format poin pendek, maksimal 100 kata."
        )
    else:
        disease_desc = _DISEASE_DESC.get(disease_label, f"penyakit {disease_label}")

        prompt = f"""Kamu adalah asisten pertanian untuk membantu petani kentang.

Sistem mendeteksi penyakit pada daun kentang:
- Jenis Penyakit   : {disease_label} — {disease_desc}
- Tingkat Keparahan: {severity_label}

Berikan rekomendasi penanganan dengan aturan berikut:
1. Gunakan bahasa sederhana, mudah dipahami, dan ramah petani.
2. kasih penjelasan singkat tentang penyakit tersebut
2. jawaban menggunakan hasil dari hasil klasifikasi dan pilih salah satu dari hasil klasifikasi:
   - Penanganan untuk tingkat ringan
   - Penanganan untuk tingkat sedang
   - Penanganan untuk tingkat parah
3. kasih tahu jenis pestisida yang cocok , atau pestisida alami untuk digunakan tapi kasih note (tetap gunakan pestisida dari seorang ahli)
Jawaban jelas, dan langsung ke inti."""

    # ── Panggil OpenRouter ────────────────────────────────────────────────────
    api_key = config.get("api_key")
    errors = []

    print("\n[🔄 INFO] Menghubungi API OpenRouter untuk rekomendasi AI...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://kentangsehat.ai",
        "X-Title": "KentangSehat AI",
        "Content-Type": "application/json"
    }

    for model_name in OPENROUTER_MODELS:
        print(f"[⏳ TRYING] Model: {model_name}")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )

            response_data = response.json()

            if response.status_code == 200 and "choices" in response_data:
                print(f"[✅ SUCCESS] Respons berhasil dari {model_name}")
                return response_data["choices"][0]["message"]["content"]
            else:
                err_msg = response_data.get("error", {}).get("message", "Unknown error")
                error_detail = f"{model_name}: {err_msg}"
                errors.append(error_detail)
                print(f"[⚠️  API ERROR] Status {response.status_code} - {error_detail}")

        except Exception as e:
            error_detail = f"{model_name}: {str(e)[:60]}"
            errors.append(error_detail)
            print(f"[❌ EXCEPTION] {error_detail}")
            continue

    # ── Semua model gagal ─────────────────────────────────────────────────────
    print(f"[❌ FAILED] Semua model gagal ({len(errors)} error).")
    return "⚠️ **Gagal mendapatkan rekomendasi.** Silakan coba beberapa saat lagi."