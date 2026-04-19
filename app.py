import os
import streamlit as st
from PIL import Image

from huggingface_hub import hf_hub_download

def download_models():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/disease_model.h5"):
        hf_hub_download(
            repo_id="akhlis-jagad/deteksi-kentang",
            filename="disease_model.h5",
            local_dir="models",
            local_dir_use_symlinks=False,
        )
    if not os.path.exists("models/severity_model.h5"):
        hf_hub_download(
            repo_id="akhlis-jagad/deteksi-kentang",
            filename="severity_model.h5",
            local_dir="models",
            local_dir_use_symlinks=False,
        )

# ── Page config (HARUS paling atas) ──────────────────────────────────────────
st.set_page_config(
    page_title="KentangSehat AI",
    page_icon="&#129364;",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.predict import (
    load_disease_model, load_severity_model,
    predict_disease, predict_severity,
)
from utils.gemini_helper import init_gemini, get_gemini_recommendation

# ══════════════════════════════════════════════════════════════════════════════
# PATH & API KEY
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH  = os.path.join(BASE_DIR, "models", "disease_model.h5")
DISEASE_JSON_PATH   = os.path.join(BASE_DIR, "models", "disease_model.json")
SEVERITY_MODEL_PATH = os.path.join(BASE_DIR, "models", "severity_model.h5")
SEVERITY_JSON_PATH  = os.path.join(BASE_DIR, "models", "severity_model.json")

# Baca API key dari .streamlit/secrets.toml (AMAN)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# MAPPING VISUAL
# ══════════════════════════════════════════════════════════════════════════════
# Indeks disease:  0=Bacteria 1=Fungi 2=Healthy 3=Pest 4=Phytophthora 5=Virus
DISEASE_COLORS = {
    "Bacteria":     "#f97316",
    "Fungi":        "#a855f7",
    "Healthy":      "#16a34a",
    "Pest":         "#ea580c",
    "Phytophthora": "#dc2626",
    "Virus":        "#e11d48",
}
# Gunakan HTML entity agar tidak ada emoji di string HTML
DISEASE_ICONS_HTML = {
    "Bacteria":     "&#129440;",
    "Fungi":        "&#127812;",
    "Healthy":      "&#10003;",
    "Pest":         "&#128027;",
    "Phytophthora": "&#9888;",
    "Virus":        "&#128308;",
}
# Indeks severity: 0=Ringan 1=Sedang 2=Parah
SEVERITY_COLORS = {
    "Tidak Ada": "#16a34a",
    "Ringan":    "#22c55e",
    "Sedang":    "#f59e0b",
    "Parah":     "#dc2626",
}
SEVERITY_SCORES = {"Tidak Ada": 0, "Ringan": 33, "Sedang": 66, "Parah": 100}
NO_SEVERITY = {"Healthy"}  # hanya Healthy yang tidak pakai keparahan

# ══════════════════════════════════════════════════════════════════════════════
# LOAD CSS (utf-8 agar tidak error di Windows)
# ══════════════════════════════════════════════════════════════════════════════
def load_css():
    p = os.path.join(BASE_DIR, "css", "style.css")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Override global: paksa light mode di seluruh file uploader ───────────────
st.markdown("""
<style>
/* Paksa seluruh area file uploader pakai light color-scheme */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] * {
    color-scheme: light !important;
    forced-color-adjust: none !important;
}
/* Teks utama dropzone */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {
    color: #14532d !important;
    background: transparent !important;
}
/* Background container dropzone — hilangkan dark bg */
[data-testid="stFileUploadDropzone"] {
    background-color: #f0fdf4 !important;
    background: linear-gradient(160deg, #ffffff 60%, #f0fdf4 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS (cached agar tidak reload tiap interaksi)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_all_models():
    dm, dc = load_disease_model(DISEASE_MODEL_PATH, DISEASE_JSON_PATH)
    sm, sc = load_severity_model(SEVERITY_MODEL_PATH, SEVERITY_JSON_PATH)
    return dm, dc, sm, sc

def load_gemini():
    # Tidak di-cache karena hanya set API key + return dict config
    return init_gemini(GEMINI_API_KEY)

model_ok  = False
model_err = ""
disease_model = disease_cfg = severity_model = severity_cfg = gemini_model = None

try:
    with st.spinner("Memuat model ResNet50..."):
        disease_model, disease_cfg, severity_model, severity_cfg = load_all_models()
        gemini_model = load_gemini()
    model_ok = True
except FileNotFoundError as e:
    model_err = str(e)
except Exception as e:
    model_err = str(e)

n_disease  = len(disease_cfg.get("indices", disease_cfg.get("classes", []))) if model_ok else 6
n_severity = len(severity_cfg.get("indices", severity_cfg.get("classes", []))) if model_ok else 3

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
if not model_ok:
    badge_err = '<span style="color:#fca5a5;font-weight:700">&#10005; Model Gagal</span>'
else:
    badge_err = ''

# Load logo sebagai base64 agar bisa ditampilkan di HTML
def _logo_html(size=30):
    import base64
    logo_path = os.path.join(BASE_DIR, 'assets', 'LOGO KENTANG.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:{size}px;height:{size}px;object-fit:contain;border-radius:8px"/>'
    return '&#129364;'  # fallback emoji kentang

st.markdown(
    '<div class="navbar">'
    '<div class="nb-brand">'
    '<div class="nb-logo">'
    + _logo_html(28)
    + '</div>'
    '<span>Kentang<span>Sehat</span>'
    '</div>'
    '<div class="nb-links">'
    '<span> <b style="color: green;"> Sistem Deteksi Penyakit Daun Kentang — Convolutional Neural Network </b> </span>'
    
    '</div>'
    + (('<div class="nb-btn" style="background:#ef4444">' + badge_err + '</div>') if not model_ok else '')
    +     '</div>',
    unsafe_allow_html=True,
)

# Spacer agar konten tidak tertutup navbar fixed
st.markdown('<div class="navbar-spacer"></div>', unsafe_allow_html=True)

if model_err:
    st.error(f"Gagal memuat model: {model_err}")
    st.info("Pastikan file `.h5` dan `.json` ada di folder `models/`")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero">'
    '<div class="hero-blob1"></div>'
    '<div class="hero-blob2"></div>'
    '<div class="hero-inner">'
    '<h1 class="hero-title">'
    '<span class="hero-title-green">Deteksi Penyakit Daun Kentang</span><br>'
    '<span class="hero-title-green">denganMenggunakan CNN ResNet50 </span>'
    '</h1>'
    '<p class="hero-desc">'
    'Sistem cerdas berbasis <em>deep learning</em> dengan arsitektur CNN ResNet50 untuk mendeteksi '
    'penyakit pada daun kentang secara cepat dan akurat. Lindungi hasil panen Anda dengan dukungan '
    'teknologi kecerdasan buatan terkini.'
    '</p>'
  
    '<div class="hero-stats">'
    '<div class="stat-item">'
    '<div class="stat-number">' + str(n_disease) + '</div>'
    '<div class="stat-label">Kelas Penyakit</div>'
    '</div>'
    '<div class="stat-item">'
    '<div class="stat-number">95%+</div>'
    '<div class="stat-label">Akurasi</div>'
    '</div>'
    '<div class="stat-item">'
    '<div class="stat-number">&lt;2s</div>'
    '<div class="stat-label">Waktu Proses</div>'
    '</div>'
    '<div class="stat-item">'
    '<div class="stat-number">' + str(n_severity) + '</div>'
    '<div class="stat-label">Tingkat Keparahan</div>'
    '</div>'
    '</div>'
    '</div>'  # /hero-inner
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TECHNOLOGY SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="tech-section">'
    '<h2><span class="sec-title-acc">Teknologi Di Balik Sistem</span></h2>'
    '<p class="sec-desc">'
    'Sistem ini dibangun dengan teknologi terkini berbasis CNN untuk memberikan hasil deteksi '
    'penyakit secara akurat, serta dilengkapi rekomendasi penanganan yang informatif melalui Gemini AI.'
    '</p>'
    '<div class="tech-grid">'

    '<div class="tech-card">'
    '<div class="tech-icon">&#129504;</div>'
    '<div class="tech-name">ResNet50</div>'
    '<div class="tech-desc">'
    'Menggunakan arsitektur ResNet50 sebagai base model dengan metode transfer learning '
    'dari dataset ImageNet untuk melakukan klasifikasi penyakit daun kentang secara akurat.'
    '</div>'
    '<div class="tech-tags">'
    '<span class="tech-tag">Transfer Learning</span>'
    '<span class="tech-tag">Residual Network</span>'
    '<span class="tech-tag">Deep Learning</span>'
    '</div>'
    '</div>'

    '<div class="tech-card">'
    '<div class="tech-icon">&#9881;&#65039;</div>'
    '<div class="tech-name">Streamlit</div>'
    '<div class="tech-desc">'
    'Streamlit digunakan sebagai framework untuk mengembangkan antarmuka sistem '
    'berbasis web yang interaktif, sehingga memudahkan pengguna dalam mengakses '
    'dan melakukan proses deteksi penyakit daun kentang.'
    '</div>'
    '<div class="tech-tags">'
    '<span class="tech-tag">Web Interface</span>'
    '<span class="tech-tag">Interactive UI</span>'
    '<span class="tech-tag">Python</span>'
    '</div>'
    '</div>'

    '<div class="tech-card featured">'
    '<div class="tech-icon">&#128451;&#65039;</div>'
    '<div class="tech-name">Dataset Daun Kentang</div>'
    '<div class="tech-desc">'
    'Model dilatih dengan dataset citra daun kentang dari Mendeley Data, '
    'dikategorikan ke dalam 6 kelas: Bacteria, Fungi, Healthy, Pest, '
    'Phytophthora, dan Virus dengan augmentasi data.'
    '</div>'
    '<div class="tech-tags">'
    '<span class="tech-tag">Augmented</span>'
    '<span class="tech-tag">Labeled</span>'
    '<span class="tech-tag">3.076 Citra</span>'
    '</div>'
    '</div>'

    '</div>'  # /tech-grid
    '</div>', # /tech-section
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="detect-section" id="section-deteksi">'
    '<div class="detect-blob1"></div>'
    '<div class="detect-blob2"></div>'
    '<h2 class="detect-title">Mulai Deteksi Sekarang</h2>'
    '<p class="detect-desc">'
    'Unggah gambar daun kentang dan dapatkan hasil analisis berbasis AI secara instan dengan tingkat akurasi tinggi serta rekomendasi penanganan otomatis.'
    '</p>',
    unsafe_allow_html=True,
)

# ── File uploader Streamlit — satu kotak, CSS styled ─────────────────────────
# Inject CSS tepat sebelum uploader (highest specificity)
# Sembunyikan nama file via CSS inline
st.markdown("""
<style>
/* ══ Sembunyikan nama file setelah upload ══ */
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploader"] section ~ div { display: none !important; }

/* ══ PAKSA light mode pada SEMUA elemen dropzone ══ */
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] *,
[data-testid="stFileUploadDropzone"] *::before,
[data-testid="stFileUploadDropzone"] *::after {
    color-scheme: light only !important;
}

/* ══ Dropzone container utama ══ */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(160deg, #ffffff 60%, #f0fdf4 100%) !important;
    border: 2px dashed #86efac !important;
    border-radius: 20px !important;
    padding: 32px 24px !important;
    min-height: 140px !important;
    box-shadow: none !important;
}

/* ══ Paksa semua teks di dalam dropzone jadi gelap ══ */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] label,
[data-testid="stFileUploadDropzone"] section,
[data-testid="stFileUploadDropzone"] small {
    color: #14532d !important;
    background: transparent !important;
}

/* ══ Teks hint kecil ══ */
[data-testid="stFileUploadDropzone"] small {
    color: #4b5563 !important;
}

/* ══ SVG icon — hijau ══ */
[data-testid="stFileUploadDropzone"] svg {
    color: #22c55e !important;
    fill: #22c55e !important;
}
[data-testid="stFileUploadDropzone"] svg path {
    fill: #22c55e !important;
    stroke: none !important;
}

/* ══ Tombol Browse Files ══ */
[data-testid="stFileUploadDropzone"] button,
button[data-testid="baseButton-secondary"] {
    background: #22c55e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 14px rgba(34,197,94,0.3) !important;
}
[data-testid="stFileUploadDropzone"] button:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: #16a34a !important;
}
</style>
<script>
(function fixDropzone() {
    function applyAll() {
        const dz = document.querySelector('[data-testid="stFileUploadDropzone"]');
        if (!dz) return;

        // 1. Paksa background putih/hijau muda
        dz.style.setProperty('background', 'linear-gradient(160deg,#ffffff 60%,#f0fdf4 100%)', 'important');
        dz.style.setProperty('border', '2px dashed #86efac', 'important');
        dz.style.setProperty('border-radius', '20px', 'important');
        dz.style.setProperty('padding', '32px 24px', 'important');
        dz.style.setProperty('color-scheme', 'light', 'important');

        // 2. Paksa SEMUA child elements: background transparan, teks gelap
        dz.querySelectorAll('*').forEach(el => {
            const tag = el.tagName.toLowerCase();
            el.style.setProperty('background-color', 'transparent', 'important');
            el.style.setProperty('color-scheme', 'light', 'important');

            if (tag === 'small') {
                el.style.setProperty('color', '#4b5563', 'important');
            } else if (tag === 'button') {
                el.style.setProperty('background', '#22c55e', 'important');
                el.style.setProperty('background-color', '#22c55e', 'important');
                el.style.setProperty('color', '#ffffff', 'important');
                el.style.setProperty('border-radius', '50px', 'important');
                el.style.setProperty('border', 'none', 'important');
                el.style.setProperty('font-weight', '700', 'important');
                el.style.setProperty('padding', '10px 28px', 'important');
            } else if (tag === 'svg') {
                el.style.setProperty('color', '#22c55e', 'important');
                el.style.setProperty('fill', '#22c55e', 'important');
            } else if (tag === 'path') {
                el.style.setProperty('fill', '#22c55e', 'important');
            } else {
                el.style.setProperty('color', '#14532d', 'important');
            }
        });
    }

    // Jalankan langsung
    applyAll();

    // Observe perubahan DOM (Streamlit sering re-render)
    new MutationObserver(applyAll).observe(document.body, { childList: true, subtree: true });

    // Fallback dengan delay
    [200, 500, 1000, 2000, 3500].forEach(t => setTimeout(applyAll, t));
})();
</script>
""", unsafe_allow_html=True)

col_fl, col_fc, col_fr = st.columns([1, 3, 1])
with col_fc:
    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)  # /detect-section

# ── Popup: gambar sedang diunggah ────────────────────────────────────────────
if uploaded:
    # Tampilkan popup upload sebentar
    upload_popup = st.empty()
    upload_popup.markdown("""
<div class="popup-overlay">
  <div class="popup-card">
    <div class="popup-spinner"></div>
    <div class="popup-title">Gambar Sedang Diunggah</div>
    <div class="popup-sub">Memproses gambar daun kentang...</div>
  </div>
</div>
""", unsafe_allow_html=True)
    import time, base64
    from io import BytesIO
    time.sleep(0.6)
    upload_popup.empty()   # hilangkan popup

    img_preview = Image.open(uploaded).convert("RGB")
    img_preview.thumbnail((800, 400), Image.LANCZOS)

    buf = BytesIO()
    img_preview.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_w, img_h = img_preview.size

    col_pl, col_pc, col_pr = st.columns([1, 3, 1])
    with col_pc:
        st.markdown(
            '<div class="preview-card">'
            '<div class="preview-card-label">&#128247; Preview Gambar</div>'
            '<div class="preview-img-wrap">'
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'     width="{img_w}" height="{img_h}" '
            '     style="max-height:200px;width:auto;max-width:100%;'
            '            object-fit:cover;border-radius:10px;'
            '            display:block;margin:0 auto"/>'
            '</div>'
            '<div class="preview-card-filename">' + uploaded.name + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Tombol analisis ───────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    run = st.button(
        "Analisis Gambar",
        use_container_width=True,
        disabled=(not uploaded),
    )

# ══════════════════════════════════════════════════════════════════════════════
# PREDIKSI & HASIL
# ══════════════════════════════════════════════════════════════════════════════
if uploaded and run:
    image = Image.open(uploaded)

    # 1. Prediksi penyakit — dengan popup klasifikasi
    classify_popup = st.empty()
    classify_popup.markdown("""
<div class="popup-overlay">
  <div class="popup-card">
    <div class="popup-spinner"></div>
    <div class="popup-title">Sedang Diklasifikasi</div>
    <div class="popup-sub">Model CNN ResNet50 sedang menganalisis citra daun kentang...</div>
    <div class="popup-steps">
      <span class="popup-step active">&#9679; Ekstraksi fitur</span>
      <span class="popup-step">&#9679; Prediksi kelas</span>
      <span class="popup-step">&#9679; Hitung confidence</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    try:
        d_res = predict_disease(disease_model, disease_cfg, image)
    except Exception as e:
        classify_popup.empty()
        st.error(f"Error prediksi penyakit: {e}")
        d_res = None
    classify_popup.empty()

    if d_res is None:
        pass

    elif not d_res["valid"]:
        # Notif inline — pakai kolom tengah agar rapi
        col_nl, col_nc, col_nr = st.columns([1, 3, 1])
        with col_nc:
            st.markdown("""
<div class="notif-invalid">
  <div class="notif-icon">&#127807;</div>
  <div class="notif-title">Bukan Gambar Daun Kentang</div>
  <div class="notif-body">
    Sistem tidak dapat mengenali gambar yang diunggah.<br>
    Pastikan foto menampilkan <b>daun kentang</b> dengan pencahayaan yang cukup dan fokus pada daun.
  </div>
  <div class="notif-tips">
    <div class="notif-tip">&#9679; Fokus pada daun, bukan batang atau tanah</div>
    <div class="notif-tip">&#9679; Pencahayaan cukup, tidak terlalu gelap</div>
    <div class="notif-tip">&#9679; Jarak dekat agar detail terlihat jelas</div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:
        # Langkah 1: hasil penyakit
        d_label = d_res["label"]
        d_conf  = d_res["confidence"]
        d_color = DISEASE_COLORS.get(d_label, "#16a34a")
        d_icon  = DISEASE_ICONS_HTML.get(d_label, "&#127807;")

        # Langkah 2: keparahan — skip hanya jika Healthy
        if d_label in NO_SEVERITY:
            s_res = {"valid": True, "label": "Tidak Ada", "confidence": 1.0, "all_probs": {}}
        else:
            severity_popup = st.empty()
            severity_popup.markdown("""
<div class="popup-overlay">
  <div class="popup-card">
    <div class="popup-spinner"></div>
    <div class="popup-title">Menganalisis Tingkat Keparahan</div>
    <div class="popup-sub">Menghitung luas bercak dan kerusakan daun...</div>
    <div class="popup-steps">
      <span class="popup-step active">&#9679; Segmentasi area terinfeksi</span>
      <span class="popup-step">&#9679; Klasifikasi keparahan</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            try:
                s_res = predict_severity(severity_model, severity_cfg, image, d_label)
                s_res["valid"] = True  # confidence rendah tetap tampil
            except Exception as e:
                    severity_popup.empty()
                    s_res = {
                        "valid": True,
                        "label": "Ringan",
                        "confidence": 0.0,
                        "all_probs": {},
                        "reason": str(e),
                    }
            severity_popup.empty()

        s_label = s_res["label"]
        s_conf  = s_res["confidence"]
        s_color = SEVERITY_COLORS.get(s_label, "#6b7280")
        s_score = SEVERITY_SCORES.get(s_label, 0)

        if s_label == "Tidak Ada":
            s_sub = "Tanaman terdeteksi sehat &#9989;"
        elif not s_res.get("valid", True):
            s_sub = s_res.get("reason", "")
        else:
            s_sub = (
                'Score: <b style="color:#111">' + str(s_score) + '%</b>'
                + ' &nbsp;&middot;&nbsp; '
                + 'Conf: <b style="color:#111">' + f'{s_conf*100:.1f}%' + '</b>'
            )

        # ── Results Section ───────────────────────────────────────────────────
        st.markdown(
            '<div class="res-title">&#128300; Hasil Klasifikasi</div>',
            unsafe_allow_html=True,
        )

        # Dua kartu hasil
        st.markdown(
            '<div class="result-grid">'

            '<div class="result-card" style="border-color:' + d_color + '">'
            '<div class="rc-badge">&#129440; &nbsp; Nama Penyakit</div>'
            '<div class="rc-value" style="color:' + d_color + '">'
            + d_icon + ' ' + d_label +
            '</div>'
            '<div class="rc-conf">Confidence: <b>' + f'{d_conf*100:.1f}%' + '</b></div>'
            '<div class="bar-wrap">'
            '<div class="bar-inner" style="width:' + f'{d_conf*100:.1f}%' + ';background:' + d_color + '"></div>'
            '</div>'
            '</div>'

            '<div class="result-card" style="border-color:' + s_color + '">'
            '<div class="rc-badge">&#127777;&#65039; &nbsp; Tingkat Keparahan</div>'
            '<div class="rc-value" style="color:' + s_color + '">' + s_label + '</div>'
            '<div class="rc-conf">' + s_sub + '</div>'
            '<div class="bar-wrap">'
            '<div class="bar-inner" style="width:' + str(s_score) + '%;background:' + s_color + '"></div>'
            '</div>'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )

        # Distribusi probabilitas
        if d_res.get("all_probs"):
            rows = ""
            for cls, prob in sorted(d_res["all_probs"].items(), key=lambda x: -x[1]):
                top    = cls == d_label
                bc     = DISEASE_COLORS.get(cls, "#bbf7d0") if top else "#d1fae5"
                nclass = "prob-name top" if top else "prob-name"
                pclass = "prob-pct top"  if top else "prob-pct"
                prefix = "&#9658; " if top else "&nbsp;&nbsp; "
                rows += (
                    '<div class="prob-row">'
                    '<span class="' + nclass + '">' + prefix + cls + '</span>'
                    '<div class="prob-bw">'
                    '<div class="prob-bf" style="width:' + f'{prob*100:.1f}%' + ';background:' + bc + '"></div>'
                    '</div>'
                    '<span class="' + pclass + '">' + f'{prob*100:.1f}%' + '</span>'
                    '</div>'
                )
            st.markdown(
                '<div class="prob-box">'
                '<div class="prob-box-title">Distribusi Probabilitas Semua Kelas</div>'
                + rows +
                '</div>',
                unsafe_allow_html=True,
            )

        # Loading inline di dalam container gemini
        gemini_slot = st.empty()
        gemini_slot.markdown(
            '<div class="gemini-box">'
            '<div class="gemini-hdr">&#129302; &nbsp; Rekomendasi &amp; Penjelasan AI</div>'
            '<div class="gemini-loading">'
            '<div class="gemini-spinner"></div>'
            '<span>Gemini AI sedang menganalisis dan menyusun rekomendasi...</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        try:
            rec = get_gemini_recommendation(
                gemini_model, d_label, d_conf, s_label, s_conf
            )
        except Exception as e:
            rec = f"Gagal menghubungi AI: {e}"
        gemini_slot.empty()


        print("resc:", rec)  # debug output di console
        konten_html = (
            '<div class="gemini-box">'
            '<div class="gemini-hdr">&#129302; &nbsp; Rekomendasi &amp; Penjelasan AI</div>'
            '<div class="gemini-body">\n\n'  # <-- Tambahkan dua enter di sini
            + rec +
            '\n\n</div>'                    # <-- Tambahkan dua enter di sini
            '</div>'
        )
        st.markdown(konten_html, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="site-footer">'
    '<div class="footer-inner">'
    '<div class="footer-brand"> <span>KentangSehat</div>'
    '<div class="footer-tagline">Sistem Deteksi Penyakit Tanaman Kentang Berbasis CNN ResNet50 - 2026 </div>'
   
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)