"""
Bitcoin Sentiment Analyzer
==========================
Streamlit web application untuk klasifikasi sentimen tweet Bitcoin
menggunakan DistilBERT yang sudah di-fine-tune.

Cara jalankan:
    streamlit run app.py
"""
import os
import re
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# ====================================================================
# KONFIGURASI HALAMAN
# ====================================================================
st.set_page_config(
    page_title="Bitcoin Sentiment Analyzer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# KONSTANTA
# ====================================================================
# Model akan di-load dari HuggingFace Hub kalau folder lokal ./model tidak ada.
# Ini membuat app.py jalan baik di lokal (development) maupun di Streamlit Cloud (production).
LOCAL_MODEL_PATH = "./model"
HF_MODEL_PATH = "KevinKagusman/bitcoin-sentiment-distilbert"
MODEL_PATH = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else HF_MODEL_PATH

MAX_LENGTH = 64
LABELS = {0: "Non-Positive", 1: "Positive"}
LABEL_COLORS = {"Non-Positive": "#E76F51", "Positive": "#2A9D8F"}

# ====================================================================
# PREPROCESSING (HARUS SAMA DENGAN PIPELINE BERT DI TAHAP 1)
# ====================================================================
URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^RT[\s]+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_for_bert(text: str) -> str:
    """Pipeline preprocessing ringan untuk BERT (sama persis dgn yg dipakai saat training)."""
    if not isinstance(text, str):
        return ""
    s = text
    s = RT_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = MENTION_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


# ====================================================================
# LOAD MODEL (CACHED)
# ====================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Load model & tokenizer sekali saja, lalu di-cache di memory.
    @st.cache_resource membuat ini hanya dijalankan satu kali per sesi.
    Path bisa folder lokal atau HuggingFace Hub repo (format 'username/repo')."""
    is_local = os.path.isdir(MODEL_PATH)
    is_hf_hub = "/" in MODEL_PATH and not is_local
    if not (is_local or is_hf_hub):
        return None, None, f"Path model tidak valid: {MODEL_PATH}"
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        return None, None, f"Error: {e}"


# ====================================================================
# FUNGSI INFERENCE
# ====================================================================
def predict_single(text: str, model, tokenizer, device: str):
    """Prediksi sentimen satu kalimat. Return dict berisi label, confidence, & probabilities per class."""
    cleaned = clean_for_bert(text)
    if len(cleaned) < 2:
        return None
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "label": LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "prob_non_positive": float(probs[0]),
        "prob_positive": float(probs[1]),
        "cleaned_text": cleaned,
    }


def predict_batch(texts: list, model, tokenizer, device: str, batch_size: int = 32):
    """Batch prediction untuk efisiensi. Pakai progress callback."""
    results = []
    progress = st.progress(0.0)
    total = len(texts)
    for i in range(0, total, batch_size):
        chunk = texts[i : i + batch_size]
        cleaned = [clean_for_bert(t) for t in chunk]
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        for j in range(len(chunk)):
            results.append(
                {
                    "text": chunk[j],
                    "cleaned_text": cleaned[j],
                    "predicted_label": LABELS[int(preds[j])],
                    "confidence": float(probs[j, preds[j]]),
                    "prob_non_positive": float(probs[j, 0]),
                    "prob_positive": float(probs[j, 1]),
                }
            )
        progress.progress(min((i + batch_size) / total, 1.0))
    progress.empty()
    return results


# ====================================================================
# UI: SIDEBAR
# ====================================================================
with st.sidebar:
    st.markdown("## ₿ Bitcoin Sentiment Analyzer")
    st.markdown(
        "Aplikasi klasifikasi sentimen tweet Bitcoin menggunakan **DistilBERT** "
        "yang di-fine-tune pada 200.000 tweet."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(
        """
- **Arsitektur**: DistilBERT (distilbert-base-uncased)
- **Parameter**: ~67 juta
- **Kelas**: Positive vs Non-Positive
- **F1-macro (test)**: ~98%
- **Training data**: 160.000 tweet
- **Max length**: 64 token
"""
    )

    st.markdown("---")
    st.markdown("### 💡 Tips Penggunaan")
    st.markdown(
        """
1. Tulis atau paste tweet/kalimat tentang Bitcoin di kotak input
2. Klik tombol **Analyze**
3. Lihat hasil prediksi & confidence score
4. Untuk multiple input, gunakan tab **Batch Analysis**
"""
    )

    st.markdown("---")
    st.caption(
        "Final Project NLP COMP6885001 · BINUS University · "
        "Computer Science Department · 2026"
    )


# ====================================================================
# UI: MAIN
# ====================================================================
st.title("₿ Bitcoin Sentiment Analyzer")
st.markdown(
    "Analisis sentimen tweet atau kalimat tentang **Bitcoin** secara otomatis "
    "menggunakan model deep learning berbasis Transformer."
)

# Load model
with st.spinner("⏳ Memuat model DistilBERT (mungkin perlu beberapa detik di first run)..."):
    model, tokenizer, device = load_model_and_tokenizer()

if model is None:
    # device variable berisi pesan error kalau model None
    st.error(f"❌ Gagal memuat model. {device}")
    st.markdown(
        f"""
**Cara memperbaiki:**
1. Download `distilbert_btc_final.zip` dari Google Drive (hasil Tahap 3)
2. Extract isinya ke folder `{MODEL_PATH}/` (di samping file `app.py` ini)
3. Refresh halaman ini

Struktur folder yang benar:
```
streamlit_app/
├── app.py
├── requirements.txt
└── model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```
"""
    )
    st.stop()

# Status model
st.success(f"✅ Model loaded successfully | Device: **{device.upper()}**")

# ====================================================================
# TABS
# ====================================================================
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Analysis", "ℹ️ About"])

# ===================
# TAB 1: SINGLE
# ===================
with tab1:
    st.markdown("### Masukkan tweet atau kalimat tentang Bitcoin")

    # Sample examples (untuk usability)
    examples = [
        "Bitcoin is going to the moon! Best investment ever 🚀",
        "BTC crashed again, this is a terrible investment",
        "BTC price update: $52,000 today",
        "I love how decentralized Bitcoin is, very promising future",
        "Just bought more BTC during the dip",
        "Bitcoin is a scam, stay away from it",
    ]

    # Inisialisasi session_state untuk text input
    if "single_input" not in st.session_state:
        st.session_state.single_input = ""

    # Callback yang dijalankan SEBELUM widget di-render
    # → tidak melanggar aturan modifikasi session_state Streamlit
    def _apply_example():
        choice = st.session_state.get("example_select", "")
        if choice:
            st.session_state.single_input = choice

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "Input text:",
            height=120,
            placeholder="contoh: Bitcoin is going to the moon!",
            key="single_input",
        )
    with col2:
        st.markdown("**Quick examples:**")
        st.selectbox(
            "Pilih contoh →",
            [""] + examples,
            key="example_select",
            on_change=_apply_example,
        )

    if st.button("🔬 Analyze Sentiment", type="primary", use_container_width=True):
        # Input validation
        if not user_input or len(user_input.strip()) < 2:
            st.warning("⚠️ Mohon masukkan teks yang valid (minimal 2 karakter)")
        elif len(user_input) > 1000:
            st.warning("⚠️ Teks terlalu panjang. Maksimal 1000 karakter.")
        else:
            with st.spinner("Analyzing..."):
                t0 = time.time()
                result = predict_single(user_input, model, tokenizer, device)
                latency = (time.time() - t0) * 1000

            if result is None:
                st.error("Tidak dapat memproses teks setelah pembersihan. Coba teks lain.")
            else:
                # Display result
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                color = LABEL_COLORS[result["label"]]

                col_a.markdown(
                    f"""
<div style="background:{color}; padding:20px; border-radius:10px; text-align:center;">
    <p style="color:white; font-size:14px; margin:0;">PREDICTED SENTIMENT</p>
    <h2 style="color:white; margin:5px 0;">{result['label']}</h2>
</div>
""",
                    unsafe_allow_html=True,
                )

                col_b.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%",
                    delta=f"{(result['confidence']-0.5)*100:+.1f}% vs random",
                )
                col_c.metric("Latency", f"{latency:.0f} ms")

                # Probability bar chart
                st.markdown("#### Probability Distribution")
                fig, ax = plt.subplots(figsize=(8, 2.2))
                labels = ["Non-Positive", "Positive"]
                values = [result["prob_non_positive"], result["prob_positive"]]
                colors = [LABEL_COLORS[l] for l in labels]
                bars = ax.barh(labels, values, color=colors, edgecolor="black")
                for bar, v in zip(bars, values):
                    ax.text(
                        min(v + 0.02, 0.95),
                        bar.get_y() + bar.get_height() / 2,
                        f"{v*100:.2f}%",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                    )
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show preprocessing
                with st.expander("🔧 Preprocessing Details"):
                    st.markdown("**Original input:**")
                    st.code(user_input)
                    st.markdown("**After cleaning (BERT pipeline):**")
                    st.code(result["cleaned_text"])
                    st.caption(
                        "Pipeline: hapus URL, mention (@user), retweet marker. "
                        "Punctuation, casing, dan emoji dipertahankan agar BERT bisa belajar konteks."
                    )

# ===================
# TAB 2: BATCH
# ===================
with tab2:
    st.markdown("### Upload CSV untuk batch analysis")
    st.markdown(
        "Upload file CSV yang punya kolom berisi teks. "
        "Aplikasi akan memprediksi sentimen untuk setiap baris dan menyediakan hasil untuk diunduh."
    )

    uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dimuat: **{len(df_input):,}** baris, **{len(df_input.columns)}** kolom")

            # Preview
            with st.expander("👀 Preview data (5 baris pertama)"):
                st.dataframe(df_input.head(), use_container_width=True)

            # Pilih kolom teks
            text_columns = df_input.select_dtypes(include="object").columns.tolist()
            if not text_columns:
                st.error("Tidak ada kolom teks di file ini.")
            else:
                col_select = st.selectbox(
                    "Pilih kolom yang berisi teks untuk dianalisis:",
                    text_columns,
                )

                # Limit untuk demo (supaya tidak crash)
                max_rows = min(len(df_input), 10_000)
                n_rows = st.slider(
                    "Jumlah baris yang akan diproses:",
                    min_value=10,
                    max_value=max_rows,
                    value=min(500, max_rows),
                    step=10,
                )

                if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                    texts = df_input[col_select].fillna("").astype(str).tolist()[:n_rows]
                    t0 = time.time()
                    with st.spinner(f"Memproses {len(texts):,} teks..."):
                        results = predict_batch(texts, model, tokenizer, device, batch_size=32)
                    elapsed = time.time() - t0

                    df_results = pd.DataFrame(results)
                    st.success(
                        f"✅ Selesai dalam {elapsed:.1f}s "
                        f"(rata-rata {elapsed/len(texts)*1000:.1f} ms/teks)"
                    )

                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    n_pos = (df_results["predicted_label"] == "Positive").sum()
                    n_neg = (df_results["predicted_label"] == "Non-Positive").sum()
                    avg_conf = df_results["confidence"].mean()
                    col1.metric("Total processed", f"{len(df_results):,}")
                    col2.metric("Positive", f"{n_pos:,} ({n_pos/len(df_results)*100:.1f}%)")
                    col3.metric("Non-Positive", f"{n_neg:,} ({n_neg/len(df_results)*100:.1f}%)")
                    col4.metric("Avg confidence", f"{avg_conf*100:.1f}%")

                    # Distribution chart
                    st.markdown("#### Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    counts = df_results["predicted_label"].value_counts()
                    colors_bar = [LABEL_COLORS[l] for l in counts.index]
                    bars = ax.bar(counts.index, counts.values, color=colors_bar, edgecolor="black")
                    for bar, v in zip(bars, counts.values):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + len(df_results) * 0.01,
                            f"{v:,}\n({v/len(df_results)*100:.1f}%)",
                            ha="center",
                            fontsize=10,
                        )
                    ax.set_ylabel("Count")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Hasil tabel
                    st.markdown("#### Detailed Results")
                    st.dataframe(
                        df_results[
                            ["text", "predicted_label", "confidence", "prob_positive"]
                        ].style.format(
                            {"confidence": "{:.4f}", "prob_positive": "{:.4f}"}
                        ),
                        use_container_width=True,
                        height=400,
                    )

                    # Download button
                    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download hasil sebagai CSV",
                        data=csv_bytes,
                        file_name=f"sentiment_results_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")

# ===================
# TAB 3: ABOUT
# ===================
with tab3:
    st.markdown("### 📚 Tentang Aplikasi Ini")
    st.markdown(
        """
Aplikasi ini adalah hasil implementasi *final project* mata kuliah
**Natural Language Processing (COMP6885001)** di BINUS University.

#### Pipeline
1. **Input** — User memasukkan tweet atau kalimat tentang Bitcoin
2. **Preprocessing** — Pembersihan ringan: hapus URL, @mention, dan retweet marker
3. **Tokenization** — DistilBERT WordPiece tokenizer (max 64 token)
4. **Inference** — Forward pass DistilBERT → softmax probabilities
5. **Output** — Label prediksi (Positive / Non-Positive) + confidence score

#### Dataset
- **Sumber**: [Bitcoin Sentiment Analysis Twitter Dataset (Kaggle)](https://www.kaggle.com/datasets/gautamchettiar/bitcoin-sentiment-analysis-twitter-data)
- **Total**: 1.000.025 tweet (kami menggunakan stratified sample 200.000)
- **Label**: binary (Positive vs Non-Positive), berasal dari TextBlob polarity scoring

#### Model Performance (Test Set, n=20.000)
| Model | Accuracy | F1-macro |
|---|---|---|
| Multinomial Naive Bayes (baseline) | 82.3% | 82.1% |
| Logistic Regression (baseline) | 94.8% | 94.8% |
| **DistilBERT (fine-tuned)** | **~98.2%** | **~98.2%** |

#### Limitations
- Label asli dataset dihasilkan secara otomatis oleh TextBlob, sehingga model
  belajar meniru pola TextBlob (bukan ground truth manusia).
- Dataset hanya berisi tweet berbahasa Inggris.
- Klasifikasi binary (positive vs non-positive); kelas neutral & negative digabung.

#### Tech Stack
- **Python 3.10+**, **PyTorch 2.x**, **Hugging Face Transformers 4.44+**
- **Streamlit** untuk web UI
- **scikit-learn** untuk baseline models & evaluation
"""
    )
