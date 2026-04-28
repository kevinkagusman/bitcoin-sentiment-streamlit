# Bitcoin Sentiment Analyzer — Streamlit App

Aplikasi web untuk klasifikasi sentimen tweet Bitcoin menggunakan DistilBERT
yang sudah di-fine-tune. Bagian dari Final Project NLP COMP6885001.

## Fitur

- **Single Prediction** — Input satu kalimat, dapat label + confidence + visualisasi probabilitas
- **Batch Analysis** — Upload CSV, dapat hasil klasifikasi semua baris sekaligus + summary stats + download
- **Preprocessing transparan** — User bisa lihat hasil cleaning text sebelum masuk model
- **Model info & limitations** — Halaman About yang jujur soal cara kerja & batasan model

## Struktur Folder

```
streamlit_app/
├── app.py                  # aplikasi utama
├── requirements.txt        # dependencies untuk Streamlit Cloud
├── .streamlit/
│   └── config.toml         # theme & config
├── model/                  # <-- ISI DENGAN MODEL DISTILBERT (lihat di bawah)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
└── README.md
```

## Persiapan: Letakkan Model di Folder yang Benar

Setelah selesai run notebook DistilBERT di Colab, kamu punya file
`distilbert_btc_final.zip` di Google Drive. **Extract isinya ke folder `./model/`**
di samping `app.py`.

```bash
# Misalnya kamu sudah download zip-nya ke folder streamlit_app/
cd streamlit_app
unzip distilbert_btc_final.zip -d model/

# Verifikasi struktur
ls model/
# config.json  model.safetensors  tokenizer.json  tokenizer_config.json  vocab.txt  ...
```

## Cara 1: Jalankan Lokal di Komputer

```bash
# 1. Pastikan Python 3.10+ terinstall
python --version

# 2. Buat virtual env (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser di `http://localhost:8501`.

**Saat first run**, model akan di-load ke RAM (perlu ~5-10 detik).
Run berikutnya akan instant berkat caching `@st.cache_resource`.

## Cara 2: Jalankan di Google Colab (untuk Demo Video)

Kalau komputer kamu tidak kuat menjalankan PyTorch, kamu bisa jalankan langsung di Colab
dengan tunneling:

```python
# Cell 1
!pip install -q streamlit transformers==4.44.2 torch
!npm install -g localtunnel

# Cell 2 - upload model.zip dan extract
from google.colab import files
files.upload()  # pilih distilbert_btc_final.zip
!unzip -q distilbert_btc_final.zip -d model/

# Cell 3 - upload app.py
files.upload()  # pilih app.py

# Cell 4 - jalankan
!streamlit run app.py &>/dev/null &
!npx localtunnel --port 8501
# klik link yang muncul
```

## Cara 3: Deploy ke Streamlit Cloud (URL Publik untuk Demo Video)

Streamlit Community Cloud gratis. Cocok untuk demo profesional ke dosen.

### Langkah:

1. **Push ke GitHub** — Buat repo baru, push semua file:
   ```bash
   git init
   git add app.py requirements.txt README.md .streamlit/
   git commit -m "Initial commit"
   git push
   ```

   ⚠️ **JANGAN push folder `model/`** — ukurannya >250MB dan GitHub batas 100MB per file.

2. **Upload model ke HuggingFace Hub** (gratis):
   - Buat akun di https://huggingface.co
   - Buat new model repo (private boleh)
   - Push folder `model/` ke sana via `huggingface_hub` library
   - Edit `app.py`, ganti `MODEL_PATH = "./model"` jadi nama repo HF kamu

3. **Connect ke Streamlit Cloud**:
   - Sign up di https://streamlit.io/cloud (pakai GitHub login)
   - Click "New app" → pilih repo & branch → main file: `app.py`
   - Tunggu ~5 menit deployment selesai
   - Dapat URL publik: `https://<nama-app>.streamlit.app`

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Folder model tidak ditemukan` | Extract `distilbert_btc_final.zip` ke folder `./model/` |
| `CUDA out of memory` | App otomatis fallback ke CPU; ini hanya warning |
| Model loading lambat (>30 detik) | Normal di first run; berikutnya instant karena di-cache |
| Tampilan rusak di mobile | Streamlit memang lebih nyaman di desktop; tetap berfungsi di mobile tapi layout sempit |
| Error saat batch analysis CSV | Pastikan kolom yg dipilih berisi teks (bukan angka/tanggal) |

## Untuk Demo Video

Saran flow demo (< 5 menit):

1. **Intro singkat** (30 detik) — *"Aplikasi ini melakukan klasifikasi sentimen tweet Bitcoin menggunakan DistilBERT..."*
2. **Single prediction demo** (1 menit) — Coba 3-4 contoh kalimat (positive, negatif, netral). Tunjukkan probability bar.
3. **Show preprocessing detail** (30 detik) — Klik expander, jelaskan apa yang di-clean.
4. **Batch analysis demo** (1.5 menit) — Upload CSV (siapkan file dengan 50-100 tweet), tunjukkan summary stats & download hasil.
5. **About page** (30 detik) — Briefly mention pipeline, dataset, performance, limitations.
6. **Closing** (30 detik) — Kesimpulan singkat.

## Spesifikasi Performance

- **Single prediction latency**: ~50-150 ms (CPU), ~10-30 ms (GPU)
- **Batch throughput**: ~200-500 teks/detik (CPU), ~1000-2000 teks/detik (GPU)
- **RAM usage**: ~600 MB (model + tokenizer + Streamlit overhead)
- **Disk usage**: ~270 MB (model files)
