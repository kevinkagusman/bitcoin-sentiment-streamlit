# Deploy ke Streamlit Cloud (Optional — untuk URL publik)

Ini panduan singkat **kalau kamu mau aplikasinya bisa diakses publik via URL** untuk demo video yang lebih profesional. Kalau tidak butuh URL publik dan cukup demo lokal, **skip langkah ini**.

## Kenapa pakai HuggingFace untuk model?

Model DistilBERT setelah fine-tune ukurannya ~270MB. GitHub batas file 100MB, jadi kita simpan model di HuggingFace Hub (gratis, unlimited untuk public repo) dan Streamlit Cloud akan download model dari sana saat deploy.

## Langkah 1: Upload model ke HuggingFace Hub

```bash
# 1. Install
pip install huggingface_hub

# 2. Login (akan minta token; buat di https://huggingface.co/settings/tokens)
huggingface-cli login

# 3. Upload folder model
huggingface-cli upload <username>/bitcoin-sentiment-distilbert ./model
```

Ganti `<username>` dengan username HuggingFace kamu. Misal: `kevinkagusman/bitcoin-sentiment-distilbert`.

## Langkah 2: Modifikasi app.py untuk load dari HuggingFace

Edit baris di `app.py`:

```python
# DARI:
MODEL_PATH = "./model"

# JADI:
MODEL_PATH = "kevinkagusman/bitcoin-sentiment-distilbert"  # ganti dengan repo HF kamu
```

`from_pretrained()` HuggingFace pintar — kalau path adalah folder lokal, dia load dari disk; kalau berbentuk `username/repo`, dia download dari HuggingFace Hub.

## Langkah 3: Push ke GitHub

```bash
cd streamlit_app/
git init
git add app.py requirements.txt README.md .streamlit/ sample_tweets.csv
git commit -m "Bitcoin Sentiment Analyzer Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/btc-sentiment-app.git
git push -u origin main
```

⚠️ **JANGAN push folder `model/`** — sudah ada di HuggingFace Hub.

## Langkah 4: Deploy di Streamlit Cloud

1. Buka https://streamlit.io/cloud
2. Sign in dengan akun GitHub
3. Klik **"New app"**
4. Pilih repo `btc-sentiment-app`, branch `main`, file `app.py`
5. Klik **"Deploy"**
6. Tunggu 3-5 menit (saat first deploy, dia download dependencies dari `requirements.txt` dan model dari HuggingFace)

Kamu akan dapat URL seperti:
```
https://btc-sentiment-app-xyz.streamlit.app
```

## Sertakan URL ini di Demo Video

Saat rekam demo video, intro dengan:

> "Aplikasi ini sudah di-deploy publik di Streamlit Community Cloud. Anyone bisa akses di [URL]. Saya akan demonstrasikan fungsi-fungsinya..."

Ini **sangat menambah profesionalisme** dan poin di rubric "Demo Video → presentation (10 pts)".

## Catatan Resource Limits (Streamlit Cloud Free Tier)

- **RAM**: 1 GB → cukup untuk DistilBERT (~600MB)
- **CPU only** (no GPU) → inference ~150ms/request, masih sangat acceptable
- **Sleep after 7 days idle** → klik "wake up" kalau perlu
- **Public app** → siapa saja bisa akses URL kamu

Untuk demo final project ini, free tier sudah lebih dari cukup.
