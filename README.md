# Indo TTS Runner

Skrip mandiri sederhana untuk mengubah teks bahasa Indonesia menjadi suara (TTS), menggunakan model hasil fine-tune komunitas [alkhrzmy/qwen3-tts-0.6b-indonesian](https://huggingface.co/alkhrzmy/qwen3-tts-0.6b-indonesian) (fine-tune dari Qwen3-TTS-0.6B khusus bahasa Indonesia).

**Beberapa catatan penting:**

- Model ini hanya punya **satu suara tetap** (`indonesian_speaker`) - tidak bisa memilih suara lain, mendesain suara dari deskripsi teks, atau meniru (clone) suara orang tertentu.
- Ini adalah **model fine-tune dari komunitas**, bukan rilis resmi dari Qwen (Qwen3-TTS resmi saat ini belum mendukung bahasa Indonesia). Konfigurasi trainingnya cukup sederhana (4 epoch), jadi kualitasnya tidak dijamin resmi - sebaiknya coba dengarkan beberapa contoh dulu untuk memastikan hasilnya cukup baik untuk kebutuhanmu.

**Folder ini tidak berisi file model** - bobot model akan otomatis diunduh dari Hugging Face saat pertama kali dijalankan (disimpan di cache `~/.cache/huggingface`), jadi tidak perlu diunduh manual.

## Instalasi

1. Buat environment Python (contoh pakai conda):

   ```bash
   conda create -n qwen3-tts python=3.12
   conda activate qwen3-tts
   ```

2. Instal PyTorch. Kalau punya GPU NVIDIA, buka [halaman instalasi PyTorch](https://pytorch.org/get-started/locally/) untuk mencari versi CUDA yang sesuai, contoh:

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   Kalau tidak punya GPU, instal versi CPU saja (model 0.6B ini kecil, tetap bisa jalan di CPU meski lebih lambat dibanding GPU):

   ```bash
   pip install torch torchaudio
   ```

3. Instal paket Qwen-TTS:

   ```bash
   pip install qwen-tts soundfile
   ```

## Cara Pakai

Masukkan teks bahasa Indonesia yang ingin diucapkan ke file `text.txt` (di folder yang sama dengan skrip ini), lalu jalankan:

```bash
python run_indo_tts.py
```

Atau di Windows, klik dua kali `run_indo_tts.bat` (kalau conda tidak terinstal di lokasi default, buka dulu file ini dan ubah bagian `CONDA_BASE`).

Bisa juga langsung menentukan teks lewat command line:

```bash
python run_indo_tts.py --text "Halo, apa kabar?"

# Menggunakan file lain sebagai input
python run_indo_tts.py --text_file naskah.txt

# Memberi jeda 0.4 detik setelah setiap tanda titik "."
python run_indo_tts.py --period_pause 0.4
```

File wav hasil output akan tersimpan di folder `indo_tts_output/`.
