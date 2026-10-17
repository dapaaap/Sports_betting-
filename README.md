# Football Match Analysis

Tool analisis pertandingan sepak bola yang menyajikan estimasi probabilitas, konteks statistik, expected goals (xG), dan ringkasan faktor pertandingan.

> [!WARNING]
> **Tujuan proyek ini hanya untuk edukasi dan analisis olahraga. Proyek ini bukan untuk berjudi, tidak menyediakan rekomendasi taruhan, dan tidak menampilkan nominal atau saran terkait uang.**
>
> **This project is for sports analysis and educational purposes only. It is not intended for gambling, does not provide wagering recommendations, and does not display monetary amounts or money-related advice.**

Estimasi dan keluaran model dapat keliru atau tidak lengkap. Jangan menggunakannya sebagai dasar keputusan finansial atau aktivitas perjudian.

## Fitur

- Estimasi probabilitas hasil pertandingan dan pasar gol.
- Konteks statistik tim dan expected goals dari Understat/FBref bila tersedia.
- Ringkasan konteks pertandingan berbantuan AI (opsional).
- Dashboard terminal, tampilan detail pertandingan, dan ekspor hasil analisis ke CSV.
- Data contoh untuk mencoba dashboard tanpa konfigurasi layanan data.

## Instalasi

```powershell
py -m pip install -r requirements.txt
```

## Menjalankan

```powershell
# Jalankan dengan data contoh bila kunci layanan data belum dikonfigurasi
py main.py

# Gunakan data langsung setelah mengatur kunci API secara lokal
py main.py --live

# Lihat analisis satu tim/pertandingan
py main.py --detail "Arsenal"

# Perbarui dashboard otomatis atau ubah rentang analisis
py main.py --watch
py main.py --lookahead 24

# Tampilkan konfigurasi analisis
py main.py --config
```

Kunci layanan data atau AI dapat disimpan dalam environment lokal atau `api_keys.py`. Jangan memasukkan kunci rahasia ke GitHub. Data contoh tidak menggambarkan pertandingan atau hasil yang sebenarnya.

## Sumber Data dan AI

Data pertandingan dapat diambil melalui provider yang dikonfigurasi pada proyek. xG dan konteks historis hanya ditambahkan bila provider yang sesuai tersedia. Integrasi AI bersifat opsional; gunakan `--no-ai` untuk menonaktifkannya.

## Batasan

- Probabilitas adalah estimasi model, bukan kepastian atau jaminan hasil.
- Ketersediaan dan kualitas data eksternal dapat berbeda antar kompetisi.
- Analisis ini tidak dimaksudkan untuk mendukung perjudian atau keputusan finansial.
