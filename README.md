# PDF Chat - LangChain Q&A

Proyek ini adalah aplikasi **Streamlit** yang memungkinkan Anda mengunggah file PDF, kemudian bertanya apa saja terkait isi dokumen melalui **chat interface**. Jawaban akan diberikan oleh model OpenAI (misalnya GPT-3.5, GPT-4, dsb.) dengan memanfaatkan kemampuan **LangChain** untuk *retrieval-based Q&A*.

[tampilan app](tampilan_app.mov]

## Fitur
- Unggah PDF melalui antarmuka Streamlit
- Teks dari PDF akan di-*extract* dan di-*embed* (menggunakan OpenAIEmbeddings & FAISS)
- Aplikasi menyediakan tampilan **chat** interaktif:
  - User bertanya
  - AI menampilkan jawaban
- Jika Anda mengunggah **file PDF baru**, percakapan akan **di-reset** sehingga Anda bisa memulai diskusi baru dengan dokumen baru

## Persyaratan
- Python 3.7 atau lebih
- Paket Python berikut:
  - `streamlit`
  - `openai`
  - `PyPDF2`
  - `langchain`
  - `tiktoken` (untuk token handling)
- **OpenAI API Key** (diperlukan untuk memanggil layanan OpenAI)

## Cara Menjalankan
1. **Kloning / Download** proyek ini.
2. Masuk ke folder proyek, dan **buat** environment (opsional tapi disarankan):
   ```bash
   python3 -m venv env
   source env/bin/activate
3. Instal dependencies:
   ```bash
   pip install -r requirements.txt
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
5. Buka link di terminal (biasanya http://localhost:8501).

## Penggunaan
- Masukkan OpenAI API Key Anda di sidebar.
- Pilih model LLM (misalnya gpt-3.5-turbo).
- Upload PDF.
- Tanyakan apa saja di kolom chat input di bawah.
- Jika Anda mengunggah PDF baru, aplikasi reset percakapan dan memproses PDF baru.
