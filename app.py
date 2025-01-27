import streamlit as st
import openai
from PyPDF2 import PdfReader

# Komponen LangChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ----- FUNGSI UNTUK EKSTRAKSI & CHUNKING PDF -----

def extract_text_from_pdf(pdf_file):
    """
    Membaca teks dari setiap halaman file PDF dan menggabungkannya menjadi satu string.
    """
    reader = PdfReader(pdf_file)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    return all_text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """
    Membagi teks menjadi beberapa potongan (chunks) agar lebih mudah diproses oleh LLM.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ----- FUNGSI UNTUK MENAMPILKAN CHAT HISTORY -----

def display_chat_history():
    """
    Menampilkan riwayat pesan di st.session_state["messages"].
    """
    messages = st.session_state["messages"]
    for msg in messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])

# ----- FUNGSI UTAMA STREAMLIT -----

def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📄")
    st.title("PDF Chat - Q&A dengan LangChain")

    st.markdown("""
    **Fitur**:
    - Unggah file PDF
    - Bertanya secara interaktif menggunakan *chat interface*
    - Setiap kali file baru diunggah, percakapan di-*reset*
    """)

    # 1) Input OpenAI API Key di sidebar
    st.sidebar.header("OpenAI API Key")
    openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Harap masukkan API Key terlebih dahulu.")
        st.stop()

    # 2) Pilihan model
    st.sidebar.header("Pilih Model")
    model_list = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox("Model:", model_list, index=2)

    # 3) Upload PDF
    uploaded_pdf = st.file_uploader("Upload PDF di sini:", type=["pdf"])
    
    # -- Inisialisasi session_state (percakapan & chain) --
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None

    # -- Jika user mengunggah PDF baru, reset percakapan & chain --
    if uploaded_pdf is not None:
        # Cek apakah file baru (berbeda nama dari sebelumnya)
        if st.session_state["uploaded_file_name"] != uploaded_pdf.name:
            # Reset seluruh percakapan dan chain
            st.session_state["messages"] = []
            st.session_state["qa_chain"] = None
            st.session_state["uploaded_file_name"] = uploaded_pdf.name

        # Jika chain belum dibuat (entah pertama kali atau setelah reset)
        if st.session_state["qa_chain"] is None:
            with st.spinner("Memproses PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                chunks = split_text_into_chunks(pdf_text, chunk_size=1000, chunk_overlap=100)

                # Buat embeddings & vector store
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                knowledge_base = FAISS.from_texts(chunks, embeddings)

                # Buat ChatOpenAI
                llm = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name=selected_model,
                    temperature=0
                )

                # Buat RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=knowledge_base.as_retriever()
                )
                st.session_state["qa_chain"] = qa_chain

                # Tambahkan salam awal
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "PDF berhasil diproses. Silakan ajukan pertanyaan."
                })

    # Tampilkan chat history
    display_chat_history()

    # Jika chain sudah siap, sediakan input chat
    if st.session_state["qa_chain"] is not None:
        user_input = st.chat_input("Ketik pertanyaan Anda...")
        if user_input:
            # 1) Tampilkan pertanyaan user
            st.session_state["messages"].append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.write(user_input)

            # 2) Proses jawaban
            with st.spinner("Sedang berpikir..."):
                try:
                    response = st.session_state["qa_chain"].run(user_input)
                    # 3) Simpan & Tampilkan jawaban asisten
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response
                    })
                    with st.chat_message("assistant"):
                        st.write(response)

                except Exception as e:
                    err_msg = f"Terjadi error saat memproses pertanyaan: {e}"
                    st.session_state["messages"].append({"role": "assistant", "content": err_msg})
                    st.chat_message("assistant").write(err_msg)
                    st.error(err_msg)
    else:
        st.info("Silakan unggah PDF untuk memulai percakapan.")

if __name__ == "__main__":
    main()
