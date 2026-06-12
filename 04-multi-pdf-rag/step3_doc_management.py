"""Step 3: 문서 관리 — 추가/삭제 + 벡터 저장소 재활용"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )


def get_stored_files():
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])

        file_info = {}
        for meta in results["metadatas"]:
            fname = meta.get("source_file", "알 수 없음")
            if fname not in file_info:
                file_info[fname] = {"chunks": 0, "pages": set()}
            file_info[fname]["chunks"] += 1
            file_info[fname]["pages"].add(meta.get("page", 0))

        return file_info
    except Exception:
        return {}


def add_pdf_to_store(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source_file"] = filename
        doc.metadata["file_path"] = pdf_path

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(documents), len(chunks)


def delete_file_from_store(filename: str):
    vector_store = get_vector_store()
    collection = vector_store._collection

    results = collection.get(include=["metadatas"])
    ids_to_delete = []
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        if meta.get("source_file") == filename:
            ids_to_delete.append(doc_id)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)


def main():
    st.set_page_config("04-multi-pdf-rag | Step 3: 문서 관리", layout="wide")
    st.header("Step 3: 문서 추가/삭제 관리")

    if not os.path.exists(CHROMA_DIR):
        st.warning("벡터DB가 없습니다. step1에서 먼저 PDF를 저장하세요.")
        return

    if "flash_message" in st.session_state:
        st.success(st.session_state.pop("flash_message"))

    st.subheader("저장된 문서 목록")

    file_info = get_stored_files()

    if not file_info:
        st.info("저장된 문서가 없습니다.")
    else:
        total_chunks = sum(info["chunks"] for info in file_info.values())
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("총 파일 수", len(file_info))
        col_m2.metric("총 청크 수", total_chunks)

        st.divider()

        for fname, info in sorted(file_info.items()):
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{fname}**")
            with col2:
                st.write(f"{info['chunks']}개 청크 / {len(info['pages'])}페이지")
            with col3:
                if st.button("삭제", key=f"del_{fname}"):
                    deleted = delete_file_from_store(fname)
                    st.session_state["flash_message"] = f"{fname}에서 {deleted}개 청크 삭제 완료"
                    st.rerun()

    st.divider()

    st.subheader("문서 추가")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    new_files = st.file_uploader(
        "추가할 PDF 파일",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"add_files_{st.session_state['uploader_key']}"
    )

    if new_files and st.button("문서 추가", type="primary"):
        added_messages = []
        for pdf_file in new_files:
            temp_dir = str(BASE_DIR / "PDF_임시폴더")
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            with st.spinner(f"{pdf_file.name} 추가 중..."):
                pages, chunks = add_pdf_to_store(file_path)
            added_messages.append(f"{pdf_file.name}: {pages}페이지 → {chunks}청크")

        st.session_state["flash_message"] = "추가 완료! " + " / ".join(added_messages)
        st.session_state["uploader_key"] += 1
        st.rerun()



if __name__ == "__main__":
    main()
