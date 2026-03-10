"""Step 1: 다중 PDF 로딩 — 여러 PDF를 하나의 벡터 저장소에 통합"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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


def save_uploaded_files(uploaded_files) -> list:
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
    os.makedirs(temp_dir, exist_ok=True)

    pdf_paths = []
    for pdf_file in uploaded_files:
        file_path = os.path.join(temp_dir, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        pdf_paths.append(file_path)
    return pdf_paths


def load_multiple_pdfs(pdf_paths: list) -> list:
    all_documents = []

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        filename = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata["source_file"] = filename
            doc.metadata["file_path"] = pdf_path

        all_documents.extend(documents)

    return all_documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def save_to_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        existing = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents"
        )
        existing._collection.delete(where={})
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="documents"
    )
    return vector_store


def main():
    st.set_page_config("04-multi-pdf-rag | Step 1: 다중 PDF 로딩", layout="wide")
    st.header("Step 1: 여러 PDF → 하나의 벡터 저장소")

    pdf_files = st.file_uploader(
        "PDF 파일들을 업로드하세요 (여러 개 가능)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if pdf_files:
        st.info(f"**{len(pdf_files)}**개 파일 선택됨: {', '.join(f.name for f in pdf_files)}")

        if st.button("모든 PDF를 벡터DB에 저장", type="primary"):
            with st.spinner("1. 파일 저장 중..."):
                pdf_paths = save_uploaded_files(pdf_files)
            st.success(f"1. {len(pdf_paths)}개 파일 저장 완료")

            with st.spinner("2. PDF 로딩 중..."):
                all_documents = load_multiple_pdfs(pdf_paths)
            st.success(f"2. 총 {len(all_documents)}개 페이지 로드 완료")

            file_counts = {}
            for doc in all_documents:
                fname = doc.metadata.get("source_file", "알 수 없음")
                file_counts[fname] = file_counts.get(fname, 0) + 1
            for fname, count in file_counts.items():
                st.write(f"  - {fname}: {count}페이지")

            with st.spinner("3. 청크 분할 중..."):
                chunks = chunk_documents(all_documents)
            st.success(f"3. {len(chunks)}개 청크 생성")

            with st.spinner("4. ChromaDB에 저장 중..."):
                vector_store = save_to_vector_store(chunks)
            st.success("4. 벡터DB 저장 완료!")

            st.balloons()



if __name__ == "__main__":
    main()
