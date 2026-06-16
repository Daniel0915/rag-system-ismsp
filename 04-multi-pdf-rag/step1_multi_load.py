"""Step 1: 다중 PDF 로딩 — 여러 PDF를 하나의 벡터 저장소에 통합"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

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


def compute_file_hash(pdf_path: str) -> str:
    """파일 내용(바이트)으로 해시 생성 — 변경 감지용 지문"""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_single_pdf(pdf_path: str, file_hash: str) -> list:
    """PDF 한 개를 로드하고 메타데이터(파일명, 경로, 해시)를 부여"""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source_file"] = filename
        doc.metadata["file_path"] = pdf_path
        doc.metadata["file_hash"] = file_hash

    return documents


def get_vector_store() -> Chroma:
    """ChromaDB 벡터 저장소 열기 (없으면 새로 생성). delete 없이 재사용 → 누적 가능"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )


def find_stored_hash(vector_store: Chroma, filename: str):
    """저장소에 있는 해당 파일명의 해시를 반환. 없으면 None (= 처음 보는 파일)."""
    found = vector_store._collection.get(
        where={"source_file": filename},
        include=["metadatas"],
        limit=1
    )
    metadatas = found.get("metadatas")
    return metadatas[0].get("file_hash") if metadatas else None


def index_pdf(vector_store: Chroma, pdf_path: str) -> str:
    """PDF 한 개를 신규/변경/중복으로 판별해 저장하고, 처리 상태를 돌려준다.

    - "skipped" : 같은 파일·같은 내용 → 청킹/임베딩 없이 스킵
    - "updated" : 같은 파일·다른 내용 → 기존 청크 삭제 후 재저장
    - "new"     : 처음 보는 파일      → 그대로 저장
    """
    filename = os.path.basename(pdf_path)
    new_hash = compute_file_hash(pdf_path)
    stored_hash = find_stored_hash(vector_store, filename)

    if stored_hash == new_hash:
        return "skipped"

    if stored_hash is None:
        status = "new"
    else:
        # 내용이 바뀜 → 그 파일의 기존 청크만 제거 후 재저장
        vector_store._collection.delete(where={"source_file": filename})
        status = "updated"

    chunks = chunk_documents(load_single_pdf(pdf_path, new_hash))
    vector_store.add_documents(chunks)
    return status


def sync_pdfs_to_vector_store(pdf_paths: list):
    """여러 PDF를 파일명 + 해시 기준으로 저장. 파일마다 신규/변경/중복을 구분한다.

    반환: (vector_store, {"new": [...], "updated": [...], "skipped": [...]})
    """
    vector_store = get_vector_store()
    result = {"new": [], "updated": [], "skipped": []}

    for pdf_path in pdf_paths:
        status = index_pdf(vector_store, pdf_path)
        result[status].append(os.path.basename(pdf_path))

    return vector_store, result


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

            with st.spinner("2. 신규/변경/중복 판별 후 저장 중..."):
                vector_store, result = sync_pdfs_to_vector_store(pdf_paths)

            if result["new"]:
                st.success(f"신규 저장 ({len(result['new'])}개): {', '.join(result['new'])}")
            if result["updated"]:
                st.warning(f"변경 감지 → 갱신 ({len(result['updated'])}개): {', '.join(result['updated'])}")
            if result["skipped"]:
                st.info(f"중복 → 스킵 ({len(result['skipped'])}개): {', '.join(result['skipped'])}")

            st.success(f"현재 저장소 총 청크: {vector_store._collection.count()}개")
            st.balloons()



if __name__ == "__main__":
    main()
