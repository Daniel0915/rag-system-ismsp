"""Step 1: 메타데이터 추가"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def compute_content_hash(pdf_path: str, metadata: dict) -> str:
    """파일 내용 + 메타데이터로 해시 생성 — 내용이나 메타데이터가 바뀌면 갱신 대상"""
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    meta_bytes = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(file_bytes + meta_bytes).hexdigest()


def load_pdf_with_metadata(pdf_path: str, metadata: dict, content_hash: str) -> list:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata.update(metadata)
        doc.metadata["source_file"] = filename
        doc.metadata["file_hash"] = content_hash

    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


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


def index_pdf(vector_store: Chroma, pdf_path: str, metadata: dict):
    """PDF 한 개를 신규/변경/중복으로 판별해 저장하고, (상태, 청크리스트)를 돌려준다.

    - "skipped" : 같은 파일·같은 내용·같은 메타데이터 → 청킹/임베딩 없이 스킵
    - "updated" : 같은 파일·내용 또는 메타데이터가 다름  → 기존 청크 삭제 후 재저장
    - "new"     : 처음 보는 파일                         → 그대로 저장
    """
    filename = os.path.basename(pdf_path)
    new_hash = compute_content_hash(pdf_path, metadata)
    stored_hash = find_stored_hash(vector_store, filename)

    if stored_hash == new_hash:
        return "skipped", []

    if stored_hash is None:
        status = "new"
    else:
        # 내용/메타데이터가 바뀜 → 그 파일의 기존 청크만 제거 후 재저장
        vector_store._collection.delete(where={"source_file": filename})
        status = "updated"

    chunks = chunk_documents(load_pdf_with_metadata(pdf_path, metadata, new_hash))
    vector_store.add_documents(chunks)
    return status, chunks


def main():
    st.set_page_config("05-metadata-filtered-rag | Step 1: 메타데이터 추가", layout="wide")
    st.header("Step 1: 청크에 메타데이터 추가하기")

    st.divider()

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if pdf_file:
        st.subheader("메타데이터 설정")
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "카테고리",
                ["계약서", "매뉴얼", "보고서", "정책", "기타"]
            )
        with col2:
            department = st.selectbox(
                "부서",
                ["영업", "인사", "개발", "경영", "마케팅", "고객지원"]
            )

        metadata = {
            "category": category,
            "department": department
        }

        st.info(f"설정된 메타데이터: `{metadata}`")

        if st.button("메타데이터와 함께 저장", type="primary"):
            file_path = save_uploaded_file(pdf_file)

            with st.spinner("신규/변경/중복 판별 후 저장 중..."):
                vector_store = get_vector_store()
                status, chunks = index_pdf(vector_store, file_path, metadata)

            if status == "new":
                st.success(f"신규 저장: {len(chunks)}개 청크 생성")
            elif status == "updated":
                st.warning(f"변경 감지 → 갱신: {len(chunks)}개 청크로 교체")
            else:
                st.info("중복 → 스킵 (이미 동일한 파일·메타데이터)")

            if chunks:
                st.subheader("저장된 청크의 메타데이터 확인")
                for i, chunk in enumerate(chunks[:3]):
                    with st.expander(f"청크 #{i + 1} 메타데이터"):
                        st.json(chunk.metadata)
                        st.text(chunk.page_content[:200] + "...")

                if len(chunks) > 3:
                    st.caption(f"... 외 {len(chunks) - 3}개 청크")



if __name__ == "__main__":
    main()
