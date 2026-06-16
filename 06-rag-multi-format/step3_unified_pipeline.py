"""Step 3: 통합 파이프라인"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import os
import hashlib
from dotenv import load_dotenv

from step1_multi_loader import load_pdf, load_csv, load_text, save_uploaded_file
from step2_image_ocr import load_image_as_document

# 환경변수 로드
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def load_any_document(file_path: str) -> list:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".csv":
        return load_csv(file_path)
    elif ext in [".txt", ".md", ".markdown"]:
        return load_text(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        return [load_image_as_document(file_path)]
    else:
        try:
            return load_text(file_path)
        except Exception:
            return [Document(
                page_content=f"[지원하지 않는 형식: {ext}]",
                metadata={"filename": path.name, "file_type": ext, "error": "unsupported"}
            )]


def compute_file_hash(file_path: str) -> str:
    """파일 내용(바이트)으로 해시 생성 — 변경 감지용 지문"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


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
        where={"filename": filename},
        include=["metadatas"],
        limit=1
    )
    metadatas = found.get("metadatas")
    return metadatas[0].get("file_hash") if metadatas else None


def index_file(vector_store: Chroma, file_path: str):
    """파일 한 개를 신규/변경/중복으로 판별해 저장하고, (상태, 청크수)를 돌려준다.

    - "skipped" : 같은 파일·같은 내용 → 청킹/임베딩 없이 스킵
    - "updated" : 같은 파일·다른 내용 → 기존 청크 삭제 후 재저장
    - "new"     : 처음 보는 파일      → 그대로 저장
    """
    filename = Path(file_path).name
    new_hash = compute_file_hash(file_path)
    stored_hash = find_stored_hash(vector_store, filename)

    if stored_hash == new_hash:
        return "skipped", 0

    if stored_hash is None:
        status = "new"
    else:
        # 내용이 바뀜 → 그 파일의 기존 청크만 제거 후 재저장
        vector_store._collection.delete(where={"filename": filename})
        status = "updated"

    documents = load_any_document(file_path)
    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["file_hash"] = new_hash

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)
    return status, len(chunks)


def main():
    st.set_page_config("06-rag-multi-format | Step 3: 통합 파이프라인", layout="wide")
    st.header("Step 3: load_any_document() — 통합 파이프라인")

    st.divider()

    uploaded_files = st.file_uploader(
        "다양한 형식의 파일을 업로드하세요",
        type=["pdf", "csv", "txt", "md", "jpg", "jpeg", "png", "gif", "bmp", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            ext = Path(f.name).suffix.lower()
            st.write(f"- **{f.name}** ({ext}) — {f.size:,} bytes")

        if st.button("모든 파일 → 통합 벡터DB 저장", type="primary"):
            vector_store = get_vector_store()
            result = {"new": [], "updated": [], "skipped": []}
            total_chunks = 0

            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)

                try:
                    with st.spinner(f"{uploaded_file.name} 신규/변경/중복 판별 중..."):
                        status, n = index_file(vector_store, file_path)
                    result[status].append(uploaded_file.name)
                    total_chunks += n
                except Exception as e:
                    st.error(f"{uploaded_file.name} 처리 실패: {e}")

            if result["new"]:
                st.success(f"신규 저장 ({len(result['new'])}개): {', '.join(result['new'])}")
            if result["updated"]:
                st.warning(f"변경 감지 → 갱신 ({len(result['updated'])}개): {', '.join(result['updated'])}")
            if result["skipped"]:
                st.info(f"중복 → 스킵 ({len(result['skipped'])}개): {', '.join(result['skipped'])}")

            st.success(f"이번에 추가된 청크: {total_chunks}개 / 저장소 총 청크: {vector_store._collection.count()}개")
            st.balloons()

            st.subheader("저장된 청크 확인")
            collection = vector_store._collection
            stored = collection.get(include=["documents", "metadatas"], limit=5)

            for i, (doc, meta) in enumerate(zip(stored["documents"], stored["metadatas"])):
                with st.expander(f"청크 #{i + 1} — {meta.get('filename', '?')} ({meta.get('file_type', '?')})"):
                    st.text(doc[:300])
                    st.json(meta)



if __name__ == "__main__":
    main()
