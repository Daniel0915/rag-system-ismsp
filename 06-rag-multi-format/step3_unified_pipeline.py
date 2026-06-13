"""Step 3: 통합 파이프라인"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import os
from dotenv import load_dotenv

from step1_multi_loader import load_pdf, load_csv, load_text, save_uploaded_file
from step2_image_ocr import load_image_as_document

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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


def chunk_and_store(documents: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        existing = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"}
        )
        # delete(where={})는 ValueError를 내고 아무것도 안 지움 → ID로 전체 삭제
        ids = existing._collection.get()["ids"]
        if ids:
            existing._collection.delete(ids=ids)
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store, len(chunks)


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
            all_documents = []

            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                ext = Path(uploaded_file.name).suffix.lower()

                try:
                    with st.spinner(f"{uploaded_file.name} 로딩 중..."):
                        docs = load_any_document(file_path)
                        all_documents.extend(docs)
                    st.success(f"{uploaded_file.name} → {len(docs)}개 Document ({ext} 로더)")
                except Exception as e:
                    st.error(f"{uploaded_file.name} 로딩 실패: {e}")

            with st.spinner("통합 벡터DB에 저장 중..."):
                vector_store, chunk_count = chunk_and_store(all_documents)

            st.success(f"총 {len(all_documents)}개 Document → {chunk_count}개 청크 → ChromaDB 저장 완료!")
            st.balloons()

            st.subheader("저장된 청크 확인")
            collection = vector_store._collection
            result = collection.get(include=["documents", "metadatas"], limit=5)

            for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
                with st.expander(f"청크 #{i + 1} — {meta.get('filename', '?')} ({meta.get('file_type', '?')})"):
                    st.text(doc[:300])
                    st.json(meta)



if __name__ == "__main__":
    main()
