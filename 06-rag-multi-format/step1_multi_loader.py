"""Step 1: 포맷별 로더"""

import streamlit as st
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
)
from pathlib import Path
import os
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent


def load_pdf(file_path: str) -> list:
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "pdf"
    return docs


def load_csv(file_path: str) -> list:
    loader = CSVLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "csv"
    return docs


def load_text(file_path: str) -> list:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["file_type"] = Path(file_path).suffix.lower()
    return docs


def save_uploaded_file(uploaded_file) -> str:
    upload_dir = str(BASE_DIR / "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    st.set_page_config("06-rag-multi-format | Step 1: 포맷별 로더", layout="wide")
    st.header("Step 1: 포맷별 LangChain 로더")

    st.divider()

    uploaded_file = st.file_uploader(
        "파일을 업로드하세요",
        type=["pdf", "csv", "txt", "md"],
        help="PDF, CSV, TXT, MD 파일을 지원합니다"
    )

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        ext = Path(uploaded_file.name).suffix.lower()

        st.info(f"파일: **{uploaded_file.name}** | 확장자: `{ext}` | 크기: {uploaded_file.size:,} bytes")

        with st.spinner(f"{ext} 로더로 문서 변환 중..."):
            if ext == ".pdf":
                documents = load_pdf(file_path)
                st.success(f"**PyMuPDFLoader** 사용 → {len(documents)}개 Document (페이지별)")
            elif ext == ".csv":
                documents = load_csv(file_path)
                st.success(f"**CSVLoader** 사용 → {len(documents)}개 Document (행별)")
            elif ext in [".txt", ".md"]:
                documents = load_text(file_path)
                st.success(f"**TextLoader** 사용 → {len(documents)}개 Document (파일 전체)")
            else:
                st.error(f"지원하지 않는 형식: {ext}")
                return

        st.subheader("Document 내용 확인")
        for i, doc in enumerate(documents[:10]):
            with st.expander(f"Document {i + 1} (길이: {len(doc.page_content)}자)"):
                st.json(doc.metadata)
                content_preview = doc.page_content[:500]
                if len(doc.page_content) > 500:
                    content_preview += "..."
                st.text_area("page_content", value=content_preview, height=150, key=f"doc_{i}")

        if len(documents) > 10:
            st.caption(f"... 외 {len(documents) - 10}개 Document")



if __name__ == "__main__":
    main()
