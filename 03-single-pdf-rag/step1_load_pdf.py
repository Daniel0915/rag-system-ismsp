"""Step 1: 문서 로딩 — PDF를 Document 객체로 변환"""

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent


def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path


def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["file_path"] = pdf_path
        doc.metadata["filename"] = os.path.basename(pdf_path)

    return documents


def main():
    st.set_page_config("03-single-pdf-rag | Step 1: 문서 로딩", layout="wide")
    st.header("Step 1: PDF → Document 변환")

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if pdf_file:
        with st.spinner("PDF 파일을 저장하는 중..."):
            pdf_path = save_uploadedfile(pdf_file)
        st.success(f"파일 저장 완료: {pdf_file.name}")

        with st.spinner("PDF를 Document로 변환하는 중..."):
            documents = load_pdf(pdf_path)

        st.success(f"총 **{len(documents)}**개 페이지를 Document로 변환했습니다.")

        st.subheader("Document 내용 확인")

        for i, doc in enumerate(documents):
            with st.expander(f"페이지 {i + 1} (길이: {len(doc.page_content)}자)"):
                st.json(doc.metadata)
                st.text_area(
                    "page_content (처음 500자)",
                    value=doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                    height=200,
                    key=f"doc_{i}"
                )



if __name__ == "__main__":
    main()
