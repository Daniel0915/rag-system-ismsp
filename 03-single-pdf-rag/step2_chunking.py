"""Step 2: 텍스트 분할 — RecursiveCharacterTextSplitter"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from step1_load_pdf import save_uploadedfile, load_pdf

# 환경변수 로드
load_dotenv()


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def main():
    st.set_page_config("03-single-pdf-rag | Step 2: 텍스트 분할", layout="wide")
    st.header("Step 2: 텍스트 분할 (Chunking)")

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if pdf_file:
        pdf_path = save_uploadedfile(pdf_file)
        documents = load_pdf(pdf_path)

        st.success(f"PDF 로드 완료: {len(documents)}개 페이지")

        st.subheader("청킹 파라미터 조절")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("chunk_size (청크 크기)", 200, 2000, 1000, step=100)
        with col2:
            chunk_overlap = st.slider("chunk_overlap (겹침)", 0, 500, 200, step=50)

        chunks = chunk_documents(documents, chunk_size, chunk_overlap)

        st.subheader("분할 결과")

        col_before, col_after = st.columns(2)
        with col_before:
            st.metric("분할 전 (페이지)", len(documents))
            total_chars = sum(len(d.page_content) for d in documents)
            st.metric("총 글자 수", f"{total_chars:,}")

        with col_after:
            st.metric("분할 후 (청크)", len(chunks))
            avg_chars = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
            st.metric("평균 청크 크기", f"{avg_chars:,}자")

        st.subheader("청크 내용 확인")
        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get("page", 0) + 1
            with st.expander(f"청크 #{i + 1} (페이지 {page}, {len(chunk.page_content)}자)"):
                st.text_area(
                    "내용",
                    value=chunk.page_content,
                    height=150,
                    key=f"chunk_{i}"
                )



if __name__ == "__main__":
    main()
