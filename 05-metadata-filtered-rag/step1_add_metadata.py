"""Step 1: 메타데이터 추가"""

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


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = str(BASE_DIR / "PDF_임시폴더")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_with_metadata(pdf_path: str, metadata: dict) -> list:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source_file"] = filename
        doc.metadata.update(metadata)

    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def save_to_vector_store(chunks, append=False):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not append:
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
    else:
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"}
        )
        vector_store.add_documents(chunks)

    return vector_store


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

            with st.spinner("PDF 로딩 + 메타데이터 추가 중..."):
                documents = load_pdf_with_metadata(file_path, metadata)
            st.success(f"{len(documents)}페이지 로드 완료")

            with st.spinner("청크 분할 중..."):
                chunks = chunk_documents(documents)
            st.success(f"{len(chunks)}개 청크 생성")

            with st.spinner("ChromaDB에 저장 중..."):
                save_to_vector_store(chunks)
            st.success("저장 완료!")

            st.subheader("저장된 청크의 메타데이터 확인")
            for i, chunk in enumerate(chunks[:3]):
                with st.expander(f"청크 #{i + 1} 메타데이터"):
                    st.json(chunk.metadata)
                    st.text(chunk.page_content[:200] + "...")

            if len(chunks) > 3:
                st.caption(f"... 외 {len(chunks) - 3}개 청크")



if __name__ == "__main__":
    main()
