"""Step 4: 벡터 저장소 — ChromaDB에 저장하고 검색하기"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv

from step1_load_pdf import save_uploadedfile, load_pdf
from step2_chunking import chunk_documents

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def save_to_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        existing_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"}
        )
        existing_store._collection.delete(where={})
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store


def search_similar(query: str, k: int = 3):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )
    results = vector_store.similarity_search(query, k=k)
    return results


def main():
    st.set_page_config("03-single-pdf-rag | Step 4: 벡터 저장소", layout="wide")
    st.header("Step 4: ChromaDB — 저장과 검색")

    tab1, tab2 = st.tabs(["저장", "검색"])

    with tab1:
        st.subheader("PDF → 청크 → ChromaDB 저장")
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

        if pdf_file:
            if st.button("벡터DB에 저장", type="primary"):
                with st.spinner("1. PDF 로딩 중..."):
                    pdf_path = save_uploadedfile(pdf_file)
                    documents = load_pdf(pdf_path)
                st.success(f"1. PDF 로드 완료: {len(documents)}페이지")

                with st.spinner("2. 청크 분할 중..."):
                    chunks = chunk_documents(documents)
                st.success(f"2. 청크 분할 완료: {len(chunks)}개")

                with st.spinner("3. 임베딩 생성 + ChromaDB 저장 중..."):
                    vector_store = save_to_vector_store(chunks)
                st.success(f"3. ChromaDB 저장 완료!")

                st.balloons()

    with tab2:
        st.subheader("ChromaDB에서 유사 문서 검색")

        query = st.text_input(
            "검색할 질문을 입력하세요",
            placeholder="예) 문서에서 핵심 내용을 알려줘"
        )
        k = st.slider("검색 결과 개수 (k)", 1, 10, 3)

        if query:
            try:
                with st.spinner("유사 문서 검색 중..."):
                    results = search_similar(query, k=k)

                st.success(f"**{len(results)}**개의 관련 문서를 찾았습니다.")

                for idx, doc in enumerate(results, 1):
                    page = doc.metadata.get("page", 0) + 1
                    filename = doc.metadata.get("filename", "알 수 없음")

                    with st.expander(f"관련 문서 {idx} — {filename} {page}페이지"):
                        st.write(doc.page_content)
                        st.caption(f"출처: {filename} | 페이지: {page}")

            except Exception as e:
                st.error(f"벡터DB를 찾을 수 없습니다. '저장' 탭에서 먼저 PDF를 저장하세요.\n오류: {e}")



if __name__ == "__main__":
    main()
