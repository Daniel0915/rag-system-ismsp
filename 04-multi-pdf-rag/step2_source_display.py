"""Step 2: 출처 표시 — 답변에 source_file + 페이지 표시"""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def create_rag_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )

    llm = ChatOpenAI(model="gpt-5-nano")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 기반 질문응답 AI 어시스턴트입니다.
제공된 컨텍스트에서 답을 찾아주세요.
컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답변하세요.
한국어로 간결하고 명확하게 답변하세요."""),
        ("human", """다음 문서를 참고하여 질문에 답변해주세요:

[문서 내용]
{context}

---

질문: {input}""")
    ])

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

    return rag_chain


def format_sources(context_docs) -> list:
    sources = []
    seen = set()

    for doc in context_docs:
        source_file = doc.metadata.get("source_file", "알 수 없음")
        page = doc.metadata.get("page", 0) + 1
        key = f"{source_file}_{page}"

        if key not in seen:
            sources.append({
                "file": source_file,
                "page": page,
                "content": doc.page_content[:200]
            })
            seen.add(key)

    return sources


def main():
    st.set_page_config("04-multi-pdf-rag | Step 2: 출처 표시", layout="wide")
    st.header("Step 2: 답변에 출처 표시")

    if not os.path.exists(CHROMA_DIR):
        st.warning("벡터DB가 없습니다. step1에서 먼저 PDF를 저장하세요.")
        return

    if "rag_chain" not in st.session_state:
        with st.spinner("RAG 체인 초기화 중..."):
            st.session_state.rag_chain = create_rag_chain()

    question = st.text_input(
        "문서에 대해 질문하세요",
        placeholder="예) 환불 정책이 어떻게 되나요?"
    )

    if question:
        with st.spinner("답변 생성 중..."):
            response = st.session_state.rag_chain.invoke({"input": question})

        st.subheader("답변")
        st.write(response["answer"])

        context_docs = response.get("context", [])
        sources = format_sources(context_docs)

        st.divider()
        st.subheader("출처")
        for src in sources:
            st.markdown(f"**{src['file']}** — {src['page']}페이지")
            st.caption(src["content"] + "...")



if __name__ == "__main__":
    main()
