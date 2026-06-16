"""Step 2: 필터 검색"""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os

from step1_add_metadata import CHROMA_DIR


def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )


def search_with_filter(query: str, filter_dict: dict = None, k: int = 4):
    vector_store = get_vector_store()

    if filter_dict:
        results = vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
    else:
        results = vector_store.similarity_search(query, k=k)

    return results


def create_filtered_rag_chain(filter_dict: dict = None):
    vector_store = get_vector_store()
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

    search_kwargs = {"k": 4}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


def get_available_metadata():
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])

        categories = set()
        departments = set()
        for meta in results["metadatas"]:
            if "category" in meta:
                categories.add(meta["category"])
            if "department" in meta:
                departments.add(meta["department"])

        return sorted(categories), sorted(departments)
    except Exception:
        return [], []


def main():
    st.set_page_config("05-metadata-filtered-rag | Step 2: 필터 검색", layout="wide")
    st.header("Step 2: 메타데이터 필터 검색")

    st.divider()

    if not os.path.exists(CHROMA_DIR):
        st.warning("벡터DB가 없습니다. step1에서 먼저 PDF를 저장하세요.")
        return

    categories, departments = get_available_metadata()

    if not categories and not departments:
        st.warning("메타데이터가 없는 문서입니다. step1에서 메타데이터와 함께 저장하세요.")
        return

    st.subheader("검색 필터 설정")
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("카테고리", ["전체"] + list(categories))
    with col2:
        selected_department = st.selectbox("부서", ["전체"] + list(departments))

    filter_dict = {}
    if selected_category != "전체":
        filter_dict["category"] = selected_category
    if selected_department != "전체":
        filter_dict["department"] = selected_department

    if len(filter_dict) >= 2:
        filter_dict = {"$and": [{k: v} for k, v in filter_dict.items()]}

    if filter_dict:
        st.info(f"적용 필터: `{filter_dict}`")
    else:
        filter_dict = None
        st.info("필터 없음 (전체 검색)")

    st.divider()

    tab1, tab2 = st.tabs(["검색 테스트", "RAG 질문"])

    with tab1:
        st.subheader("필터 검색 테스트")
        query = st.text_input("검색어", placeholder="예) 환불 정책")

        if query:
            with st.spinner("검색 중..."):
                results = search_with_filter(query, filter_dict)

            st.success(f"**{len(results)}**개 결과")
            for idx, doc in enumerate(results, 1):
                source = doc.metadata.get("source_file", "알 수 없음")
                page = doc.metadata.get("page", 0) + 1
                cat = doc.metadata.get("category", "-")
                dept = doc.metadata.get("department", "-")

                with st.expander(f"결과 {idx}: {source} p.{page} [{cat}/{dept}]"):
                    st.write(doc.page_content[:300])
                    st.json(doc.metadata)

    with tab2:
        st.subheader("필터 적용 RAG 질문")
        question = st.text_input("질문", placeholder="예) 환불 절차가 어떻게 되나요?", key="rag_q")

        if question:
            with st.spinner("RAG 체인 실행 중..."):
                rag_chain = create_filtered_rag_chain(filter_dict)
                response = rag_chain.invoke({"input": question})

            st.write(response["answer"])

            st.divider()
            st.subheader("참조 문서")
            for doc in response.get("context", []):
                source = doc.metadata.get("source_file", "")
                cat = doc.metadata.get("category", "")
                st.caption(f"{source} [{cat}] — p.{doc.metadata.get('page', 0) + 1}")



if __name__ == "__main__":
    main()
