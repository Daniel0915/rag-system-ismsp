"""Step 4: RAG + 평가자 + 재검색 루프"""

import json
import operator
import streamlit as st
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


TEST_DOCUMENTS = [
    Document(
        page_content="Apple Inc.는 1976년 스티브 잡스, 스티브 워즈니악, 로널드 웨인이 설립한 미국의 기술 회사입니다. "
                     "본사는 캘리포니아주 쿠퍼티노에 있습니다.",
        metadata={"source": "apple_overview.txt"},
    ),
    Document(
        page_content="Apple의 주요 제품으로는 iPhone, iPad, MacBook, Apple Watch, AirPods 등이 있습니다. "
                     "MacBook은 Apple이 만든 노트북 컴퓨터 브랜드로, MacBook Air와 MacBook Pro 라인업이 있습니다.",
        metadata={"source": "apple_products.txt"},
    ),
    Document(
        page_content="Apple의 2024 회계연도 매출액은 약 3,910억 달러(USD)입니다. "
                     "이 중 iPhone이 전체 매출의 약 52%를 차지하며, 서비스 부문이 약 22%를 차지합니다.",
        metadata={"source": "apple_financials.txt"},
    ),
    Document(
        page_content="삼성전자는 대한민국의 대표적인 전자 기업으로, 반도체, 스마트폰, 가전제품 등을 생산합니다. "
                     "2024년 매출액은 약 258조 원입니다.",
        metadata={"source": "samsung_overview.txt"},
    ),
    Document(
        page_content="구글(Google)은 알파벳(Alphabet Inc.)의 자회사로, 검색 엔진, 클라우드 서비스, "
                     "안드로이드 운영체제 등을 제공합니다. 2024년 알파벳의 매출액은 약 3,500억 달러입니다.",
        metadata={"source": "google_overview.txt"},
    ),
    Document(
        page_content="마이크로소프트(Microsoft)는 Windows, Office, Azure 클라우드 서비스를 제공하는 "
                     "미국의 기술 기업입니다. 2024 회계연도 매출액은 약 2,450억 달러입니다.",
        metadata={"source": "microsoft_overview.txt"},
    ),
    Document(
        page_content="노트북 컴퓨터 시장에서 Apple MacBook, 삼성 Galaxy Book, 레노버 ThinkPad, "
                     "Dell XPS 등이 주요 브랜드로 경쟁하고 있습니다.",
        metadata={"source": "laptop_market.txt"},
    ),
]


@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=TEST_DOCUMENTS,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


class State(TypedDict):
    question: str
    search_query: str
    answer: str
    is_complete: bool
    evaluation_reason: str
    iteration: int
    max_iterations: int
    all_context: Annotated[list[str], operator.add]


def rewrite_query(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano")
    iteration = state.get("iteration", 0) + 1

    if iteration == 1:
        prompt = (
            f"다음 사용자 질문을 벡터 검색에 적합한 검색 쿼리로 변환하세요.\n"
            f"핵심 키워드를 추출하고, 검색에 유리한 형태로 바꾸세요.\n"
            f"쿼리만 답하세요. 다른 설명은 하지 마세요.\n\n"
            f"질문: {state['question']}"
        )
    else:
        prompt = (
            f"이전 검색으로 충분한 답변을 얻지 못했습니다.\n\n"
            f"원래 질문: {state['question']}\n"
            f"이전 검색 쿼리: {state['search_query']}\n"
            f"불완전 이유: {state.get('evaluation_reason', '')}\n\n"
            f"부족한 정보를 찾을 수 있도록 새로운 검색 쿼리를 작성하세요.\n"
            f"쿼리만 답하세요. 다른 설명은 하지 마세요."
        )

    response = llm.invoke(prompt)
    return {"search_query": response.content.strip(), "iteration": iteration}


def retrieve(state: State) -> dict:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(state["search_query"], k=2)

    new_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        new_chunks.append(f"(출처: {source}) {doc.page_content}")

    return {"all_context": new_chunks}


def generate(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano")

    # 중복 제거 후 순서 유지 (재검색을 반복할 때 같은 문서가 여러 번 나올 수 있음)
    unique_context = list(dict.fromkeys(state["all_context"]))

    context_text = "\n\n".join(
        f"[문서 {i}] {chunk}" for i, chunk in enumerate(unique_context, 1)
    )

    response = llm.invoke(
        f"아래 참고 문서를 바탕으로 질문에 답변하세요.\n"
        f"참고 문서에 없는 내용은 '제공된 문서에서 확인할 수 없습니다'라고 답하세요.\n\n"
        f"참고 문서:\n{context_text}\n\n"
        f"질문: {state['question']}"
    )
    return {"answer": response.content}


def evaluate(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano")

    response = llm.invoke(
        f"당신은 답변 품질 평가자입니다.\n"
        f"사용자의 질문에 대해 답변이 충분하고 완전한지 평가하세요.\n\n"
        f"질문: {state['question']}\n"
        f"답변: {state['answer']}\n\n"
        f"평가 기준:\n"
        f"- 질문의 모든 부분에 답변했는가?\n"
        f"- 구체적인 수치나 정보가 포함되었는가?\n"
        f"- '확인할 수 없습니다'라는 표현이 있으면 불완전\n\n"
        f"반드시 아래 JSON 형식으로만 답하세요:\n"
        f'{{"is_complete": true 또는 false, "reason": "<평가 사유>"}}'
    )

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        is_complete = bool(parsed["is_complete"])
        reason = parsed["reason"]
    except (json.JSONDecodeError, KeyError, ValueError):
        is_complete = True
        reason = "평가 파싱 실패 — 완전한 것으로 간주합니다."

    return {"is_complete": is_complete, "evaluation_reason": reason}


def should_retry(state: State) -> str:
    if not state["is_complete"] and state["iteration"] < state["max_iterations"]:
        return "retry"
    return "done"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "rewrite_query",
            "done": END,
        }
    )

    return graph.compile()


def run_graph(question: str, max_iterations: int) -> dict:
    """그래프를 끝까지 실행하고 최종 State를 반환한다.

    app.invoke(...)는 START부터 END까지 그래프를 모두 돌린 뒤 '최종 State'를 돌려준다.
    (중간 과정을 단계별로 보려면 app.stream(...)을 쓰지만, 여기서는
     LangGraph 흐름과 최종 결과만 확인하면 되므로 invoke로 충분하다.)
    """
    app = build_graph()
    return app.invoke({
        "question": question,
        "iteration": 0,
        "max_iterations": max_iterations,
        "all_context": [],
    })


def main():
    st.set_page_config("07-langgraph-basics | Step 4: RAG + 평가자", layout="wide")
    st.header("Step 4: LangGraph — RAG + 평가자 + 재검색 루프")

    with st.expander("그래프 구조 보기"):
        app = build_graph()
        st.markdown(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")

    st.divider()

    with st.expander("테스트 데이터 (인메모리 벡터 저장소)"):
        for doc in TEST_DOCUMENTS:
            st.markdown(f"**{doc.metadata.get('source', 'unknown')}**")
            st.text(doc.page_content)
            st.markdown("---")

    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 맥북 만든 회사 매출액 알려줘",
    )
    max_iterations = st.slider("최대 검색 반복 횟수", min_value=1, max_value=5, value=3)

    if question and st.button("RAG 실행", type="primary"):
        with st.spinner("RAG + 평가 루프 실행 중..."):
            result = run_graph(question, max_iterations)

        st.subheader("최종 답변")
        st.success(result.get("answer", ""))

        st.divider()
        st.subheader("실행 정보")
        st.markdown(f"**총 검색 횟수:** {result.get('iteration', 0)}회 (재검색 루프)")
        st.markdown(f"**마지막 검색 쿼리:** {result.get('search_query', '')}")

        is_complete = result.get("is_complete")
        st.markdown(f"**평가 결과:** {'완전' if is_complete else '불완전'}")
        st.markdown(f"**평가 사유:** {result.get('evaluation_reason', '')}")

        # all_context는 Annotated[operator.add]로 누적되므로 중복 제거 후 표시
        unique_context = list(dict.fromkeys(result.get("all_context", [])))
        with st.expander(f"검색된 문서 ({len(unique_context)}건)"):
            for chunk in unique_context:
                st.text(chunk)



if __name__ == "__main__":
    main()
