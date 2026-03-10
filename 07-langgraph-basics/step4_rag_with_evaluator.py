"""Step 4: RAG + 평가자 + 재검색 루프"""

import json
import operator
import streamlit as st
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


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
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
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
    docs = vectorstore.similarity_search(state["search_query"], k=3)

    new_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        new_chunks.append(f"(출처: {source}) {doc.page_content}")

    return {"all_context": new_chunks}


def generate(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)

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
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

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


def run_graph_with_history(question: str, max_iterations: int) -> list[dict]:
    app = build_graph()
    iterations = []
    current = {}
    total_context = []

    for event in app.stream(
        {"question": question, "iteration": 0, "max_iterations": max_iterations,
         "all_context": []},
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            if node_name == "rewrite_query":
                if current:
                    iterations.append(current)
                current = {
                    "iteration": updates.get("iteration", len(iterations) + 1),
                    "search_query": updates.get("search_query", ""),
                }
            elif node_name == "retrieve":
                current["new_chunks"] = updates.get("all_context", [])
                total_context.extend(updates.get("all_context", []))
                current["total_context_count"] = len(set(total_context))
            elif node_name == "generate":
                current["answer"] = updates.get("answer", "")
            elif node_name == "evaluate":
                current["is_complete"] = updates.get("is_complete")
                current["evaluation_reason"] = updates.get("evaluation_reason", "")

    if current:
        iterations.append(current)

    return iterations


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
            history = run_graph_with_history(question, max_iterations)

        if not history:
            st.warning("결과가 없습니다.")
            return

        st.subheader("검색 이력")
        for item in history:
            iteration = item.get("iteration", "?")
            is_complete = item.get("is_complete")
            reason = item.get("evaluation_reason", "")

            label = f"반복 {iteration}: 검색 쿼리 = \"{item.get('search_query', '')}\""
            if is_complete is not None:
                label += " 완전" if is_complete else " 재검색 필요"

            with st.expander(label, expanded=(item == history[-1])):
                st.markdown(f"**검색 쿼리:** {item.get('search_query', '')}")

                new_chunks = item.get("new_chunks", [])
                st.markdown(f"**이번 검색 결과:** {len(new_chunks)}건")
                for chunk in new_chunks:
                    st.text(chunk)

                total_count = item.get("total_context_count", 0)
                st.markdown(f"**누적 고유 문서:** {total_count}건 (Annotated로 자동 누적)")

                if item.get("answer"):
                    st.markdown("**생성된 답변:**")
                    st.write(item["answer"])

                if is_complete is not None:
                    status = "완전" if is_complete else "불완전"
                    st.markdown(f"**평가:** {status}")
                    st.markdown(f"**사유:** {reason}")

        st.divider()
        st.subheader("최종 답변")
        last = history[-1]
        st.success(last.get("answer", ""))

        total = len(history)
        st.caption(f"총 {total}회 검색 수행")



if __name__ == "__main__":
    main()
