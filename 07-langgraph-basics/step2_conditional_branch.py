"""Step 2: LangGraph 조건 분기"""

import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


class State(TypedDict):
    question: str
    question_type: str
    answer: str


def classify_question(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    response = llm.invoke(
        "다음 질문의 유형을 분류하세요.\n"
        "- factual: 사실이나 정보를 묻는 질문 (예: '파이썬은 누가 만들었나요?')\n"
        "- opinion: 의견이나 판단을 묻는 질문 (예: '파이썬이 좋은 언어인가요?')\n"
        "- creative: 창의적 작성을 요구하는 질문 (예: '파이썬으로 시를 써주세요')\n\n"
        "반드시 factual, opinion, creative 중 하나만 답하세요. 다른 말은 하지 마세요.\n\n"
        f"질문: {state['question']}"
    )
    question_type = response.content.strip().lower()
    if question_type not in ("factual", "opinion", "creative"):
        question_type = "factual"
    return {"question_type": question_type}


def route_question(state: State) -> str:
    question_type = state["question_type"]
    question = state["question"]

    if question_type == "factual" and "비교" in question:
        return "opinion"

    return question_type


def answer_factual(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    response = llm.invoke(
        f"다음은 사실 정보를 묻는 질문입니다. 정확하고 객관적으로 답변하세요.\n\n"
        f"질문: {state['question']}"
    )
    return {"answer": response.content}


def answer_opinion(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
    response = llm.invoke(
        f"다음은 의견을 묻는 질문입니다. 여러 관점을 균형 있게 제시하며 답변하세요.\n\n"
        f"질문: {state['question']}"
    )
    return {"answer": response.content}


def answer_creative(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=1.0)
    response = llm.invoke(
        f"다음은 창의적인 작성을 요구하는 질문입니다. 자유롭고 창의적으로 답변하세요.\n\n"
        f"질문: {state['question']}"
    )
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(State)

    graph.add_node("classify", classify_question)
    graph.add_node("factual_answer", answer_factual)
    graph.add_node("opinion_answer", answer_opinion)
    graph.add_node("creative_answer", answer_creative)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_question,
        {
            "factual": "factual_answer",
            "opinion": "opinion_answer",
            "creative": "creative_answer",
        }
    )

    graph.add_edge("factual_answer", END)
    graph.add_edge("opinion_answer", END)
    graph.add_edge("creative_answer", END)

    return graph.compile()


def run_graph(question: str) -> dict:
    app = build_graph()
    result = app.invoke({"question": question})

    llm_type = result["question_type"]
    final_type = route_question(result)
    result["_routed_type"] = final_type
    result["_was_rerouted"] = (llm_type != final_type)
    return result


TYPE_LABELS = {
    "factual": "사실형 (Factual)",
    "opinion": "의견형 (Opinion)",
    "creative": "창의형 (Creative)",
}

TYPE_DESCRIPTIONS = {
    "factual": "정확하고 객관적인 정보를 제공합니다. (temperature=0)",
    "opinion": "여러 관점을 균형 있게 제시합니다. (temperature=0.7)",
    "creative": "자유롭고 창의적으로 응답합니다. (temperature=1.0)",
}


def main():
    st.set_page_config("07-langgraph-basics | Step 2: 조건 분기", layout="wide")
    st.header("Step 2: LangGraph — 조건 분기")

    with st.expander("그래프 구조 보기"):
        app = build_graph()
        st.markdown(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")

    st.divider()

    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 파이썬은 누가 만들었나요? / 자바와 파이썬 비교 / 봄에 대한 시를 써주세요 / 시 써"
    )

    if question and st.button("그래프 실행", type="primary"):
        with st.spinner("질문을 분류하고 답변 생성 중..."):
            result = run_graph(question)

        llm_type = result["question_type"]
        routed_type = result["_routed_type"]
        was_rerouted = result["_was_rerouted"]

        st.subheader("1단계: 질문 분류 (classify 노드)")
        st.info(f"**LLM 분류 결과**: {TYPE_LABELS.get(llm_type, llm_type)}")

        st.subheader("2단계: 분기 함수 (route_question)")
        if was_rerouted:
            st.warning(
                f"**재분류 발생!** LLM은 {TYPE_LABELS.get(llm_type, llm_type)}로 분류했지만,\n\n"
                f"분기 함수가 추가 규칙을 적용하여 → **{TYPE_LABELS.get(routed_type, routed_type)}**로 변경"
            )
        else:
            st.success(f"LLM 분류 결과를 그대로 사용: **{TYPE_LABELS.get(routed_type, routed_type)}**")

        st.subheader("3단계: 매핑 → 답변 노드 실행")
        cols = st.columns(3)
        for i, (key, label) in enumerate(TYPE_LABELS.items()):
            with cols[i]:
                if key == routed_type:
                    st.success(f"**{label}** ← 선택됨")
                else:
                    st.markdown(f"{label}")

        st.subheader("최종 답변")
        st.success(result["answer"])

        with st.expander("전체 State 확인"):
            st.json({
                "question": result["question"],
                "question_type": result["question_type"],
                "answer": result["answer"],
            })



if __name__ == "__main__":
    main()
