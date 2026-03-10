"""Step 1: LangGraph 기초"""

import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class State(TypedDict):
    question: str
    analysis: str
    answer: str


def analyze_question(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    response = llm.invoke(
        f"다음 질문을 한 줄로 분석해주세요. 어떤 종류의 질문인지, 핵심이 뭔지만 간단히:\n\n{state['question']}"
    )
    return {"analysis": response.content}


def generate_answer(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
    response = llm.invoke(
        f"질문: {state['question']}\n"
        f"질문 분석: {state['analysis']}\n\n"
        f"위 분석을 참고하여 친절하게 답변해주세요."
    )
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(State)

    graph.add_node("analyze", analyze_question)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_graph(question: str) -> dict:
    app = build_graph()
    result = app.invoke({"question": question})
    return result


def main():
    st.set_page_config("07-langgraph-basics | Step 1: LangGraph 기초", layout="wide")
    st.header("Step 1: LangGraph — 그래프 워크플로우")

    with st.expander("그래프 구조 보기"):
        app = build_graph()
        st.markdown(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")

    st.divider()

    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 파이썬에서 리스트와 튜플의 차이점은?"
    )

    if question and st.button("그래프 실행", type="primary"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1단계: 질문 분석")
            with st.spinner("질문을 분석 중..."):
                result = run_graph(question)
            st.info(result["analysis"])

        with col2:
            st.subheader("2단계: 답변 생성")
            st.success(result["answer"])

        with st.expander("전체 State 확인"):
            st.json({
                "question": result["question"],
                "analysis": result["analysis"],
                "answer": result["answer"]
            })



if __name__ == "__main__":
    main()
