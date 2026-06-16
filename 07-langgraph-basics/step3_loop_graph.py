"""Step 3: LangGraph 루프"""

import json
import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    score: int
    iteration: int
    max_iterations: int


def write_draft(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano")

    iteration = state.get("iteration", 0) + 1

    if iteration == 1:
        prompt = (
            f"다음 주제에 대해 300자 내외의 짧은 글을 작성하세요.\n\n"
            f"주제: {state['topic']}"
        )
    else:
        prompt = (
            f"다음 주제에 대해 글을 다시 작성하세요.\n\n"
            f"주제: {state['topic']}\n\n"
            f"이전 초안:\n{state['draft']}\n\n"
            f"평가자 피드백:\n{state['feedback']}\n\n"
            f"피드백을 반영하여 개선된 300자 내외의 글을 작성하세요."
        )

    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": iteration}


def evaluate_draft(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-5-nano")

    response = llm.invoke(
        f"당신은 글쓰기 평가 전문가입니다. 다음 글을 평가하세요.\n\n"
        f"주제: {state['topic']}\n"
        f"글:\n{state['draft']}\n\n"
        f"반드시 아래 JSON 형식으로만 답하세요. 다른 말은 하지 마세요.\n"
        f'{{"score": <1~10 정수>, "feedback": "<구체적 개선 피드백>"}}'
    )

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        score = int(parsed["score"])
        feedback = parsed["feedback"]
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 5
        feedback = "평가를 파싱할 수 없어 기본값을 사용합니다."

    return {"score": score, "feedback": feedback}


def should_continue(state: State) -> str:
    if state["score"] < 7 and state["iteration"] < state["max_iterations"]:
        return "continue"
    return "end"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("write", write_draft)
    graph.add_node("evaluate", evaluate_draft)

    graph.add_edge(START, "write")
    graph.add_edge("write", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "continue": "write",
            "end": END,
        }
    )

    return graph.compile()


def run_graph_with_history(topic: str, max_iterations: int) -> list[dict]:
    app = build_graph()

    merged = {}
    current_iteration = 0

    for event in app.stream(
        {"topic": topic, "iteration": 0, "max_iterations": max_iterations},
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            if node_name == "write":
                current_iteration = updates["iteration"]
                if current_iteration not in merged:
                    merged[current_iteration] = {}
                merged[current_iteration]["iteration"] = current_iteration
                merged[current_iteration]["draft"] = updates["draft"]

            elif node_name == "evaluate":
                merged[current_iteration]["score"] = updates["score"]
                merged[current_iteration]["feedback"] = updates["feedback"]

    return list(merged.values())


def main():
    st.set_page_config("07-langgraph-basics | Step 3: 루프", layout="wide")
    st.header("Step 3: LangGraph — 루프 (반복 개선)")

    with st.expander("그래프 구조 보기"):
        app = build_graph()
        st.markdown(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")

    st.divider()

    topic = st.text_input(
        "글쓰기 주제를 입력하세요",
        placeholder="예: 인공지능의 미래"
    )
    max_iterations = st.slider("최대 반복 횟수", min_value=1, max_value=5, value=3)

    if topic and st.button("글쓰기 시작", type="primary"):
        with st.spinner("글쓰기 반복 개선 중..."):
            history = run_graph_with_history(topic, max_iterations)

        if not history:
            st.warning("결과가 없습니다.")
            return

        for item in history:
            iteration = item.get("iteration", "?")
            score = item.get("score")
            draft = item.get("draft", "")
            feedback = item.get("feedback", "")

            label = f"반복 {iteration}"
            if score is not None:
                label += f" — 점수: {score}/10"
                if score >= 7:
                    label += " 통과"
                else:
                    label += " → 재작성"

            with st.expander(label, expanded=(item == history[-1])):
                if draft:
                    st.markdown("**초안:**")
                    st.write(draft)
                if score is not None:
                    st.markdown(f"**점수:** {score}/10")
                if feedback:
                    st.markdown(f"**피드백:** {feedback}")

        st.divider()
        last = history[-1]
        st.subheader("최종 결과")
        final_score = last.get("score", "N/A")
        st.metric("최종 점수", f"{final_score}/10")
        st.success(last.get("draft", ""))



if __name__ == "__main__":
    main()
