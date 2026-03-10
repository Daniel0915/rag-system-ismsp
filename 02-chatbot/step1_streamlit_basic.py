"""Step 1: Streamlit 기초 — 입력과 출력"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def ask_ai(question: str) -> str:
    llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
    messages = [
        SystemMessage(content="당신은 친절한 AI 도우미입니다. 한국어로 간결하게 답변하세요."),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content


def main():
    st.set_page_config("02-chatbot | Step 1: Streamlit 기초", layout="wide")
    st.title("Step 1: Streamlit 기초 위젯")

    st.divider()

    st.subheader("1. 텍스트 입력")
    name = st.text_input("이름을 입력하세요", placeholder="예) 홍길동")
    if name:
        st.write(f"안녕하세요, **{name}**님!")

    st.divider()

    st.subheader("2. 버튼으로 AI 호출")
    question = st.text_input(
        "AI에게 질문하세요",
        placeholder="예) 파이썬의 장점 3가지 알려줘"
    )

    if st.button("질문하기", type="primary"):
        if question:
            with st.spinner("AI가 답변을 생성 중..."):
                answer = ask_ai(question)
            st.success("답변 완료!")
            st.write(answer)
        else:
            st.warning("질문을 먼저 입력하세요.")

    st.divider()

    st.subheader("3. 위젯 조합")
    col1, col2 = st.columns(2)

    with col1:
        topic = st.selectbox("주제 선택", ["파이썬", "자바스크립트", "AI", "데이터분석"])
    with col2:
        level = st.radio("난이도", ["초급", "중급", "고급"])

    if st.button("설명 요청", key="combo_btn"):
        prompt = f"{topic}에 대해 {level} 수준으로 한 문단으로 설명해주세요."
        with st.spinner("생성 중..."):
            result = ask_ai(prompt)
        st.info(result)



if __name__ == "__main__":
    main()
