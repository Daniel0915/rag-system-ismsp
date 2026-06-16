"""Step 2: 채팅 UI — st.chat_message + st.session_state"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from pathlib import Path

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_ai_response(messages: list) -> str:
    llm = ChatOpenAI(model="gpt-5-nano")
    response = llm.invoke(messages)
    return response.content


def main():
    st.set_page_config("02-chatbot | Step 2: 채팅 UI", layout="wide")
    st.title("Step 2: 채팅 UI 만들기")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("채팅 설정")
        if st.button("대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.metric("대화 수", len(st.session_state.messages))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        langchain_messages = [
            SystemMessage(content="당신은 친절한 AI 도우미입니다. 한국어로 답변하세요.")
        ]
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = get_ai_response(langchain_messages)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
