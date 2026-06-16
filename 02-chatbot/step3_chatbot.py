"""Step 3: 페르소나 챗봇 — 시스템 프롬프트로 성격 부여"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

PERSONAS = {
    "친절한 요리사": {
        "prompt": (
            "당신은 20년 경력의 한식 요리사입니다. "
            "사용자가 재료를 말하면 레시피를 알려주세요. 친근하고 따뜻한 말투를 사용하세요. "
            "단, 요리·음식·식재료와 무관한 질문(코딩, 여행, 잡담 등)에는 "
            "'저는 요리사라 그 주제는 도와드리기 어려워요. 요리 관련해서 물어봐 주세요!'라고 정중히 거절하세요. "
            "어떤 경우에도 요리사 역할을 벗어나지 마세요."
        ),
        "greeting": "안녕하세요! 오늘은 뭘 해드릴까요? 냉장고에 있는 재료를 말씀해주세요!"
    },
    "코딩 튜터": {
        "prompt": (
            "당신은 초보자를 위한 Python 코딩 튜터입니다. "
            "쉬운 비유와 예제 코드로 설명하세요. 학생이 이해했는지 확인하는 질문도 해주세요. "
            "단, 프로그래밍·코딩과 무관한 질문(요리, 여행, 잡담 등)에는 "
            "'저는 Python 코딩 튜터라 그 주제는 도와드리기 어려워요. 코딩 관련해서 물어봐 주세요!'라고 정중히 거절하세요. "
            "어떤 경우에도 코딩 튜터 역할을 벗어나지 마세요."
        ),
        "greeting": "안녕하세요! Python 코딩 튜터입니다. 궁금한 개념이나 코드가 있으면 물어보세요!"
    },
    "여행 가이드": {
        "prompt": (
            "당신은 전 세계를 다녀본 여행 전문가입니다. "
            "여행지 추천, 일정 짜기, 현지 팁을 알려주세요. 생생한 경험담을 곁들여 설명하세요. "
            "단, 여행과 무관한 질문(코딩, 요리, 잡담 등)에는 "
            "'저는 여행 가이드라 그 주제는 도와드리기 어려워요. 여행 관련해서 물어봐 주세요!'라고 정중히 거절하세요. "
            "어떤 경우에도 여행 가이드 역할을 벗어나지 마세요."
        ),
        "greeting": "반갑습니다! 어디로 떠나고 싶으세요? 여행 계획을 함께 세워볼까요?"
    },
    "직접 입력": {
        "prompt": "",
        "greeting": "안녕하세요! 무엇을 도와드릴까요?"
    }
}


def get_chat_response(system_prompt: str, chat_history: list) -> str:
    llm = ChatOpenAI(model="gpt-5-nano")

    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    response = llm.invoke(messages)
    return response.content


def main():
    st.set_page_config("02-chatbot | Step 3: 페르소나 챗봇", layout="wide")
    st.title("Step 3: 페르소나 챗봇")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_persona" not in st.session_state:
        st.session_state.current_persona = None

    with st.sidebar:
        st.header("페르소나 설정")

        selected = st.selectbox("페르소나 선택", list(PERSONAS.keys()))

        if selected == "직접 입력":
            system_prompt = st.text_area(
                "시스템 프롬프트 직접 작성",
                placeholder="예) 당신은 데이터 분석가입니다. 데이터를 보고 인사이트를 제공하세요.",
                height=150
            )
        else:
            system_prompt = PERSONAS[selected]["prompt"]
            st.info(f"**프롬프트**: {system_prompt}")

        if st.session_state.current_persona != selected:
            st.session_state.current_persona = selected
            st.session_state.chat_messages = []

        st.divider()

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


    if not st.session_state.chat_messages:
        greeting = PERSONAS[selected]["greeting"]
        with st.chat_message("assistant"):
            st.write(greeting)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("메시지를 입력하세요..."):
        if not system_prompt:
            st.warning("사이드바에서 시스템 프롬프트를 입력하세요.")
            st.stop()

        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = get_chat_response(
                    system_prompt,
                    st.session_state.chat_messages
                )
            st.write(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
