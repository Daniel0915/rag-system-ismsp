"""
Step 1: LangChain ChatModel — ChatOpenAI 사용하기

학습 목표:
- LangChain이 필요한 이유 (프레임워크의 가치)
- ChatOpenAI로 LLM을 호출하는 방법
- raw OpenAI와 LangChain 방식의 차이
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


def call_with_langchain(user_message: str, temperature: float = 0.7) -> str:
    """LangChain ChatOpenAI로 LLM 호출"""
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=temperature
    )

    messages = [
        SystemMessage(content="당신은 친절한 AI 도우미입니다. 한국어로 답변하세요."),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    return response.content


def main():
    print("=" * 60)
    print("Step 1: LangChain ChatModel")
    print("=" * 60)


    print("-" * 60)

    while True:
        user_message = input("\n질문을 입력하세요 (종료: q): ").strip()
        if not user_message or user_message.lower() == "q":
            print("프로그램을 종료합니다.")
            break

        print("LangChain으로 답변 생성 중...")
        answer = call_with_langchain(user_message)

        print(f"\n[AI 응답]\n{answer}")


if __name__ == "__main__":
    main()
