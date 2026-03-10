"""
Step 1: OpenAI API 첫 호출

학습 목표:
- OpenAI API Key를 설정하는 방법
- ChatCompletion API의 기본 구조 (model, messages, temperature)
- system / user / assistant 메시지 역할
"""

from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()


def call_openai(user_message: str, temperature: float = 0.7) -> str:
    """OpenAI API를 호출하여 응답을 받는 함수"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "당신은 친절한 AI 도우미입니다. 한국어로 답변하세요."},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("  Step 1: OpenAI API 첫 호출")
    print("=" * 60)


    while True:
        print("-" * 60)

        # Temperature 설정
        temp_input = input("Temperature를 입력하세요 (0.0~1.0, 기본값 0.7): ").strip()
        if temp_input == "":
            temperature = 0.7
        else:
            try:
                temperature = float(temp_input)
                temperature = max(0.0, min(1.0, temperature))
            except ValueError:
                print("잘못된 입력입니다. 기본값 0.7을 사용합니다.")
                temperature = 0.7

        print(f"  -> Temperature: {temperature} (0: 일관된 답변 | 1: 창의적 답변)")

        # 질문 입력
        user_message = input("\n질문을 입력하세요 (예: 파이썬이 뭐야?): ").strip()
        if not user_message:
            print("질문이 비어있습니다. 다시 시도하세요.")
            continue

        print("\nAI가 답변을 생성하는 중...")
        answer = call_openai(user_message, temperature)

        print(f"\n[AI 응답]\n{answer}")


        # 계속 여부
        again = input("\n다시 질문하시겠습니까? (y/n, 기본값 y): ").strip().lower()
        if again == "n":
            break



if __name__ == "__main__":
    main()
