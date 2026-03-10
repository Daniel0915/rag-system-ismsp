"""
Step 2: 역할 부여 — system 메시지로 AI 행동 제어

학습 목표:
- system 메시지로 AI에게 역할을 부여하는 방법
- 같은 모델이라도 프롬프트에 따라 완전히 다른 결과가 나온다는 것
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def call_with_role(user_message: str, system_prompt: str) -> str:
    """역할이 부여된 AI에게 질문하는 함수"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("  Step 2: 역할 부여 (system 메시지)")
    print("=" * 60)


    role_options = {
        "1": ("번역가", "당신은 전문 번역가입니다. 사용자가 입력한 텍스트를 영어로 번역하세요. 번역만 출력하세요."),
        "2": ("코드 리뷰어", "당신은 시니어 Python 개발자입니다. 사용자의 코드를 리뷰하고 개선점을 알려주세요."),
        "3": ("요약 전문가", "당신은 요약 전문가입니다. 사용자가 입력한 텍스트를 3줄 이내로 요약하세요."),
        "4": ("직접 입력", ""),
    }

    while True:
        print("-" * 60)
        print("역할을 선택하세요:")
        for key, (name, _) in role_options.items():
            print(f"  {key}. {name}")
        print("  q. 종료")

        choice = input("\n번호 입력: ").strip().lower()
        if choice == "q":
            break
        if choice not in role_options:
            print("잘못된 선택입니다. 다시 선택하세요.")
            continue

        role_name, system_prompt = role_options[choice]

        if choice == "4":
            system_prompt = input("System Prompt를 직접 작성하세요: ").strip()
            if not system_prompt:
                print("비어있습니다. 돌아갑니다.")
                continue
        else:
            print(f"\n[System Prompt]: {system_prompt}")

        user_input = input("\n입력: ").strip()
        if not user_input:
            print("비어있습니다. 돌아갑니다.")
            continue

        print("\n응답 생성 중...")
        result = call_with_role(user_input, system_prompt)
        print(f"\n[AI 응답]\n{result}")



if __name__ == "__main__":
    main()
