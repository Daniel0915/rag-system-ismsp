"""
Step 2: 역할 부여 — system 메시지로 AI 행동 제어

학습 목표:
- system 메시지로 AI에게 역할을 부여하는 방법
- 같은 모델이라도 프롬프트에 따라 완전히 다른 결과가 나온다는 것
ex : 파이썬(Python)은 1991년 발표된 이후 간결하고 직관적인 문법으로 전 세계적인 사랑을 받고 있는 프로그래밍 언어입니다. 초보자도 쉽게 배울 수 있어 접근성이 높으면서도, 강력한 확장성을 갖춰 웹 개발, 데이터 분석, 인공지능 등 다양한 분야에서 핵심적으로 활용됩니다. 또한, 방대한 라이브러리와 활발한 커뮤니티 지원 덕분에 개발자들이 복잡한 문제를 효율적으로 해결할 수 있도록 돕습니다. 이러한 이유로 파이썬은 교육, 연구, 산업 전반에서 가장 인기 있는 프로그래밍 언어 중 하나로 자리매김하고 있습니다.
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
