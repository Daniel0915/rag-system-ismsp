"""
Step 3: 출력 형식 제어 — JSON 구조화 출력

학습 목표:
- AI에게 JSON 형식으로 응답하도록 지시하는 방법
- 응답을 파싱하여 프로그램에서 활용하는 방법
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def call_json_output(user_message: str) -> str:
    """JSON 형식으로 응답을 받는 함수"""
    system_prompt = """당신은 텍스트를 분석하는 AI입니다.
반드시 아래 JSON 형식으로만 응답하세요:
{
    "summary": "한 줄 요약",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "sentiment": "긍정/부정/중립",
    "category": "카테고리"
}"""
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("  Step 3: JSON 구조화 출력")
    print("=" * 60)


    while True:
        print("-" * 60)
        text = input("분석할 텍스트를 입력하세요 (q: 종료)\n> ").strip()

        if text.lower() == "q":
            break
        if not text:
            print("비어있습니다. 다시 입력하세요.")
            continue

        print("\n텍스트 분석 중...")
        json_result = call_json_output(text)

        print(f"\n[분석 결과 (JSON)]\n{json_result}")

        try:
            parsed = json.loads(json_result)
            print(f"\n--- 파싱된 결과 ---")
            print(f"  감정:     {parsed.get('sentiment', '-')}")
            print(f"  카테고리: {parsed.get('category', '-')}")
            print(f"  키워드:   {', '.join(parsed.get('keywords', []))}")
            print(f"  요약:     {parsed.get('summary', '-')}")
            print(f"-------------------")
        except json.JSONDecodeError:
            print("\nJSON 파싱에 실패했습니다. 다시 시도해보세요.")



if __name__ == "__main__":
    main()
