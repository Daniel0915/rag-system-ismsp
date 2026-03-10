"""
Step 2: PromptTemplate — 재사용 가능한 프롬프트

학습 목표:
- ChatPromptTemplate으로 변수가 있는 프롬프트 만들기
- LCEL 파이프 연산자 (prompt | model | parser)
- StrOutputParser로 텍스트만 추출
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def create_summary_chain():
    """요약 체인 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 요약 전문가입니다. 주어진 텍스트를 {length}줄로 요약하세요."),
        ("human", "{text}")
    ])
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    parser = StrOutputParser()

    # LCEL 파이프: prompt → model → parser
    chain = prompt | model | parser
    return chain


def main():
    print("=" * 60)
    print("  Step 2: PromptTemplate + LCEL 체인")
    print("=" * 60)


    while True:
        print("-" * 60)
        text = input("요약할 텍스트를 입력하세요 (q: 종료)\n> ").strip()

        if text.lower() == "q":
            break
        if not text:
            print("비어있습니다. 다시 입력하세요.")
            continue

        length_input = input("요약 길이 (줄 수, 기본값 3): ").strip()
        length = int(length_input) if length_input.isdigit() else 3

        print("\n요약 중...")
        chain = create_summary_chain()
        result = chain.invoke({"text": text, "length": length})
        print(f"\n[요약 결과]\n{result}")


if __name__ == "__main__":
    main()
