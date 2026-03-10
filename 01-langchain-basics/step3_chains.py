"""
Step 3: 다양한 체인 — 번역, 분류

학습 목표:
- 같은 LCEL 패턴으로 다양한 작업 체인 만들기
- 번역 체인, 분류 체인 구성
- temperature에 따른 결과 차이 이해
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def create_translate_chain():
    """번역 체인 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 번역가입니다. 주어진 텍스트를 {target_language}로 번역하세요. 번역 결과만 출력하세요."),
        ("human", "{text}")
    ])
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
    parser = StrOutputParser()

    chain = prompt | model | parser
    return chain


def create_classify_chain():
    """분류 체인 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 텍스트 분류 전문가입니다.
주어진 텍스트를 다음 카테고리 중 하나로 분류하세요: {categories}
카테고리 이름만 출력하세요."""),
        ("human", "{text}")
    ])
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser
    return chain


def run_translate():
    """번역 체인 실행"""
    print("-" * 40)
    text = input("번역할 텍스트를 입력하세요: ").strip()
    if not text:
        print("비어있습니다. 돌아갑니다.")
        return

    languages = ["영어", "일본어", "중국어", "프랑스어"]
    print("목표 언어를 선택하세요:")
    for i, lang in enumerate(languages, 1):
        print(f"  {i}. {lang}")
    lang_input = input("번호 입력 (기본값 1): ").strip()
    lang_idx = int(lang_input) - 1 if lang_input.isdigit() and 1 <= int(lang_input) <= len(languages) else 0
    target_lang = languages[lang_idx]

    print(f"\n{target_lang}로 번역 중...")
    chain = create_translate_chain()
    result = chain.invoke({"text": text, "target_language": target_lang})
    print(f"\n[번역 결과]\n{result}")


def run_classify():
    """분류 체인 실행"""
    print("-" * 40)
    text = input("분류할 텍스트를 입력하세요: ").strip()
    if not text:
        print("비어있습니다. 돌아갑니다.")
        return

    categories_input = input("카테고리 목록 (쉼표로 구분, 기본값: 기술, 경제, 스포츠, 문화, 정치): ").strip()
    categories = categories_input if categories_input else "기술, 경제, 스포츠, 문화, 정치"

    print("\n분류 중...")
    chain = create_classify_chain()
    result = chain.invoke({"text": text, "categories": categories})
    print(f"\n[분류 결과] {result}")


def main():
    print("=" * 60)
    print("  Step 3: 번역 / 분류 체인")
    print("=" * 60)


    while True:
        print("-" * 60)
        print("체인을 선택하세요:")
        print("  1. 번역 체인")
        print("  2. 분류 체인")
        print("  q. 종료")

        choice = input("\n선택: ").strip().lower()

        if choice == "1":
            run_translate()
        elif choice == "2":
            run_classify()
        elif choice == "q":
            break
        else:
            print("잘못된 입력입니다.")



if __name__ == "__main__":
    main()
