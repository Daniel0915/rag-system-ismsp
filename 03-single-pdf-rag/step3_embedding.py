"""Step 3: 임베딩 체험 — 의미를 숫자로 표현하기"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def get_embedding(text: str) -> list:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embeddings.embed_query(text)
    return vector


def cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    st.set_page_config("03-single-pdf-rag | Step 3: 임베딩 체험", layout="wide")
    st.header("Step 3: 임베딩 — 의미를 숫자로")

    st.subheader("문장 유사도 비교")

    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_input("문장 1", value="파이썬은 배우기 쉬운 프로그래밍 언어입니다")
    with col2:
        text2 = st.text_input("문장 2", value="Python은 초보자에게 적합한 코딩 언어입니다")

    if text1 and text2:
        if st.button("유사도 계산", type="primary"):
            with st.spinner("임베딩 생성 중..."):
                vec1 = get_embedding(text1)
                vec2 = get_embedding(text2)
                similarity = cosine_similarity(vec1, vec2)

            st.subheader("결과")

            if similarity > 0.8:
                st.success(f"코사인 유사도: **{similarity:.4f}** (매우 유사)")
            elif similarity > 0.5:
                st.warning(f"코사인 유사도: **{similarity:.4f}** (어느 정도 유사)")
            else:
                st.error(f"코사인 유사도: **{similarity:.4f}** (관련 없음)")

            st.progress(min(similarity, 1.0))

            st.subheader("임베딩 벡터 정보")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("벡터 차원", len(vec1))
                st.write("**벡터 미리보기 (처음 10개)**:")
                st.code(str([round(v, 4) for v in vec1[:10]]) + " ...")
            with col_b:
                st.metric("벡터 차원", len(vec2))
                st.write("**벡터 미리보기 (처음 10개)**:")
                st.code(str([round(v, 4) for v in vec2[:10]]) + " ...")

    st.divider()

    st.subheader("여러 문장 비교 실험")

    base_text = st.text_input("기준 문장", value="인공지능이 세상을 바꾸고 있다")
    compare_texts = st.text_area(
        "비교 문장들 (한 줄에 하나씩)",
        value="AI 기술이 혁신을 이끌고 있다\n머신러닝은 데이터에서 패턴을 학습한다\n오늘 날씨가 매우 좋습니다\n주가가 급등하고 있다",
        height=120
    )

    if base_text and compare_texts:
        sentences = [s.strip() for s in compare_texts.strip().split("\n") if s.strip()]

        if st.button("전체 유사도 계산", key="batch_btn"):
            with st.spinner("임베딩 생성 중..."):
                base_vec = get_embedding(base_text)
                results = []
                for sent in sentences:
                    vec = get_embedding(sent)
                    sim = cosine_similarity(base_vec, vec)
                    results.append((sent, sim))

            results.sort(key=lambda x: x[1], reverse=True)

            st.subheader("유사도 순위")
            for rank, (sent, sim) in enumerate(results, 1):
                col_rank, col_sent, col_sim = st.columns([1, 6, 2])
                with col_rank:
                    st.write(f"**{rank}위**")
                with col_sent:
                    st.write(sent)
                with col_sim:
                    st.write(f"`{sim:.4f}`")
                    st.progress(min(sim, 1.0))



if __name__ == "__main__":
    main()
