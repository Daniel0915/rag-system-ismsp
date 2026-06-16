"""Step 2: 이미지 OCR

이미지는 PDF/CSV처럼 바로 읽는 로더가 없다. 그래서 GPT Vision으로 "보고 → 텍스트로"
변환한 뒤, 다른 파일들과 똑같이 Document 로 포장한다.

처리 흐름:
    이미지 파일
      → ① base64 인코딩      (바이너리를 텍스트로 안전하게 포장)
      → ② MIME 타입 결정      ("image/png" 등 데이터 종류 라벨)
      → ③ 멀티모달 메시지      (질문 텍스트 + 이미지를 한 메시지에)
      → ④ GPT Vision 호출      (JSON 모드로 결과를 받음)
      → ⑤ {extracted_text, description} 파싱
      → ⑥ Document 포장        (page_content = 추출 텍스트만, 설명은 metadata)
"""

import os
import json
import base64
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


# ── ① 이미지 파일 → base64 문자열 ───────────────────────────────
def image_to_base64(image_path: str) -> str:
    # "rb"(바이너리 읽기)로 날것 바이트를 읽어 → base64 글자로 변환
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── ② 확장자 → MIME 타입 ────────────────────────────────────────
def get_mime_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")  # 모르는 확장자는 png로 가정


# ── ③~⑤ 이미지 → GPT Vision → {extracted_text, description} ──────
def extract_image_data(image_path: str) -> dict:
    """이미지를 GPT Vision에 보내 추출 텍스트와 설명을 JSON으로 받아온다."""
    # ③ 멀티모달 메시지 만들기: 질문 텍스트 + (base64로 포장한) 이미지
    image_b64 = image_to_base64(image_path)
    mime_type = get_mime_type(image_path)

    message = HumanMessage(content=[
        {"type": "text", "text": (
            "이 이미지를 분석해 아래 JSON 형식으로 답하세요.\n"
            '{\n'
            '  "extracted_text": "이미지에서 읽을 수 있는 모든 글자를 위에서 아래 순서로 원문 그대로. 묘사·군말 없이. 글자가 없으면 빈 문자열",\n'
            '  "description": "배경·표·도장·사진 등 시각적 특징을 한국어로 간단히 설명"\n'
            '}'
        )},
        {"type": "image_url", "image_url": {
            "url": f"data:{mime_type};base64,{image_b64}"
        }},
    ])

    # ④ GPT Vision 호출 — JSON 모드라 항상 유효한 JSON 문자열이 온다
    vision_llm = ChatOpenAI(
        model="gpt-5.4-mini",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    response = vision_llm.invoke([message])

    # ⑤ JSON 문자열 → dict
    return json.loads(response.content)


# ── ⑥ 분석 결과 → Document 포장 ─────────────────────────────────
def build_image_document(image_path: str, data: dict) -> Document:
    filename = os.path.basename(image_path)
    extracted = data.get("extracted_text", "")
    description = data.get("description", "")

    # page_content(= 청킹·임베딩 대상)에는 추출 텍스트만 담는다.
    # 글자 없는 사진이면 검색이 되도록 설명으로 대체한다.
    body = extracted or description

    return Document(
        page_content=body,  # 파일명은 metadata로 — 본문엔 추출 텍스트만
        metadata={
            "filename": filename,
            "file_type": Path(image_path).suffix.lower(),
            "ocr_engine": "gpt-vision",
            "description": description,  # 검색 대상 아님 — 표시용으로만 보관
        },
    )


def load_image_as_document(image_path: str) -> Document:
    """이미지 한 장 → Document. (③~⑥을 한 번에; step3 파이프라인이 이 함수를 사용)"""
    data = extract_image_data(image_path)
    return build_image_document(image_path, data)


def main():
    st.set_page_config("06-rag-multi-format | Step 2: 이미지 OCR", layout="wide")
    st.header("Step 2: 이미지 → 텍스트 (GPT Vision)")

    st.divider()

    uploaded_image = st.file_uploader(
        "이미지 파일을 업로드하세요",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        help="JPG, PNG, GIF, BMP, WEBP 지원",
    )

    if not uploaded_image:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        st.image(uploaded_image, width=400)

    # 업로드 파일을 디스크에 저장
    upload_dir = str(BASE_DIR / "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_image.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    with col2:
        st.subheader("추출 결과")

        if not st.button("텍스트 추출 (GPT Vision)", type="primary"):
            return

        # 데모라 중간 결과(추출 텍스트 vs 설명)를 따로 보여주려고 단계를 나눠 호출
        with st.spinner("GPT Vision이 이미지를 분석 중..."):
            data = extract_image_data(file_path)
            document = build_image_document(file_path, data)

        extracted = data.get("extracted_text", "")
        description = data.get("description", "")

        st.success(f"추출 완료! (추출 텍스트 {len(extracted)}자)")

        st.markdown("**📄 추출된 텍스트** — 이 부분만 청킹·임베딩됩니다")
        st.text_area(
            "extracted_text", value=extracted or "(읽을 수 있는 글자 없음)",
            height=250, label_visibility="collapsed",
        )

        st.markdown("**🖼️ 이미지 설명** — 화면 표시용 (청킹 안 됨)")
        st.info(description or "(설명 없음)")

        with st.expander("청킹 대상 page_content (실제 벡터DB에 들어가는 값)"):
            st.code(document.page_content)

        with st.expander("Document 메타데이터"):
            st.json(document.metadata)

        with st.expander("base64 미리보기 (처음 100자)"):
            b64 = image_to_base64(file_path)
            st.code(b64[:100] + "...")
            st.caption(f"전체 길이: {len(b64):,}자")


if __name__ == "__main__":
    main()
