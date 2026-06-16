"""Step 2: 이미지 OCR"""

import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from pathlib import Path
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    return mime_map.get(ext, "image/png")


def extract_text_from_image(image_path: str) -> str:
    image_b64 = image_to_base64(image_path)
    mime_type = get_mime_type(image_path)

    vision_llm = ChatOpenAI(model="gpt-5.4-mini")

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "이 이미지의 내용을 상세하게 한국어로 설명해주세요. "
                "텍스트가 있다면 모두 추출하고, 표/차트/그래프가 있다면 내용을 설명하고, "
                "사진이라면 무엇이 보이는지 설명해주세요."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}"
            }
        }
    ])

    response = vision_llm.invoke([message])
    return response.content


def load_image_as_document(image_path: str) -> Document:
    text = extract_text_from_image(image_path)
    filename = os.path.basename(image_path)

    return Document(
        page_content=f"[이미지: {filename}]\n{text}",
        metadata={
            "filename": filename,
            "file_type": Path(image_path).suffix.lower(),
            "ocr_engine": "gpt-vision"
        }
    )


def main():
    st.set_page_config("06-rag-multi-format | Step 2: 이미지 OCR", layout="wide")
    st.header("Step 2: 이미지 → 텍스트 (GPT Vision)")

    st.divider()

    uploaded_image = st.file_uploader(
        "이미지 파일을 업로드하세요",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        help="JPG, PNG, GIF, BMP, WEBP 지원"
    )

    if uploaded_image:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본 이미지")
            st.image(uploaded_image, width=400)

        upload_dir = str(BASE_DIR / "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        with col2:
            st.subheader("추출 결과")

            if st.button("텍스트 추출 (GPT Vision)", type="primary"):
                with st.spinner("GPT Vision이 이미지를 분석 중..."):
                    document = load_image_as_document(file_path)

                st.success(f"추출 완료! ({len(document.page_content)}자)")
                st.write(document.page_content)

                with st.expander("Document 메타데이터"):
                    st.json(document.metadata)

                with st.expander("base64 미리보기 (처음 100자)"):
                    b64 = image_to_base64(file_path)
                    st.code(b64[:100] + "...")
                    st.caption(f"전체 길이: {len(b64):,}자")



if __name__ == "__main__":
    main()
