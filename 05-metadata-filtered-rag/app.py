"""Metadata-Filtered RAG System - Streamlit UI"""

import os
import re
import fitz
import streamlit as st
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = "PDF_임시폴더"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_with_metadata(pdf_path, metadata, vectorstore):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata.update(metadata)
        doc.metadata["source_file"] = filename

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vectorstore.add_documents(splits)

    return vectorstore


def create_filtered_rag_chain(vectorstore, filter_metadata=None):
    template = """당신은 문서 기반 질문 답변 전문가입니다. 제공된 문서를 바탕으로 질문에 상세하고 정확하게 답변해주세요.

## 답변 가이드라인
1. 제공된 문서의 내용만을 기반으로 답변하세요.
2. 문서에 포함된 구체적인 수치, 날짜, 이름 등의 정보를 포함하여 답변하세요.
3. 문서에서 관련 정보를 찾을 수 없는 경우 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
4. 답변은 명확하고 구조적으로 작성하세요. 불렛 포인트나 번호를 사용하여 가독성을 높이세요.
5. 문서 내용을 인용하거나 참조할 때 출처를 명확히 하세요.

## 참고 문서
{context}

## 질문
{question}

## 답변
위 문서를 바탕으로 질문에 답변하세요:"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    search_kwargs = {"k": 6}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    llm = ChatOpenAI(model="gpt-5-nano", temperature=1)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )


@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []

    basename = Path(pdf_path).stem
    output_folder = os.path.join("PDF_이미지", basename)
    os.makedirs(output_folder, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths


def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def main():
    st.set_page_config("05-metadata-filtered-rag", layout="wide")

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_info' not in st.session_state:
        st.session_state.doc_info = None
    if 'pdfs_loaded' not in st.session_state:
        st.session_state.pdfs_loaded = False
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = {}
    if 'uploaded_files_list' not in st.session_state:
        st.session_state.uploaded_files_list = []

    with st.sidebar:
        st.header("설정")
        st.markdown("---")

        st.subheader("PDF 문서")
        uploaded_file = st.file_uploader("PDF 파일을 업로드 해주세요", type=["pdf"])

        if uploaded_file:
            st.markdown("#### 메타데이터")
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox(
                    "카테고리",
                    ["계약서", "매뉴얼", "보고서", "정책", "기타"],
                    key="metadata_category"
                )
                department = st.selectbox(
                    "부서",
                    ["영업", "인사", "개발", "경영", "마케팅", "고객지원", "기타"],
                    key="metadata_department"
                )
            with col2:
                year = st.selectbox(
                    "연도",
                    list(range(2000, 2031)),
                    index=24,
                    key="metadata_year"
                )
                language = st.selectbox(
                    "언어",
                    ["한국어", "영어", "일본어", "중국어"],
                    key="metadata_language"
                )
            priority = st.selectbox(
                "우선순위",
                ["높음", "중간", "낮음"],
                key="metadata_priority"
            )

        upload_button = st.button("문서 저장")

        if uploaded_file and upload_button:
            metadata = {
                "category": category,
                "department": department,
                "year": year,
                "language": language,
                "priority": priority
            }

            with st.spinner("문서를 저장하고 있습니다..."):
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.vectorstore = load_pdf_with_metadata(
                    pdf_path=file_path,
                    metadata=metadata,
                    vectorstore=st.session_state.vectorstore
                )
                st.session_state.pdfs_loaded = True
                st.session_state.uploaded_files_list.append((uploaded_file.name, metadata))

                st.session_state.doc_info = {
                    'files': [f[0] for f in st.session_state.uploaded_files_list],
                    'count': len(st.session_state.uploaded_files_list),
                    'chunks': st.session_state.vectorstore._collection.count()
                }
                st.success(f"{uploaded_file.name} 이(가) 저장되었습니다!")

            with st.spinner("PDF 페이지를 이미지로 변환하는 중입니다..."):
                images = convert_pdf_to_images(file_path)
                st.session_state.pdf_images[uploaded_file.name] = images

        elif st.session_state.vectorstore is None and os.path.exists(CHROMA_DIR):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                st.session_state.pdfs_loaded = True

                collection = st.session_state.vectorstore._collection
                results = collection.get(include=["metadatas"])
                if results and results['metadatas']:
                    filenames = set()
                    for m in results['metadatas']:
                        filenames.add(m.get('source_file', 'Unknown'))
                    st.session_state.doc_info = {
                        'files': sorted(list(filenames)),
                        'count': len(filenames),
                        'chunks': collection.count()
                    }
                st.info("기존 문서를 로드했습니다.")
            except Exception:
                st.session_state.vectorstore = None

        if st.session_state.doc_info:
            st.markdown("---")
            st.markdown("### 로드된 문서")
            for fname in st.session_state.doc_info['files']:
                st.markdown(f"- {fname}")
            st.markdown(f"**파일 수:** {st.session_state.doc_info['count']}개")
            st.markdown(f"**청크:** {st.session_state.doc_info['chunks']}개")

            if st.button("모든 문서 삭제", use_container_width=True):
                import shutil, gc
                if st.session_state.vectorstore:
                    try:
                        st.session_state.vectorstore.delete_collection()
                    except Exception:
                        pass
                st.session_state.vectorstore = None
                gc.collect()
                try:
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                except Exception:
                    pass
                st.session_state.pdfs_loaded = False
                st.session_state.doc_info = None
                st.session_state.messages = []
                st.session_state.pdf_images = {}
                st.session_state.uploaded_files_list = []
                for key in ['last_context_docs', 'page_number', 'page_source', 'last_question']:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("---")
        st.markdown("### 모델 정보")
        st.markdown("- **LLM:** gpt-5-nano")
        st.markdown("- **Embedding:** text-embedding-3-small")
        st.markdown("---")
        st.markdown("### 사용 방법")
        st.markdown("1. PDF 파일 업로드")
        st.markdown("2. 메타데이터 입력")
        st.markdown("3. '문서 저장' 클릭")
        st.markdown("4. 질문 입력 (필터 선택 가능)")

    st.header("Metadata-Filtered RAG")

    tab_chat, tab_chunks = st.tabs(["채팅", "청크 미리보기"])

    with tab_chat:
        filter_metadata = None
        if st.session_state.vectorstore:
            st.markdown("#### 검색 필터")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                selected_category = st.selectbox(
                    "카테고리", ["전체", "계약서", "매뉴얼", "보고서", "정책", "기타"],
                    key="filter_category"
                )
                selected_department = st.selectbox(
                    "부서", ["전체", "영업", "인사", "개발", "경영", "마케팅", "고객지원", "기타"],
                    key="filter_department"
                )
            with fc2:
                selected_year = st.selectbox(
                    "연도", ["전체"] + [str(y) for y in range(2000, 2031)],
                    key="filter_year"
                )
                selected_language = st.selectbox(
                    "언어", ["전체", "한국어", "영어", "일본어", "중국어"],
                    key="filter_language"
                )
            with fc3:
                selected_priority = st.selectbox(
                    "우선순위", ["전체", "높음", "중간", "낮음"],
                    key="filter_priority"
                )

            filter_metadata = {}
            if selected_category != "전체":
                filter_metadata["category"] = selected_category
            if selected_department != "전체":
                filter_metadata["department"] = selected_department
            if selected_year != "전체":
                filter_metadata["year"] = int(selected_year)
            if selected_language != "전체":
                filter_metadata["language"] = selected_language
            if selected_priority != "전체":
                filter_metadata["priority"] = selected_priority

            if len(filter_metadata) >= 2:
                st.info(f"적용된 필터: {', '.join([f'{k}={v}' for k, v in filter_metadata.items()])}")
                filter_metadata = {"$and": [{k: v} for k, v in filter_metadata.items()]}
            elif filter_metadata:
                st.info(f"적용된 필터: {', '.join([f'{k}={v}' for k, v in filter_metadata.items()])}")
            else:
                filter_metadata = None

            st.markdown("---")

        left_column, right_column = st.columns([1, 1])

        with left_column:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input(
                "질문을 입력해주세요",
                disabled=not st.session_state.pdfs_loaded
            )

            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.last_question = prompt
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중..."):
                        chain = create_filtered_rag_chain(
                            st.session_state.vectorstore, filter_metadata
                        )
                        result = chain.invoke({"query": prompt})
                        answer = result["result"]
                        sources = result["source_documents"]
                    st.markdown(answer)

                st.session_state.last_context_docs = sources
                st.session_state.messages.append({"role": "assistant", "content": answer})

            if st.session_state.get("last_context_docs"):
                if not prompt:
                    st.markdown("---")
                st.markdown("### 관련 문서")
                for idx, document in enumerate(st.session_state.last_context_docs):
                    source = document.metadata.get('source_file', 'Unknown')
                    page = document.metadata.get('page', 0) + 1
                    cat = document.metadata.get('category', '')
                    label = f"**{source} — 페이지 {page}**"
                    if cat:
                        label += f" [{cat}]"
                    st.markdown(label)

                    st.text_area(
                        f"청크 {idx + 1}",
                        value=document.page_content,
                        height=150,
                        key=f"chunk_{idx}_{st.session_state.get('last_question', '')}"
                    )

                    if st.button(f"PDF {page}페이지 보기", key=f"btn_{page}_{idx}"):
                        st.session_state.page_number = str(page)
                        st.session_state.page_source = source
                        st.rerun()

        with right_column:
            if st.session_state.get("page_number"):
                st.subheader("PDF 뷰어")
                page_number = int(st.session_state.page_number)
                source = st.session_state.get("page_source", "")

                image_paths = st.session_state.pdf_images.get(source, [])

                if not image_paths:
                    stem = Path(source).stem if source else ""
                    image_folder = os.path.join("PDF_이미지", stem)
                    if os.path.exists(image_folder):
                        images = sorted(os.listdir(image_folder), key=natural_sort_key)
                        image_paths = [os.path.join(image_folder, img) for img in images]

                if image_paths and 0 < page_number <= len(image_paths):
                    display_pdf_page(image_paths[page_number - 1], page_number)

                    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
                    with nav_col1:
                        if page_number > 1 and st.button("이전"):
                            st.session_state.page_number = str(page_number - 1)
                            st.rerun()
                    with nav_col3:
                        if page_number < len(image_paths) and st.button("다음"):
                            st.session_state.page_number = str(page_number + 1)
                            st.rerun()
                else:
                    st.warning("PDF 이미지를 찾을 수 없습니다. PDF를 다시 업로드해주세요.")
            else:
                st.info("질문을 입력하고 'PDF 페이지 보기' 버튼을 클릭하면 여기에 PDF가 표시됩니다.")

    with tab_chunks:
        st.subheader("저장된 청크 미리보기")
        if st.session_state.vectorstore:
            try:
                collection = st.session_state.vectorstore._collection
                result = collection.get(include=["documents", "metadatas"])
                docs = result.get("documents", [])
                metas = result.get("metadatas", [])

                st.info(f"총 **{len(docs)}**개 청크가 저장되어 있습니다.")

                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    source = meta.get("source_file", "Unknown")
                    page = meta.get("page", "?")
                    page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                    category = meta.get("category", "")
                    label = f"청크 #{i+1} — {source} (p.{page_display})"
                    if category:
                        label += f" [{category}]"
                    with st.expander(label):
                        st.text(doc)
                        st.caption(f"메타데이터: {meta}")
            except Exception as e:
                st.error(f"청크 조회 실패: {e}")
        else:
            st.warning("벡터 저장소가 비어 있습니다. PDF를 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
