"""Multi-PDF RAG System - Streamlit UI"""

import streamlit as st
import os
import re
import fitz  # PyMuPDF
from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from step1_multi_load import save_uploaded_files, load_multiple_pdfs, chunk_documents, save_to_vector_store, CHROMA_DIR

# 환경변수 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def create_rag_chain(vector_store):
    llm = ChatOpenAI(model="gpt-5-nano", temperature=1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 기반 질문응답 AI 어시스턴트입니다.
제공된 컨텍스트에서 답을 찾아주세요.
컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다"라고 답변하세요.
한국어로 간결하고 명확하게 답변하세요."""),
        ("human", """다음 문서를 참고하여 질문에 답변해주세요:

[문서 내용]
{context}

---

질문: {input}""")
    ])

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


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
    st.set_page_config("04-multi-pdf-rag", layout="wide")

    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdfs_loaded' not in st.session_state:
        st.session_state.pdfs_loaded = False
    if 'doc_info' not in st.session_state:
        st.session_state.doc_info = None
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = {}

    with st.sidebar:
        st.header("설정")
        st.markdown("---")

        st.subheader("PDF 문서")
        pdf_docs = st.file_uploader(
            "PDF 파일을 업로드 해주세요 (여러 개 가능)",
            type=["pdf"],
            accept_multiple_files=True
        )
        upload_button = st.button("PDF 문서 저장")

        if pdf_docs and upload_button:
            with st.spinner("PDF 문서를 저장하고 있습니다..."):
                pdf_paths = save_uploaded_files(pdf_docs)
                all_docs = load_multiple_pdfs(pdf_paths)
                chunks = chunk_documents(all_docs)
                vector_store = save_to_vector_store(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = create_rag_chain(vector_store)
                st.session_state.pdfs_loaded = True

                total_size = sum(f.size for f in pdf_docs)
                st.session_state.doc_info = {
                    'files': [f.name for f in pdf_docs],
                    'count': len(pdf_docs),
                    'size': f"{total_size / 1024 / 1024:.2f} MB",
                    'chunks': vector_store._collection.count()
                }
                st.success(f"{len(pdf_docs)}개 PDF 문서가 저장되었습니다!")

            with st.spinner("PDF 페이지를 이미지로 변환하는 중입니다..."):
                for pdf_path in pdf_paths:
                    basename = Path(pdf_path).name
                    images = convert_pdf_to_images(pdf_path)
                    st.session_state.pdf_images[basename] = images

        elif st.session_state.rag_chain is None and os.path.exists(CHROMA_DIR):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_store = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings,
                    collection_name="documents",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = create_rag_chain(vector_store)
                st.session_state.pdfs_loaded = True

                collection = vector_store._collection
                results = collection.get(include=["metadatas"])
                if results and results['metadatas']:
                    filenames = set()
                    for m in results['metadatas']:
                        filenames.add(m.get('source_file', 'Unknown'))

                    st.session_state.doc_info = {
                        'files': sorted(list(filenames)),
                        'count': len(filenames),
                        'size': 'N/A',
                        'chunks': len(results['ids'])
                    }
                st.info("기존 문서를 로드했습니다.")
            except Exception:
                st.session_state.rag_chain = None
                st.session_state.vector_store = None

        if st.session_state.doc_info:
            st.markdown("---")
            st.markdown("### 로드된 문서")
            for fname in st.session_state.doc_info['files']:
                st.markdown(f"- {fname}")
            st.markdown(f"**파일 수:** {st.session_state.doc_info['count']}개")
            st.markdown(f"**크기:** {st.session_state.doc_info['size']}")
            st.markdown(f"**청크:** {st.session_state.doc_info['chunks']}개")

            if st.button("모든 문서 삭제", use_container_width=True):
                import shutil, gc
                if st.session_state.vector_store:
                    try:
                        st.session_state.vector_store.delete_collection()
                    except Exception:
                        pass
                st.session_state.rag_chain = None
                st.session_state.vector_store = None
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
                for key in ['last_context_docs', 'page_number', 'page_source', 'last_question']:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("---")
        st.markdown("### 모델 정보")
        st.markdown("- **LLM:** gpt-5-nano")
        st.markdown("- **Embedding:** text-embedding-3-small")
        st.markdown("---")
        st.markdown("### 사용 방법")
        st.markdown("1. PDF 파일 업로드 (여러 개 가능)")
        st.markdown("2. 'PDF 문서 저장' 클릭")
        st.markdown("3. 질문 입력")

    st.header("Multi-PDF RAG")

    tab_chat, tab_chunks = st.tabs(["채팅", "청크 미리보기"])

    with tab_chat:
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
                        response = st.session_state.rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        sources = response.get("context", [])
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
                    st.markdown(f"**{source} — 페이지 {page}**")

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
        if st.session_state.vector_store:
            try:
                collection = st.session_state.vector_store._collection
                result = collection.get(include=["documents", "metadatas"])
                docs = result.get("documents", [])
                metas = result.get("metadatas", [])

                st.info(f"총 **{len(docs)}**개 청크가 저장되어 있습니다.")

                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    source = meta.get("source_file", "Unknown")
                    page = meta.get("page", "?")
                    page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                    with st.expander(f"청크 #{i+1} — {source} (p.{page_display})"):
                        st.text(doc)
            except Exception as e:
                st.error(f"청크 조회 실패: {e}")
        else:
            st.warning("벡터 저장소가 비어 있습니다. PDF를 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
