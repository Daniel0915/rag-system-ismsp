"""RAG 시스템 Streamlit UI"""

import streamlit as st
import os
import re
import base64
import fitz
from typing import List
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = str(BASE_DIR / "chroma_db")

SYSTEM_PROMPT = """당신은 도움이 되는 AI 어시스턴트입니다.
제공된 컨텍스트를 바탕으로 사용자의 질문에 친절하고 상세하게 답변하세요.

규칙:
- 컨텍스트에 있는 정보를 최대한 활용하여 답변하세요
- 컨텍스트의 정보가 부족하더라도 관련 내용은 최대한 설명하세요
- 출처 문서의 파일명을 언급하세요
- 한국어로 답변하세요
- 명확하고 구조적으로 답변하세요"""


def load_single_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    ext = path.suffix.lower()

    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        return load_image(file_path)

    loaders = {
        '.pdf': lambda: PyPDFLoader(file_path),
        '.csv': lambda: CSVLoader(file_path),
        '.md': lambda: UnstructuredMarkdownLoader(file_path),
        '.markdown': lambda: UnstructuredMarkdownLoader(file_path),
        '.docx': lambda: UnstructuredWordDocumentLoader(file_path),
        '.doc': lambda: UnstructuredWordDocumentLoader(file_path),
        '.pptx': lambda: UnstructuredPowerPointLoader(file_path),
        '.ppt': lambda: UnstructuredPowerPointLoader(file_path),
    }

    loader_fn = loaders.get(ext, lambda: TextLoader(file_path, encoding='utf-8'))
    docs = loader_fn().load()
    for doc in docs:
        doc.metadata['filename'] = path.name
        doc.metadata['file_type'] = ext
        doc.metadata['file_size'] = path.stat().st_size
    return docs


def load_image(image_path):
    path = Path(image_path)
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = path.suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'image/png')

        vision_llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
        message = HumanMessage(content=[
            {"type": "text", "text": (
                "이 이미지의 내용을 상세하게 한국어로 설명해주세요. "
                "텍스트가 있다면 모두 추출하고, 표/차트/그래프가 있다면 내용을 설명하고, "
                "사진이라면 무엇이 보이는지 설명해주세요. "
                "가능한 모든 정보를 빠짐없이 포함해주세요."
            )},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }}
        ])

        response = vision_llm.invoke([message])
        content = response.content

        return [Document(
            page_content=f"[이미지: {path.name}]\n{content}",
            metadata={
                'filename': path.name,
                'file_type': ext,
                'file_size': path.stat().st_size,
                'source': str(image_path),
                'ocr_engine': 'gpt-vision'
            }
        )]
    except Exception as e:
        return [Document(
            page_content=f"[이미지: {path.name}]\n[오류] 이미지 텍스트 추출 실패: {str(e)}",
            metadata={
                'filename': path.name,
                'file_type': path.suffix.lower(),
                'file_size': path.stat().st_size,
                'error': str(e),
                'ocr_engine': 'gpt-vision'
            }
        )]


def load_and_chunk_documents(file_paths, chunk_size=1000, chunk_overlap=200):
    all_docs = []
    for file_path in file_paths:
        try:
            docs = load_single_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            path = Path(file_path)
            all_docs.append(Document(
                page_content=f"[오류] 파일을 로드할 수 없습니다: {str(e)}",
                metadata={
                    'filename': path.name,
                    'file_type': path.suffix.lower(),
                    'error': str(e)
                }
            ))

    if not all_docs:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n", "\n\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)

    filtered = []
    merge_buffer = ""
    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) == 0:
            continue
        if len(content) < 50:
            merge_buffer += "\n\n" + content
        else:
            if merge_buffer:
                chunk.page_content = merge_buffer + "\n\n" + chunk.page_content
                merge_buffer = ""
            filtered.append(chunk)
    if merge_buffer and filtered:
        filtered[-1].page_content += "\n\n" + merge_buffer
    chunks = filtered

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({'chunk_index': i, 'total_chunks': len(chunks)})

    return chunks


def save_to_vector_store(chunks, vectorstore):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vectorstore.add_documents(chunks)

    return vectorstore


def create_rag_chain(vectorstore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """다음 컨텍스트를 바탕으로 질문에 답변해주세요:

<context>
{context}
</context>

질문: {input}""")
    ])

    llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


def get_all_documents(vectorstore):
    if vectorstore is None:
        return []
    try:
        collection = vectorstore._collection
        results = collection.get()
        if not results['metadatas']:
            return []
        file_map = {}
        for metadata in results['metadatas']:
            filename = metadata.get('filename', 'unknown')
            if filename not in file_map:
                file_map[filename] = {
                    'filename': filename,
                    'file_type': metadata.get('file_type', ''),
                    'file_size': metadata.get('file_size', 0),
                    'chunks': 0
                }
            file_map[filename]['chunks'] += 1
        return list(file_map.values())
    except Exception:
        return []


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
    st.set_page_config("06-rag-multi-format", layout="wide")

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_info' not in st.session_state:
        st.session_state.doc_info = None
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = {}

    if st.session_state.vectorstore is None and Path(PERSIST_DIR).exists():
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            st.session_state.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="documents",
                collection_metadata={"hnsw:space": "cosine"}
            )
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)

            docs = get_all_documents(st.session_state.vectorstore)
            if docs:
                st.session_state.doc_info = {
                    'files': [d['filename'] for d in docs],
                    'count': len(docs),
                    'chunks': sum(d['chunks'] for d in docs)
                }
        except Exception:
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None

    with st.sidebar:
        st.header("설정")
        st.markdown("---")

        st.subheader("문서")
        uploaded_files = st.file_uploader(
            "파일을 업로드 해주세요",
            type=['pdf', 'csv', 'txt', 'md', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'docx', 'pptx'],
            accept_multiple_files=True,
            help="지원 형식: PDF, CSV, TXT, MD, 이미지, DOCX, PPTX"
        )
        upload_button = st.button("문서 저장")

        if uploaded_files and upload_button:
            with st.spinner("문서를 저장하고 있습니다..."):
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)

                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    file_paths.append(str(file_path))

                chunks = load_and_chunk_documents(file_paths)
                if chunks:
                    st.session_state.vectorstore = save_to_vector_store(
                        chunks, st.session_state.vectorstore
                    )
                    st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)

                    file_results = {}
                    for chunk in chunks:
                        filename = chunk.metadata.get('filename', 'unknown')
                        if filename not in file_results:
                            file_results[filename] = 0
                        file_results[filename] += 1

                    success_count = len(file_results)
                    total_chunks = sum(file_results.values())
                    st.success(f"{success_count}개 문서가 저장되었습니다! (총 {total_chunks}개 청크)")

            with st.spinner("PDF 페이지를 이미지로 변환하는 중입니다..."):
                for file_path in file_paths:
                    if file_path.lower().endswith('.pdf'):
                        basename = Path(file_path).name
                        images = convert_pdf_to_images(file_path)
                        st.session_state.pdf_images[basename] = images

            docs = get_all_documents(st.session_state.vectorstore)
            if docs:
                st.session_state.doc_info = {
                    'files': [d['filename'] for d in docs],
                    'count': len(docs),
                    'chunks': sum(d['chunks'] for d in docs)
                }

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
                st.session_state.rag_chain = None
                gc.collect()
                try:
                    if os.path.exists(PERSIST_DIR):
                        shutil.rmtree(PERSIST_DIR)
                except Exception:
                    pass
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
        st.markdown("1. 파일 업로드 (PDF, CSV, TXT 등)")
        st.markdown("2. '문서 저장' 클릭")
        st.markdown("3. 질문 입력")

    st.header("RAG 시스템")

    has_documents = st.session_state.vectorstore and get_all_documents(st.session_state.vectorstore)

    tab_chat, tab_chunks = st.tabs(["채팅", "청크 미리보기"])

    with tab_chat:
        left_column, right_column = st.columns([1, 1])

        with left_column:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input(
                "질문을 입력해주세요",
                disabled=not has_documents
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
                        context_docs = response.get("context", [])
                    st.markdown(answer)

                st.session_state.last_context_docs = context_docs
                st.session_state.messages.append({"role": "assistant", "content": answer})

            if st.session_state.get("last_context_docs"):
                if not prompt:
                    st.markdown("---")
                st.markdown("### 관련 문서")
                for idx, document in enumerate(st.session_state.last_context_docs):
                    filename = document.metadata.get('filename', 'Unknown')
                    file_type = document.metadata.get('file_type', '')
                    page = document.metadata.get('page', None)

                    label = f"**{filename}**"
                    if page is not None:
                        page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                        label += f" — 페이지 {page_display}"
                    st.markdown(label)

                    st.text_area(
                        f"청크 {idx + 1}",
                        value=document.page_content,
                        height=150,
                        key=f"chunk_{idx}_{st.session_state.get('last_question', '')}"
                    )

                    if file_type == 'pdf' and page is not None:
                        page_num = int(page) + 1 if isinstance(page, (int, float)) else 1
                        if st.button(f"PDF {page_num}페이지 보기", key=f"btn_{page_num}_{idx}"):
                            st.session_state.page_number = str(page_num)
                            st.session_state.page_source = filename
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
                    filename = meta.get("filename", "Unknown")
                    file_type = meta.get("file_type", "")
                    page = meta.get("page", None)

                    label = f"청크 #{i+1} — {filename}"
                    if page is not None:
                        page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                        label += f" (p.{page_display})"
                    if file_type:
                        label += f" [{file_type}]"

                    with st.expander(label):
                        st.text(doc)
            except Exception as e:
                st.error(f"청크 조회 실패: {e}")
        else:
            st.warning("벡터 저장소가 비어 있습니다. 문서를 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
