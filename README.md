# RAG 기반 생성형 AI 애플리케이션 구현

OpenAI · LangChain · ChromaDB를 활용한 RAG 시스템 실습 예제 모음입니다.

## 실습 구성

| 폴더 | 내용 |
|------|------|
| `00-openai-basics` | OpenAI API 기초 |
| `01-langchain-basics` | LangChain 기초 |
| `02-chatbot` | 챗봇 구현 |
| `03-single-pdf-rag` | 단일 PDF 기반 RAG |
| `04-multi-pdf-rag` | 다중 PDF 기반 RAG |
| `05-metadata-filtered-rag` | 메타데이터 필터링 RAG |
| `06-rag-multi-format` | 다양한 포맷 문서 RAG |
| `07-langgraph-basics` | LangGraph 기초 |

## 환경 설정 (uv)

### 1. uv 설치 (pip 사용)

```powershell
pip install uv
```

> 또는 Windows 설치 스크립트:
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

### 2. 프로젝트 초기화

`pyproject.toml`이 없을 때만 실행합니다. (이미 있으면 건너뛰기)

```powershell
uv init
```

### 3. 패키지 설치

```powershell
uv add -r requirements.txt
```

`requirements.txt`의 패키지가 `pyproject.toml`에 의존성으로 추가되고, `.venv` 가상환경이 자동 생성됩니다.

> 이미 `pyproject.toml`이 구성되어 있다면 아래 한 줄로 동기화할 수 있습니다:
> ```powershell
> uv sync
> ```

### 4. OpenAI API 키 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```
OPENAI_API_KEY=sk-여기에-API-키-입력
```

### 5. 실행

**방법 A — `uv run` 사용 (가상환경 activate 불필요, 추천)**

```powershell
cd 00-openai-basics
uv run streamlit run step1_api_call.py
```

**방법 B — 가상환경 activate 후 실행**

```powershell
.venv\Scripts\activate              # 가상환경 활성화 (프롬프트 앞에 (.venv) 표시)
cd 00-openai-basics
streamlit run step1_api_call.py
```

> `uv run`은 자동으로 `.venv`를 찾아 실행하므로 activate가 필요 없습니다.
> activate 방식은 한 번 활성화하면 해당 터미널에서 계속 유지되지만, 새 터미널을 열면 다시 활성화해야 합니다.

브라우저에서 http://localhost:8501 이 열리면 완료입니다.

---

> Windows 강의장 PC 초기 세팅 등 더 자세한 안내는 [SETUP.md](SETUP.md)를 참고하세요.
