# RAG 시스템 (TypeScript + Gemini/Ollama)

원래 Python/LangChain/OpenAI/Streamlit으로 만들어졌던 RAG 학습 커리큘럼(00~07)을 **TypeScript + Express**로 새로 만든 버전입니다. LLM은 **Google Gemini(클라우드)** 또는 **Ollama(로컬, 예: Qwen)** 중 하나를 환경변수 하나로 선택할 수 있습니다. (Python 버전은 더 이상 이 저장소에 없습니다 — TypeScript가 메인입니다.)

## 실행 방법

```bash
npm install
cp .env.example .env
npm run dev
```

브라우저에서 `http://localhost:3000` 접속 → 00~07 모듈 링크로 이동.

- `npm run dev` — 개발 모드(파일 변경 자동 반영)
- `npm run build` / `npm start` — 프로덕션 빌드 후 실행
- `npm run typecheck` — 타입 체크만 실행

### Gemini로 실행 (기본값)

`.env`에서 `LLM_PROVIDER=gemini`로 두고, `GOOGLE_API_KEY`를 [Google AI Studio](https://aistudio.google.com/apikey)에서 발급받아 입력하세요.

### Ollama(로컬 Qwen)로 실행

```bash
brew install ollama          # 또는 https://ollama.com 에서 설치
ollama serve                  # 로컬 서버 (백그라운드로 계속 켜둬야 함)
ollama pull qwen2.5:3b         # 채팅 모델 (~1.9GB)
ollama pull nomic-embed-text   # 임베딩 모델 (~274MB)
```

`.env`에서 `LLM_PROVIDER=ollama`로 바꾸면 됩니다 (`GOOGLE_API_KEY` 불필요). 나머지 코드는 전혀 손댈 필요 없이 동일하게 동작합니다 — `src/llm/provider.ts`가 `ChatGoogleGenerativeAI`/`ChatOllama` 중 어느 쪽을 반환하든 둘 다 LangChain의 동일한 `BaseChatModel` 인터페이스(`.invoke()`, `.pipe()`)를 구현하기 때문입니다.

**주의**:
- `OLLAMA_BASE_URL`은 `localhost`가 아니라 `127.0.0.1`을 씁니다 — Node의 `fetch`가 `localhost`를 IPv6(`::1`)로 먼저 시도하는데 Ollama는 기본적으로 IPv4에만 바인딩되어 연결이 거부되기 때문입니다.
- 06(멀티포맷 RAG)의 이미지 OCR은 텍스트 전용 모델(Qwen2.5)로는 처리할 수 없어, `LLM_PROVIDER=ollama`일 때 이미지 업로드 시 에러를 반환합니다. 이미지가 필요하면 Gemini로 전환하거나 Ollama에서 `qwen2.5vl`/`llava` 같은 비전 모델을 pull 받아 `OLLAMA_CHAT_MODEL`을 바꾸세요.
- 구조화된 JSON 출력(00/step3, 07 evaluator 등)은 `qwen2.5:3b`에서 LangChain의 기본 `withStructuredOutput`(tool-calling/jsonSchema 그래머)이 신뢰할 수 없어서(모델이 이상한 tool-call 형태로 응답), `src/llm/provider.ts`의 `generateStructured()`가 provider별로 분기합니다: Ollama에서는 Ollama 자체 `format: "json"` 모드(자유형 JSON 강제) + 스키마의 `.describe()`로 만든 필드 힌트 + zod 검증으로 처리합니다. 실제 테스트로 정상 동작 확인함.
- 메모리/디스크가 빠듯한 환경(예: 8GB RAM)에서는 `qwen2.5:3b`처럼 작은 모델을 권장합니다. `qwen2.5:7b` 이상은 다른 앱을 다 끄고 써야 할 정도로 무거울 수 있어요.

## 원래 Python 버전과의 대응 관계 (참고용)

| Python (제거됨) | TypeScript |
|---|---|
| `ChatOpenAI` | `ChatGoogleGenerativeAI` 또는 `ChatOllama` (`LLM_PROVIDER`로 선택) |
| `OpenAIEmbeddings` | `GoogleGenerativeAIEmbeddings` 또는 `OllamaEmbeddings` |
| Streamlit UI | Express + 정적 HTML/JS (모듈당 1페이지, step별 섹션) |
| `st.session_state` (대화 이력) | `express-session` (기본 MemoryStore) |
| Chroma (`persist_directory`) | `HNSWLib` + JSON 사이드카 (`data/vectorstore/<module>/`), add/delete 시 인덱스 재구축 |
| PyMuPDF 페이지 래스터화 | `mupdf`(Artifex 공식 WASM 바인딩, PyMuPDF와 동일 엔진) |
| `unstructured` (pptx 등) | `officeparser` (텍스트만 추출하는 실용적 근사치) |
| GPT Vision 기반 이미지 OCR | Gemini 자체 멀티모달 (별도 비전 모델 불필요) |

## 알아둘 점

- **무료 티어 API 할당량**: Gemini API 무료 티어는 **모델당 하루 20회 요청**으로 제한됩니다(예: `generativelanguage.googleapis.com/generate_content_free_tier_requests`). 07의 step3/step4처럼 한 번의 실행이 내부적으로 LLM을 여러 번 호출하는 경우 금방 소진될 수 있습니다. 결제를 활성화하면 한도가 크게 늘어납니다.
- **모델명**: `.env`의 `GEMINI_CHAT_MODEL`/`GEMINI_EMBEDDING_MODEL`은 `-latest` 별칭(`gemini-flash-latest`, `gemini-embedding-001`)을 기본값으로 사용합니다. Google이 구체적인 버전 스냅샷(예: `gemini-2.5-flash`)을 신규 사용자에게 더 이상 제공하지 않는 경우가 있어, 별칭을 쓰는 편이 안전합니다.
- **07/step4 재검색 루프**: 원본은 k=2로 재검색을 유도했지만, Gemini 임베딩 공간에서는 k=1로 좁혀야 "NovaCore → Qubrix → 매출" 2-hop 재검색이 안정적으로 재현됩니다.
- **pptx**: 전용 LangChain.js 로더가 없어 `officeparser`로 텍스트만 추출합니다(레이아웃/이미지 정보 없음).
- **PDF 페이지 미리보기**: 첫 렌더링 시 `data/pdf-pages/`에 PNG로 캐싱됩니다.

## 검증 상태

- `npm run typecheck` 통과
- **Gemini**로 확인: 00(채팅/구조화 출력), 01, 03(PDF 업로드→청킹→임베딩→저장→채팅→출처→페이지 이미지), 07/step1 — Gemini 무료 티어의 모델당 하루 20회 요청 제한으로 나머지는 curl로 개별 확인은 못 했지만 동일한 공용 로직을 재사용하는 구조입니다.
- **Ollama(qwen2.5:3b + nomic-embed-text)**로 확인: 00(채팅/구조화 출력), 03(임베딩 유사도, PDF RAG 채팅+출처), 07(step1 선형 그래프, step2 조건 분기) — 로컬이라 할당량 제약 없이 실제 호출로 검증함.
