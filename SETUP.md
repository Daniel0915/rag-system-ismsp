# 환경 설정 가이드 (Windows)

강의장 PC에 처음 세팅할 때 따라하세요.

---

## 1단계: Python 설치

> **중요**: Python **3.12** 또는 **3.13**을 설치하세요. (3.14는 chromadb 미지원)

1. https://www.python.org/downloads/ 접속
2. **Python 3.13.x** 다운로드 (Windows installer 64-bit)
3. 설치 시 **반드시** `Add python.exe to PATH` 체크 ✅
4. Install Now 클릭

설치 확인:
```
python --version
```
`Python 3.13.x`가 나오면 성공

---

## 2단계: uv 설치 (패키지 관리자)

pip보다 빠른 패키지 관리자입니다.

PowerShell을 **관리자 권한**으로 실행 후:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 확인:
```
uv --version
```

> uv 설치가 안 되면 pip으로 대체 가능합니다 (3단계 참고)

---

## 3단계: 프로젝트 세팅

### 3-1. 프로젝트 폴더로 이동
```
cd rag-system-example
```

### 3-2. 가상 환경 생성
```
uv venv --python 3.13
```

### 3-3. 가상 환경 활성화
```
.venv\Scripts\activate
```

프롬프트 앞에 `(.venv)`가 보이면 성공

### 3-4. 패키지 설치
```
uv pip install -r requirements.txt
```

> **uv 없이 pip으로 설치하는 경우:**
> ```
> python -m venv .venv
> .venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

## 4단계: OpenAI API 키 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```
OPENAI_API_KEY=sk-여기에-API-키-입력
```

> 메모장으로 만들 경우: 파일명을 `.env`로 저장 (확장자 `.txt`가 붙지 않도록 주의)

---

## 5단계: 실행 확인

```
cd 00-openai-basics
streamlit run step1_api_call.py
```

브라우저에서 http://localhost:8501 이 열리면 세팅 완료!

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| `python`이 인식 안 됨 | Python 설치 시 `Add to PATH` 체크 안 한 경우 → 재설치 |
| `uv`가 인식 안 됨 | 터미널을 닫고 새로 열어보기. 안 되면 pip 사용 |
| `streamlit`이 인식 안 됨 | 가상 환경 활성화 확인 (`(.venv)` 표시) |
| `ModuleNotFoundError` | `uv pip install -r requirements.txt` 다시 실행 |
| `OPENAI_API_KEY` 오류 | `.env` 파일 경로 확인 (프로젝트 루트에 있어야 함) |
| chromadb 설치 오류 | Python 3.14 사용 중일 수 있음 → 3.12 또는 3.13으로 변경 |
