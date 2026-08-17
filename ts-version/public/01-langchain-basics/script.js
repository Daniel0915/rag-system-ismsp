const API = "/api/01-langchain-basics";

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "요청 실패");
  return data;
}

function withLoading(button, resultEl, fn) {
  return async () => {
    button.disabled = true;
    resultEl.textContent = "생각 중...";
    try {
      await fn();
    } catch (err) {
      resultEl.textContent = `오류: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  };
}

const s1Btn = document.getElementById("s1-btn");
const s1Result = document.getElementById("s1-result");
s1Btn.addEventListener(
  "click",
  withLoading(s1Btn, s1Result, async () => {
    const question = document.getElementById("s1-question").value;
    const { answer } = await postJSON(`${API}/step1`, { question });
    s1Result.textContent = answer;
  })
);

const s2Btn = document.getElementById("s2-btn");
const s2Result = document.getElementById("s2-result");
s2Btn.addEventListener(
  "click",
  withLoading(s2Btn, s2Result, async () => {
    const text = document.getElementById("s2-text").value;
    const length = Number(document.getElementById("s2-length").value);
    const { answer } = await postJSON(`${API}/step2`, { text, length });
    s2Result.textContent = answer;
  })
);

const s3LangSelect = document.getElementById("s3-lang");
fetch(`${API}/step3/options`)
  .then((r) => r.json())
  .then(({ languages }) => {
    for (const lang of languages) {
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang;
      s3LangSelect.appendChild(opt);
    }
  });

const s3Result = document.getElementById("s3-result");
const s3TranslateBtn = document.getElementById("s3-translate-btn");
s3TranslateBtn.addEventListener(
  "click",
  withLoading(s3TranslateBtn, s3Result, async () => {
    const text = document.getElementById("s3-text").value;
    const targetLanguage = s3LangSelect.value;
    const { answer } = await postJSON(`${API}/step3/translate`, { text, targetLanguage });
    s3Result.textContent = answer;
  })
);

const s3ClassifyBtn = document.getElementById("s3-classify-btn");
s3ClassifyBtn.addEventListener(
  "click",
  withLoading(s3ClassifyBtn, s3Result, async () => {
    const text = document.getElementById("s3-text").value;
    const { answer } = await postJSON(`${API}/step3/classify`, { text });
    s3Result.textContent = answer;
  })
);
