const API = "/api/07-langgraph-basics";

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
    resultEl.textContent = "실행 중...";
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
    const data = await postJSON(`${API}/step1`, { question });
    s1Result.textContent = `[분석]\n${data.analysis}\n\n[답변]\n${data.answer}`;
  })
);

const s2Btn = document.getElementById("s2-btn");
const s2Result = document.getElementById("s2-result");
s2Btn.addEventListener(
  "click",
  withLoading(s2Btn, s2Result, async () => {
    const question = document.getElementById("s2-question").value;
    const data = await postJSON(`${API}/step2`, { question });
    s2Result.textContent =
      `[분류] ${data.questionType}${data.rerouted ? " (재라우팅됨)" : ""}\n\n[답변]\n${data.answer}`;
  })
);

const s3Btn = document.getElementById("s3-btn");
const s3Result = document.getElementById("s3-result");
s3Btn.addEventListener(
  "click",
  withLoading(s3Btn, s3Result, async () => {
    const topic = document.getElementById("s3-topic").value;
    const maxIterations = Number(document.getElementById("s3-max").value);
    const { history } = await postJSON(`${API}/step3`, { topic, maxIterations });
    s3Result.textContent = history
      .map((h) => `--- 반복 ${h.iteration} (점수: ${h.score}) ---\n${h.draft}\n피드백: ${h.feedback}`)
      .join("\n\n");
  })
);

const s4Btn = document.getElementById("s4-btn");
const s4Result = document.getElementById("s4-result");
s4Btn.addEventListener(
  "click",
  withLoading(s4Btn, s4Result, async () => {
    const question = document.getElementById("s4-question").value;
    const maxIterations = Number(document.getElementById("s4-max").value);
    const data = await postJSON(`${API}/step4`, { question, maxIterations });
    s4Result.textContent =
      `[검색 반복 횟수] ${data.iteration}\n[완료 여부] ${data.isComplete}\n\n` +
      `[최종 검색어] ${data.searchQuery}\n\n[답변]\n${data.answer}\n\n[평가 이유]\n${data.evaluationReason}`;
  })
);
