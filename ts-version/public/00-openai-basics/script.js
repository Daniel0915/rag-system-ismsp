const API = "/api/00-openai-basics";

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

// Step 1
const s1Btn = document.getElementById("s1-btn");
const s1Result = document.getElementById("s1-result");
s1Btn.addEventListener(
  "click",
  withLoading(s1Btn, s1Result, async () => {
    const question = document.getElementById("s1-question").value;
    const temperature = Number(document.getElementById("s1-temp").value);
    const { answer } = await postJSON(`${API}/step1`, { question, temperature });
    s1Result.textContent = answer;
  })
);

// Step 2
const s2RoleSelect = document.getElementById("s2-role");
const s2CustomWrap = document.getElementById("s2-custom-wrap");
fetch(`${API}/step2/roles`)
  .then((r) => r.json())
  .then((roles) => {
    for (const [key, { label }] of Object.entries(roles)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      s2RoleSelect.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "직접 입력";
    s2RoleSelect.appendChild(customOpt);
  });
s2RoleSelect.addEventListener("change", () => {
  s2CustomWrap.style.display = s2RoleSelect.value === "custom" ? "block" : "none";
});

const s2Btn = document.getElementById("s2-btn");
const s2Result = document.getElementById("s2-result");
s2Btn.addEventListener(
  "click",
  withLoading(s2Btn, s2Result, async () => {
    const roleKey = s2RoleSelect.value;
    const input = document.getElementById("s2-input").value;
    const customSystemPrompt = document.getElementById("s2-custom").value;
    const { answer } = await postJSON(`${API}/step2`, { roleKey, input, customSystemPrompt });
    s2Result.textContent = answer;
  })
);

// Step 3
const s3Btn = document.getElementById("s3-btn");
const s3Result = document.getElementById("s3-result");
s3Btn.addEventListener(
  "click",
  withLoading(s3Btn, s3Result, async () => {
    const text = document.getElementById("s3-input").value;
    const analysis = await postJSON(`${API}/step3`, { text });
    s3Result.textContent = JSON.stringify(analysis, null, 2);
  })
);
