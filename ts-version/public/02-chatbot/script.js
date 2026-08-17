const API = "/api/02-chatbot";

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "요청 실패");
  return data;
}

function renderLog(logEl, history) {
  logEl.innerHTML = "";
  for (const turn of history) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${turn.role === "user" ? "user" : "assistant"}`;
    bubble.textContent = turn.content;
    logEl.appendChild(bubble);
  }
  logEl.scrollTop = logEl.scrollHeight;
}

// Step 1
const s1Btn = document.getElementById("s1-btn");
const s1Result = document.getElementById("s1-result");
s1Btn.addEventListener("click", async () => {
  s1Btn.disabled = true;
  s1Result.textContent = "생각 중...";
  try {
    const name = document.getElementById("s1-name").value;
    const topic = document.getElementById("s1-topic").value;
    const level = document.getElementById("s1-level").value;
    const { answer } = await postJSON(`${API}/step1`, { name, topic, level });
    s1Result.textContent = answer;
  } catch (err) {
    s1Result.textContent = `오류: ${err.message}`;
  } finally {
    s1Btn.disabled = false;
  }
});

// Step 2
const s2Log = document.getElementById("s2-log");
const s2Input = document.getElementById("s2-input");
const s2Send = document.getElementById("s2-send");
const s2Reset = document.getElementById("s2-reset");

async function loadStep2History() {
  const res = await fetch(`${API}/step2/history`, { credentials: "same-origin" });
  const { history } = await res.json();
  renderLog(s2Log, history);
}
loadStep2History();

async function sendStep2() {
  const message = s2Input.value.trim();
  if (!message) return;
  s2Send.disabled = true;
  s2Input.value = "";
  try {
    const { history } = await postJSON(`${API}/step2/message`, { message });
    renderLog(s2Log, history);
  } catch (err) {
    alert(`오류: ${err.message}`);
  } finally {
    s2Send.disabled = false;
  }
}
s2Send.addEventListener("click", sendStep2);
s2Input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendStep2();
});
s2Reset.addEventListener("click", async () => {
  await postJSON(`${API}/step2/reset`, {});
  renderLog(s2Log, []);
});

// Step 3
const s3PersonaSelect = document.getElementById("s3-persona");
const s3CustomWrap = document.getElementById("s3-custom-wrap");
const s3CustomPrompt = document.getElementById("s3-custom-prompt");
const s3Log = document.getElementById("s3-log");
const s3Input = document.getElementById("s3-input");
const s3Send = document.getElementById("s3-send");
let s3ActiveCustomPrompt = "";

fetch(`${API}/step3/personas`)
  .then((r) => r.json())
  .then(async (personas) => {
    for (const [key, { label }] of Object.entries(personas)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      s3PersonaSelect.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "직접 입력";
    s3PersonaSelect.appendChild(customOpt);
    await selectPersona(s3PersonaSelect.value);
  });

async function selectPersona(personaKey) {
  s3CustomWrap.style.display = personaKey === "custom" ? "block" : "none";
  const { greeting } = await postJSON(`${API}/step3/select-persona`, {
    personaKey,
    customSystemPrompt: s3CustomPrompt.value,
  });
  renderLog(s3Log, greeting ? [{ role: "assistant", content: greeting }] : []);
}

s3PersonaSelect.addEventListener("change", () => selectPersona(s3PersonaSelect.value));
document.getElementById("s3-apply-custom").addEventListener("click", () => selectPersona("custom"));

async function sendStep3() {
  const message = s3Input.value.trim();
  if (!message) return;
  s3Send.disabled = true;
  s3Input.value = "";
  try {
    const { history } = await postJSON(`${API}/step3/message`, {
      message,
      customSystemPrompt: s3CustomPrompt.value,
    });
    renderLog(s3Log, history);
  } catch (err) {
    alert(`오류: ${err.message}`);
  } finally {
    s3Send.disabled = false;
  }
}
s3Send.addEventListener("click", sendStep3);
s3Input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendStep3();
});
