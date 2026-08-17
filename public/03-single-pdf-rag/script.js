const API = "/api/03-single-pdf-rag";
let currentFilename = null;

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

const uploadBtn = document.getElementById("upload-btn");
const uploadResult = document.getElementById("upload-result");
uploadBtn.addEventListener("click", async () => {
  const file = document.getElementById("file-input").files[0];
  if (!file) return alert("파일을 선택하세요.");
  uploadBtn.disabled = true;
  uploadResult.textContent = "처리 중...";
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    currentFilename = data.filename;
    uploadResult.textContent = `파일: ${data.filename}\n크기: ${data.sizeBytes} bytes\n페이지: ${data.pages}\n청크: ${data.chunks}`;
  } catch (err) {
    uploadResult.textContent = `오류: ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

document.getElementById("delete-btn").addEventListener("click", async () => {
  await postJSON(`${API}/delete-all`, {});
  uploadResult.textContent = "모든 문서가 삭제되었습니다.";
  currentFilename = null;
});

const simBtn = document.getElementById("sim-btn");
const simResult = document.getElementById("sim-result");
simBtn.addEventListener("click", async () => {
  simBtn.disabled = true;
  try {
    const a = document.getElementById("sim-a").value;
    const b = document.getElementById("sim-b").value;
    const { similarity } = await postJSON(`${API}/similarity`, { a, b });
    simResult.textContent = `코사인 유사도: ${similarity.toFixed(4)}`;
  } catch (err) {
    simResult.textContent = `오류: ${err.message}`;
  } finally {
    simBtn.disabled = false;
  }
});

const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const sourcesEl = document.getElementById("sources");
const pageImage = document.getElementById("page-image");

function appendBubble(role, content) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = content;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendChat() {
  const question = chatInput.value.trim();
  if (!question) return;
  chatSend.disabled = true;
  appendBubble("user", question);
  chatInput.value = "";
  try {
    const { answer, sources } = await postJSON(`${API}/chat`, { question });
    appendBubble("assistant", answer);
    sourcesEl.innerHTML = "";
    for (const s of sources) {
      const div = document.createElement("div");
      div.className = "source-item";
      div.textContent = `[${s.index}] 페이지 ${s.page} — ${s.preview}...`;
      div.addEventListener("click", () => showPage(s.filename, s.page));
      sourcesEl.appendChild(div);
    }
  } catch (err) {
    appendBubble("assistant", `오류: ${err.message}`);
  } finally {
    chatSend.disabled = false;
  }
}

function showPage(filename, page) {
  pageImage.src = `${API}/page-image?filename=${encodeURIComponent(filename)}&page=${page}`;
  pageImage.style.display = "block";
}

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});

document.getElementById("load-chunks-btn").addEventListener("click", async () => {
  const res = await fetch(`${API}/chunks`);
  const { chunks } = await res.json();
  document.getElementById("chunks-result").textContent = chunks
    .map((c, i) => `[${i + 1}] (페이지 ${c.metadata.page}) ${c.pageContent.slice(0, 150)}...`)
    .join("\n\n");
});
