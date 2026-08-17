const API = "/api/04-multi-pdf-rag";

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
  const files = document.getElementById("file-input").files;
  if (!files.length) return alert("파일을 선택하세요.");
  uploadBtn.disabled = true;
  uploadResult.textContent = "처리 중...";
  try {
    const formData = new FormData();
    for (const f of files) formData.append("files", f);
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    uploadResult.textContent = data.results
      .map((r) => `${r.filename}: ${r.status} (${r.chunks} 청크)`)
      .join("\n");
    await loadFiles();
  } catch (err) {
    uploadResult.textContent = `오류: ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

const fileListEl = document.getElementById("file-list");
async function loadFiles() {
  const res = await fetch(`${API}/files`);
  const { files } = await res.json();
  fileListEl.innerHTML = "";
  for (const f of files) {
    const row = document.createElement("div");
    row.className = "source-item";
    row.style.display = "flex";
    row.style.justifyContent = "space-between";
    row.style.alignItems = "center";
    row.innerHTML = `<span>${f.sourceFile} — 청크 ${f.chunkCount}개</span>`;
    const delBtn = document.createElement("button");
    delBtn.textContent = "삭제";
    delBtn.style.marginTop = "0";
    delBtn.style.background = "#a33";
    delBtn.addEventListener("click", async () => {
      await postJSON(`${API}/delete-file`, { sourceFile: f.sourceFile });
      await loadFiles();
    });
    row.appendChild(delBtn);
    fileListEl.appendChild(row);
  }
}
document.getElementById("refresh-files-btn").addEventListener("click", loadFiles);
document.getElementById("delete-all-btn").addEventListener("click", async () => {
  await postJSON(`${API}/delete-all`, {});
  await loadFiles();
});
loadFiles();

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
      div.textContent = `${s.sourceFile} — 페이지 ${s.page}: ${s.preview}...`;
      div.addEventListener("click", () => showPage(s.sourceFile, s.page));
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
