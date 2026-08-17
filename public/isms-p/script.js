const API = "/api/isms-p";

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

function fillSelect(select, options, includeEmpty = false) {
  if (includeEmpty) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "전체";
    select.appendChild(opt);
  }
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = typeof o === "string" ? o : o.value;
    opt.textContent = typeof o === "string" ? o : o.label;
    select.appendChild(opt);
  }
}

const docTypeSelect = document.getElementById("meta-doc-type");
const domainSelect = document.getElementById("meta-domain");
const filterDocType = document.getElementById("filter-doc-type");
const filterDomain = document.getElementById("filter-domain");

fetch(`${API}/metadata-options`)
  .then((r) => r.json())
  .then(({ docTypes, domains }) => {
    fillSelect(docTypeSelect, docTypes);
    fillSelect(domainSelect, domains);
    fillSelect(filterDocType, docTypes, true);
    fillSelect(filterDomain, domains, true);
  });

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
    formData.append("doc_type", docTypeSelect.value);
    formData.append("domain", domainSelect.value);
    formData.append("year", document.getElementById("meta-year").value);
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    uploadResult.textContent = `${data.filename}: ${data.status} (${data.chunks} 청크)`;
  } catch (err) {
    uploadResult.textContent = `오류: ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
  }
});

document.getElementById("delete-btn").addEventListener("click", async () => {
  await postJSON(`${API}/delete-all`, {});
  uploadResult.textContent = "모든 문서가 삭제되었습니다.";
});

const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const sourcesEl = document.getElementById("sources");

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
    const filter = { doc_type: filterDocType.value, domain: filterDomain.value };
    const { answer, sources } = await postJSON(`${API}/chat`, { question, filter });
    appendBubble("assistant", answer);
    sourcesEl.innerHTML = "";
    for (const s of sources) {
      const div = document.createElement("div");
      div.className = "source-item";
      div.textContent = `[${s.doc_type ?? "-"} / ${s.chunk_strategy ?? "-"}] ${s.sourceFile}`;
      sourcesEl.appendChild(div);
    }
  } catch (err) {
    appendBubble("assistant", `오류: ${err.message}`);
  } finally {
    chatSend.disabled = false;
  }
}

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});
