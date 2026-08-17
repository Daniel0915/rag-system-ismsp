const API = "/api/05-metadata-filtered-rag";

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
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    select.appendChild(opt);
  }
}

const categorySelect = document.getElementById("meta-category");
const departmentSelect = document.getElementById("meta-department");
const prioritySelect = document.getElementById("meta-priority");
const filterCategory = document.getElementById("filter-category");
const filterDepartment = document.getElementById("filter-department");

fetch(`${API}/metadata-options`)
  .then((r) => r.json())
  .then(({ categories, departments, priorities }) => {
    fillSelect(categorySelect, categories);
    fillSelect(departmentSelect, departments);
    fillSelect(prioritySelect, priorities);
    fillSelect(filterCategory, categories);
    fillSelect(filterDepartment, departments);
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
    formData.append("category", categorySelect.value);
    formData.append("department", departmentSelect.value);
    formData.append("year", document.getElementById("meta-year").value);
    formData.append("language", document.getElementById("meta-language").value);
    formData.append("priority", prioritySelect.value);
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
    const filter = { category: filterCategory.value, department: filterDepartment.value };
    const { answer, sources } = await postJSON(`${API}/chat`, { question, filter });
    appendBubble("assistant", answer);
    sourcesEl.innerHTML = "";
    for (const s of sources) {
      const div = document.createElement("div");
      div.className = "source-item";
      div.textContent = `[${s.category ?? "-"}] ${s.sourceFile} — 페이지 ${s.page}`;
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
