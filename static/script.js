/*
  Minimal vanilla-JS chat client for api.py's /chat and /approve endpoints.
  No framework/build step on purpose — this is a small, static, single-page
  front end, so plain fetch() + DOM updates is simpler than reaching for one.
*/

const THREAD_KEY = "autostream_thread_id";
const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const examples = document.getElementById("examples");
const themeToggle = document.getElementById("themeToggle");

function getThreadId() {
  let id = localStorage.getItem(THREAD_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, id);
  }
  return id;
}

// Timecode-style timestamp (HH:MM:SS:frames-since-page-load) -- a small nod
// to the video-editing subject matter instead of a plain wall-clock time.
const sessionStart = performance.now();
function timecode() {
  const elapsedMs = performance.now() - sessionStart;
  const totalSeconds = Math.floor(elapsedMs / 1000);
  const hh = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
  const mm = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
  const ss = String(totalSeconds % 60).padStart(2, "0");
  const frames = String(Math.floor((elapsedMs % 1000) / 41.7)).padStart(2, "0"); // ~24fps
  return `${hh}:${mm}:${ss}:${frames}`;
}

function addMessage(role, text, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const tc = document.createElement("div");
  tc.className = "timecode";
  tc.textContent = timecode();
  wrap.appendChild(tc);

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (opts.pending ? " pending" : "");
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (role === "assistant" && opts.intent) {
    const tag = document.createElement("div");
    tag.className = "intent-tag";
    tag.textContent = opts.intent;
    wrap.appendChild(tag);
  }

  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.id = "typing-indicator";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

async function sendMessage(message) {
  if (!message.trim()) return;

  addMessage("user", message);
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  addTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: getThreadId(), message }),
    });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();

    removeTypingIndicator();

    // The server resolves lead approval itself when AUTO_APPROVE_DEMO_LEADS
    // is on (the default for this public demo) -- see api.py's /chat handler
    // -- so data.reply here is already the final, resumed reply in that case.
    // The client deliberately never calls /approve directly: that endpoint
    // requires an admin token, which a public page can't hold without
    // exposing it to every visitor. If a deployment turns auto-approve off,
    // pending_approval stays true and we just show that state honestly.
    addMessage("assistant", data.reply, {
      intent: data.intent,
      pending: data.pending_approval,
    });
  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", "Sorry, I couldn't reach the assistant just now. Please try again in a moment.");
    console.error(err);
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

examples.addEventListener("click", (e) => {
  if (e.target.matches(".example-chip")) {
    examples.style.display = "none";
    sendMessage(e.target.textContent);
  }
});

// Theme toggle cycles: system -> light -> dark -> system
function applyTheme(mode) {
  if (mode === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", mode);
  }
  localStorage.setItem("autostream_theme", mode);
}
themeToggle.addEventListener("click", () => {
  const current = localStorage.getItem("autostream_theme") || "system";
  const next = current === "system" ? "light" : current === "light" ? "dark" : "system";
  applyTheme(next);
});
applyTheme(localStorage.getItem("autostream_theme") || "system");

input.focus();
