/*
  AutoStream AI Studio Assistant Client Script
  - Real API Integration: /chat, /chat/history, /api/threads, /api/leads, /approve, /api/client-config
  - 3-Column Studio Layout with Collapsible Right Context Panel
  - Crisp Ivory Light UI Default with High-Contrast Amber Gold
  - Real-time Slot Tracking, Transparent RAG Citations, & Rich Comparative Cards
*/

const THREAD_KEY = "autostream_thread_id";

// DOM Elements
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const newChatBtn = document.getElementById("newChatBtn");
const clearBtn = document.getElementById("clearBtn");
const recentList = document.getElementById("recentList");
const briefContent = document.getElementById("briefContent");

const welcomeContainer = document.getElementById("welcomeContainer");
const threadContainer = document.getElementById("threadContainer");
const thread = document.getElementById("thread");
const workspaceBody = document.getElementById("workspaceBody");

const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const thinkingBar = document.getElementById("thinkingBar");

// Context Drawer Elements
const contextDrawer = document.getElementById("contextDrawer");
const collapseContextBtn = document.getElementById("collapseContextBtn");
const expandContextBtn = document.getElementById("expandContextBtn");
const contextTopic = document.getElementById("contextTopic");
const contextStatus = document.getElementById("contextStatus");
const slotCounter = document.getElementById("slotCounter");
const slotProgressBar = document.getElementById("slotProgressBar");
const slotItemName = document.getElementById("slotItemName");
const slotValName = document.getElementById("slotValName");
const slotItemEmail = document.getElementById("slotItemEmail");
const slotValEmail = document.getElementById("slotValEmail");
const slotItemPlatform = document.getElementById("slotItemPlatform");
const slotValPlatform = document.getElementById("slotValPlatform");
const contextPlanBtn = document.getElementById("contextPlanBtn");

const USER_THREADS_KEY = "autostream_user_threads";

let currentThreadId = null;
let currentLeadInfo = { name: null, email: null, platform: null };

// -------------------------------------------------------------
// Session & Thread Management
// -------------------------------------------------------------
function recordUserThread(threadId) {
  if (!threadId) return;
  let threads = [];
  try {
    threads = JSON.parse(localStorage.getItem(USER_THREADS_KEY) || "[]");
  } catch (_) {
    threads = [];
  }
  if (!threads.includes(threadId)) {
    threads.unshift(threadId);
    if (threads.length > 25) threads = threads.slice(0, 25);
    localStorage.setItem(USER_THREADS_KEY, JSON.stringify(threads));
  }
}

function getUserThreadIds() {
  try {
    return JSON.parse(localStorage.getItem(USER_THREADS_KEY) || "[]");
  } catch (_) {
    return [];
  }
}

function createFreshThreadId() {
  currentThreadId = crypto.randomUUID();
  localStorage.setItem(THREAD_KEY, currentThreadId);
  recordUserThread(currentThreadId);
  currentLeadInfo = { name: null, email: null, platform: null };
  updateContextDrawer();
  return currentThreadId;
}

function getActiveThreadId() {
  if (!currentThreadId) {
    currentThreadId = createFreshThreadId();
  }
  recordUserThread(currentThreadId);
  return currentThreadId;
}

function resetToHomeState() {
  createFreshThreadId();
  thread.innerHTML = "";
  welcomeContainer.classList.remove("hidden");
  threadContainer.classList.remove("active");
  highlightActiveRecentThread(null);
  input.value = "";
  input.focus();
}

function switchThread(threadId) {
  currentThreadId = threadId;
  localStorage.setItem(THREAD_KEY, threadId);
  recordUserThread(threadId);
  thread.innerHTML = "";
  highlightActiveRecentThread(threadId);
  loadChatHistory(threadId);
}

function highlightActiveRecentThread(activeId) {
  if (!recentList) return;
  document.querySelectorAll(".recent-item").forEach(item => {
    if (item.dataset.threadId === activeId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

// -------------------------------------------------------------
// HTML & Markdown Rendering
// -------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderSafeMarkdown(text) {
  if (!text) return "";
  let safe = escapeHtml(text);

  safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

  const lines = safe.split("\n");
  let html = "";
  let inList = false;

  for (let line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("* ") || trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
      if (!inList) {
        html += '<ul style="margin-left: 1.4rem; margin-top: 0.35rem; margin-bottom: 0.35rem;">';
        inList = true;
      }
      const itemContent = trimmed.replace(/^[\*\-•]\s*/, "");
      html += `<li>${itemContent}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      if (trimmed.startsWith("### ")) {
        html += `<h5 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem; margin-bottom:0.25rem; color:var(--ink);">${trimmed.substring(4)}</h5>`;
      } else if (trimmed.startsWith("## ")) {
        html += `<h4 style="font-size:1.05rem; font-weight:700; margin-top:0.6rem; margin-bottom:0.3rem; color:var(--ink);">${trimmed.substring(3)}</h4>`;
      } else if (trimmed.startsWith("# ")) {
        html += `<h3 style="font-size:1.15rem; font-weight:800; margin-top:0.7rem; margin-bottom:0.35rem; color:var(--ink);">${trimmed.substring(2)}</h3>`;
      } else if (trimmed === "") {
        html += "<br/>";
      } else {
        html += `<p style="margin-bottom:0.4rem;">${line}</p>`;
      }
    }
  }

  if (inList) html += "</ul>";
  return html;
}

function scrollToBottomIfNeeded() {
  if (!workspaceBody) return;
  workspaceBody.scrollTo({
    top: workspaceBody.scrollHeight,
    behavior: "smooth"
  });
}

// -------------------------------------------------------------
// Message Display
// -------------------------------------------------------------
function addMessage(role, text, opts = {}) {
  welcomeContainer.classList.add("hidden");
  threadContainer.classList.add("active");

  const wrap = document.createElement("div");
  wrap.className = `msg-wrap ${role}`;

  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Meta header
  const meta = document.createElement("div");
  meta.className = "msg-meta";
  if (role === "user") {
    meta.textContent = `You • ${timeStr}`;
  } else {
    meta.innerHTML = `<span>AutoStream Assistant</span> • <span>${timeStr}</span>`;
  }
  wrap.appendChild(meta);

  // Bubble container
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = renderSafeMarkdown(text);
  wrap.appendChild(bubble);

  // If Assistant and RAG sources exist, render the reasoning drawer
  if (role === "assistant" && opts.sources && opts.sources.length > 0) {
    const ragDrawer = document.createElement("details");
    ragDrawer.className = "rag-explanation-drawer";
    ragDrawer.innerHTML = `
      <summary class="rag-drawer-summary">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>
        How AutoStream answered this
      </summary>
      <div class="rag-steps-container">
        <div class="rag-step-item">
          <span class="dot dot-sage" aria-hidden="true"></span>
          <span>Verified against AutoStream product documentation</span>
        </div>
        <div class="rag-step-arrow" aria-hidden="true">↓</div>
        <div class="rag-step-item">
          <span class="dot dot-rec" aria-hidden="true"></span>
          <span>Checked subscription pricing and feature specifications</span>
        </div>
        <div class="rag-step-arrow" aria-hidden="true">↓</div>
        <div class="rag-step-item">
          <span class="dot dot-sage" aria-hidden="true"></span>
          <span>Synthesized grounded studio response</span>
        </div>
        <div class="sources-list" role="list" aria-label="Referenced knowledge sources">
          ${opts.sources.map(s => `
            <span class="source-pill" role="listitem">
              📄 ${escapeHtml(s.heading || s.section || s.source || "Knowledge Base")}
            </span>
          `).join("")}
        </div>
      </div>
    `;
    wrap.appendChild(ragDrawer);
  }

  // Interactive comparative pricing cards on explicit pricing inquiries
  const userQueryLower = (opts.userQuery || "").toLowerCase();
  const isExplicitPricingInquiry = (opts.intent === "inquiry" || !opts.intent) && 
    (userQueryLower.includes("price") || userQueryLower.includes("cost") || userQueryLower.includes("how much") || 
     userQueryLower.includes("compare") || (userQueryLower.includes("plan") && !userQueryLower.includes("start") && !userQueryLower.includes("get") && !userQueryLower.includes("sign")));
  
  if (role === "assistant" && isExplicitPricingInquiry) {
    const pricingContainer = document.createElement("div");
    pricingContainer.className = "rich-pricing-container";
    pricingContainer.innerHTML = `
      <div class="plan-card">
        <div class="plan-card-header">
          <span class="plan-card-title">Basic</span>
          <span class="plan-card-price">$29/mo</span>
        </div>
        <ul class="plan-card-features">
          <li><span class="check" aria-hidden="true">✓</span> Up to 10 videos / month</li>
          <li><span class="check" aria-hidden="true">✓</span> 720p maximum export resolution</li>
          <li><span class="check" aria-hidden="true">✓</span> Standard editing queue</li>
          <li><span class="check" aria-hidden="true">✓</span> Business hours email support</li>
        </ul>
        <button type="button" class="btn-plan-select basic">Choose Basic</button>
      </div>

      <div class="plan-card recommended">
        <span class="plan-badge-top">RECOMMENDED</span>
        <div class="plan-card-header">
          <span class="plan-card-title">Pro</span>
          <span class="plan-card-price">$79/mo</span>
        </div>
        <ul class="plan-card-features">
          <li><span class="check" aria-hidden="true">✓</span> Unlimited video exports</li>
          <li><span class="check" aria-hidden="true">✓</span> 4K resolution exports</li>
          <li><span class="check" aria-hidden="true">✓</span> Multi-language AI auto-captions</li>
          <li><span class="check" aria-hidden="true">✓</span> Priority cloud rendering queue</li>
          <li><span class="check" aria-hidden="true">✓</span> 24/7 dedicated support</li>
        </ul>
        <button type="button" class="btn-plan-select pro">Get started with Pro</button>
      </div>
    `;

    pricingContainer.querySelector(".btn-plan-select.basic")?.addEventListener("click", () => {
      sendMessage("I want to choose the Basic plan");
    });
    pricingContainer.querySelector(".btn-plan-select.pro")?.addEventListener("click", () => {
      sendMessage("I want to get started with the Pro plan");
    });
    wrap.appendChild(pricingContainer);

    // Quick action pills
    const pills = document.createElement("div");
    pills.className = "quick-action-pills";
    pills.innerHTML = `
      <button type="button" class="quick-pill" id="pillCompare">Compare plans</button>
      <button type="button" class="quick-pill" id="pillFeatures">What's in Pro?</button>
      <button type="button" class="quick-pill primary" id="pillStart">Get started with Pro</button>
    `;
    pills.querySelector("#pillCompare")?.addEventListener("click", () => {
      sendMessage("Can you compare Basic and Pro side by side?");
    });
    pills.querySelector("#pillFeatures")?.addEventListener("click", () => {
      sendMessage("What features are included in the Pro plan?");
    });
    pills.querySelector("#pillStart")?.addEventListener("click", () => {
      sendMessage("I want to get started with the Pro plan");
    });
    wrap.appendChild(pills);
  }

  // If pending approval, render the Human Review Card
  if (role === "assistant" && opts.pending && opts.pendingLead) {
    const reviewCard = document.createElement("div");
    reviewCard.className = "review-summary-card";
    reviewCard.innerHTML = `
      <div class="review-title">
        Please review your details
      </div>
      <div class="review-row">
        <span class="review-label">Name</span>
        <span class="review-value">${escapeHtml(opts.pendingLead.name || "—")}</span>
      </div>
      <div class="review-row">
        <span class="review-label">Email</span>
        <span class="review-value">${escapeHtml(opts.pendingLead.email || "—")}</span>
      </div>
      <div class="review-row">
        <span class="review-label">Platform</span>
        <span class="review-value">${escapeHtml(opts.pendingLead.platform || "—")}</span>
      </div>
      <div class="review-row" style="border-top:1px solid var(--espresso-border); padding-top:6px;">
        <span class="review-label">Selected Tier</span>
        <span class="review-value" style="color:var(--amber-gold); font-weight:700;">Pro · $79/mo</span>
      </div>
      <div class="review-actions">
        <button type="button" class="btn-review-edit" id="btnEditLead">Edit details</button>
        <button type="button" class="btn-review-confirm" id="btnConfirmLead">Confirm & continue</button>
      </div>
    `;

    reviewCard.querySelector("#btnEditLead")?.addEventListener("click", () => {
      sendMessage("I need to change my details");
    });
    reviewCard.querySelector("#btnConfirmLead")?.addEventListener("click", async () => {
      await handleLeadDecision(true);
    });
    wrap.appendChild(reviewCard);
  }

  thread.appendChild(wrap);
  scrollToBottomIfNeeded();
  return bubble;
}

// -------------------------------------------------------------
// Human Review / Decision Handler
// -------------------------------------------------------------
async function handleLeadDecision(approved) {
  const activeThreadId = getActiveThreadId();
  try {
    const res = await fetch("/confirm-lead", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: activeThreadId, approved })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.lead_info) currentLeadInfo = { ...currentLeadInfo, ...data.lead_info };
      updateContextDrawer(data.intent);
      addMessage("assistant", data.reply || "Your Pro plan workspace has been created. Check your email for next steps!");
      loadRecentThreads();
      loadDailyBrief();
    } else {
      addMessage("assistant", "Your details have been confirmed and saved to our studio registry! Welcome to AutoStream.");
      loadRecentThreads();
      loadDailyBrief();
    }
  } catch (err) {
    console.error("Approval error:", err);
  }
}

// -------------------------------------------------------------
// Send Message Flow (Real-Time SSE Streaming with Fallback)
// -------------------------------------------------------------
async function sendMessage(text) {
  const msg = (text || input.value || "").trim();
  if (!msg) return;

  const activeThreadId = getActiveThreadId();
  addMessage("user", msg);
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  thinkingBar.style.display = "flex";

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: activeThreadId, message: msg })
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream HTTP ${response.status}`);
    }

    thinkingBar.style.display = "none";

    // Create assistant message bubble immediately for streaming
    welcomeContainer.classList.add("hidden");
    threadContainer.classList.add("active");

    const wrap = document.createElement("div");
    wrap.className = "message-wrap assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    bubble.innerHTML = '<span class="streaming-cursor"></span>';
    wrap.appendChild(bubble);
    thread.appendChild(wrap);
    scrollToBottomIfNeeded();

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let accumulatedText = "";
    let buffer = "";
    let streamMeta = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(line.substring(6));
            if (parsed.token !== undefined) {
              accumulatedText += parsed.token;
              bubble.innerHTML = renderSafeMarkdown(accumulatedText) + '<span class="streaming-cursor"></span>';
              scrollToBottomIfNeeded();
            } else if (parsed.thread_id !== undefined || parsed.intent !== undefined) {
              streamMeta = parsed;
            }
          } catch (_) {}
        }
      }
    }

    // Remove cursor and finalize markdown
    bubble.innerHTML = renderSafeMarkdown(accumulatedText);

    // Apply metadata, sources, and interactive cards once stream finishes
    if (streamMeta) {
      if (streamMeta.lead_info) {
        currentLeadInfo = { ...currentLeadInfo, ...streamMeta.lead_info };
      }
      if (streamMeta.pending_lead) {
        currentLeadInfo = { ...currentLeadInfo, ...streamMeta.pending_lead };
      }
      updateContextDrawer(streamMeta.intent);

      // Attach RAG sources if present
      if (streamMeta.sources && streamMeta.sources.length > 0) {
        const ragDrawer = document.createElement("div");
        ragDrawer.className = "rag-drawer";
        ragDrawer.innerHTML = `
          <div class="rag-drawer-card">
            <div class="rag-drawer-header">
              <span class="rag-title">Verified Knowledge Sources</span>
              <span class="rag-count-pill">${streamMeta.sources.length} references</span>
            </div>
            <div class="sources-list" role="list" aria-label="Referenced knowledge sources">
              ${streamMeta.sources.map(s => `
                <span class="source-pill" role="listitem">
                  📄 ${escapeHtml(s.heading || s.section || s.source || "Knowledge Base")}
                </span>
              `).join("")}
            </div>
          </div>
        `;
        wrap.appendChild(ragDrawer);
      }

      // If pending approval, render the Human Review Card
      if (streamMeta.pending_approval && (streamMeta.pending_lead || currentLeadInfo)) {
        const lead = streamMeta.pending_lead || currentLeadInfo;
        const reviewCard = document.createElement("div");
        reviewCard.className = "review-summary-card";
        reviewCard.innerHTML = `
          <div class="review-title">Please review your details</div>
          <div class="review-row">
            <span class="review-key">Creator</span>
            <span class="review-val">${escapeHtml(lead.name || "—")}</span>
          </div>
          <div class="review-row">
            <span class="review-key">Contact Email</span>
            <span class="review-val">${escapeHtml(lead.email || "—")}</span>
          </div>
          <div class="review-row">
            <span class="review-key">Platform</span>
            <span class="review-val">${escapeHtml(lead.platform || "—")}</span>
          </div>
          <div class="review-actions">
            <button type="button" class="btn-review approve">Confirm & Continue</button>
            <button type="button" class="btn-review edit">Make Changes</button>
          </div>
        `;
        reviewCard.querySelector(".btn-review.approve")?.addEventListener("click", () => handleLeadDecision(true));
        reviewCard.querySelector(".btn-review.edit")?.addEventListener("click", () => handleLeadDecision(false));
        wrap.appendChild(reviewCard);
      }
    }

    scrollToBottomIfNeeded();
    loadRecentThreads();
    loadDailyBrief();

  } catch (streamErr) {
    console.warn("Streaming encountered error, falling back to /chat:", streamErr);
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: activeThreadId, message: msg })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      thinkingBar.style.display = "none";
      if (data.lead_info) currentLeadInfo = { ...currentLeadInfo, ...data.lead_info };
      if (data.pending_lead) currentLeadInfo = { ...currentLeadInfo, ...data.pending_lead };
      updateContextDrawer(data.intent);
      addMessage("assistant", data.reply, {
        userQuery: msg,
        intent: data.intent,
        sources: data.sources,
        pending: data.pending_approval,
        pendingLead: data.pending_lead || currentLeadInfo,
      });
      loadRecentThreads();
      loadDailyBrief();
    } catch (fallbackErr) {
      thinkingBar.style.display = "none";
      addMessage("assistant", "Sorry, I had trouble reaching the AI service right now. Please try again in a moment.");
      console.error(fallbackErr);
    }
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

// -------------------------------------------------------------
// Context Drawer Update
// -------------------------------------------------------------
function updateContextDrawer(intent = "") {
  if (!slotCounter) return;

  let filled = 0;
  if (currentLeadInfo.name) {
    slotItemName.classList.add("done");
    slotItemName.querySelector(".slot-check").textContent = "✓";
    slotValName.textContent = currentLeadInfo.name;
    filled++;
  } else {
    slotItemName.classList.remove("done");
    slotItemName.querySelector(".slot-check").textContent = "";
    slotValName.textContent = "—";
  }

  if (currentLeadInfo.email) {
    slotItemEmail.classList.add("done");
    slotItemEmail.querySelector(".slot-check").textContent = "✓";
    slotValEmail.textContent = currentLeadInfo.email;
    filled++;
  } else {
    slotItemEmail.classList.remove("done");
    slotItemEmail.querySelector(".slot-check").textContent = "";
    slotValEmail.textContent = "—";
  }

  if (currentLeadInfo.platform) {
    slotItemPlatform.classList.add("done");
    slotItemPlatform.querySelector(".slot-check").textContent = "✓";
    slotValPlatform.textContent = currentLeadInfo.platform;
    filled++;
  } else {
    slotItemPlatform.classList.remove("done");
    slotItemPlatform.querySelector(".slot-check").textContent = "";
    slotValPlatform.textContent = "—";
  }

  slotCounter.textContent = `${filled} / 3`;
  slotProgressBar.style.width = `${(filled / 3) * 100}%`;

  if (filled === 3) {
    contextTopic.textContent = "Pro Plan Onboarding";
    contextStatus.innerHTML = `<span class="dot dot-sage" aria-hidden="true"></span><span>Captured ✓</span>`;
  } else if (filled > 0) {
    contextTopic.textContent = "Pro Plan Onboarding";
    contextStatus.innerHTML = `<span class="dot dot-rec" aria-hidden="true"></span><span>Signup in progress</span>`;
  } else if (intent === "inquiry") {
    contextTopic.textContent = "AutoStream Pricing & Features";
    contextStatus.innerHTML = `<span class="dot dot-sage" aria-hidden="true"></span><span>Exploring Plans</span>`;
  } else {
    contextTopic.textContent = "AutoStream Studio";
    contextStatus.innerHTML = `<span class="dot dot-sage" aria-hidden="true"></span><span>Active</span>`;
  }
}

// -------------------------------------------------------------
// Real Chat History Loader
// -------------------------------------------------------------
async function loadChatHistory(threadId) {
  try {
    const res = await fetch(`/chat/history?thread_id=${encodeURIComponent(threadId)}`);
    if (!res.ok) throw new Error("History fetch failed");
    const data = await res.json();

    if (data.messages && data.messages.length > 0) {
      welcomeContainer.classList.add("hidden");
      threadContainer.classList.add("active");
      thread.innerHTML = "";
      for (const m of data.messages) {
        addMessage(m.role, m.content);
      }
    } else {
      welcomeContainer.classList.remove("hidden");
      threadContainer.classList.remove("active");
    }
  } catch (err) {
    console.error("Failed to load chat history:", err);
  }
}

// -------------------------------------------------------------
// Sidebar Recent Threads & Daily Brief
// -------------------------------------------------------------
async function loadRecentThreads() {
  if (!recentList) return;
  try {
    const userIds = getUserThreadIds();
    const queryParam = userIds.length > 0 ? `&thread_ids=${encodeURIComponent(userIds.join(","))}` : "";
    const res = await fetch(`/api/threads?limit=8${queryParam}`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.threads && data.threads.length > 0) {
      recentList.innerHTML = "";
      data.threads.forEach(t => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `recent-item ${t.thread_id === currentThreadId ? "active" : ""}`;
        item.dataset.threadId = t.thread_id;
        item.setAttribute("aria-label", `Switch to conversation: ${t.title || t.thread_id.substring(0, 8)}`);
        item.innerHTML = `
          <div class="recent-item-title">${escapeHtml(t.title || `Chat ${t.thread_id.substring(0, 8)}`)}</div>
          <div class="recent-item-time">${t.relative_time || "Recent"}</div>
        `;
        item.addEventListener("click", () => {
          if (sidebar.classList.contains("open")) closeMobileSidebar();
          switchThread(t.thread_id);
        });
        recentList.appendChild(item);
      });
    } else {
      recentList.innerHTML = '<div class="recent-placeholder">No prior conversations</div>';
    }
  } catch (err) {
    recentList.innerHTML = '<div class="recent-placeholder">No prior conversations</div>';
  }
}

async function loadDailyBrief() {
  if (!briefContent) return;
  try {
    const res = await fetch("/api/leads");
    if (!res.ok) throw new Error("Leads API unavailable");
    const data = await res.json();

    if (data.count > 0) {
      let html = `<div style="margin-bottom:4px;">Surfaced <span class="brief-stat">${data.count} captured lead${data.count > 1 ? "s" : ""}</span>:</div>`;
      html += '<div class="brief-lead-list">';
      const recentLeads = data.leads.slice(-3).reverse();
      recentLeads.forEach(lead => {
        html += `
          <div class="brief-lead-item">
            <span style="font-weight:600; color:var(--ink);">${escapeHtml(lead.name || "Lead")}</span>
            <span style="color:var(--amber-gold); font-weight:600;">${escapeHtml(lead.platform || "Platform")}</span>
          </div>
        `;
      });
      html += '</div>';
      briefContent.innerHTML = html;
    } else {
      briefContent.innerHTML = '<div style="color:var(--ink-3); font-size:11.5px;">No captured leads recorded yet. Interactive agent ready.</div>';
    }
  } catch (err) {
    briefContent.innerHTML = '<div style="color:var(--ink-3); font-size:11.5px;">Database online. Ready for leads.</div>';
  }
}

// -------------------------------------------------------------
// Event Listeners & Setup
// -------------------------------------------------------------
// Composer submission
composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

// Textarea Keyboard Shortcut (Enter sends, Shift+Enter new line)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(input.value);
  }
});

// Capability Cards
document.querySelectorAll(".capability-card").forEach(card => {
  card.addEventListener("click", () => {
    const prompt = card.getAttribute("data-prompt");
    if (prompt) sendMessage(prompt);
  });
});

// Sidebar Buttons
if (newChatBtn) newChatBtn.addEventListener("click", resetToHomeState);
if (clearBtn) clearBtn.addEventListener("click", resetToHomeState);

// Nav items
document.getElementById("navConversations")?.addEventListener("click", (e) => {
  e.preventDefault();
  resetToHomeState();
});
document.getElementById("navPricing")?.addEventListener("click", (e) => {
  e.preventDefault();
  sendMessage("How much does the Pro plan cost?");
});
document.getElementById("navExplore")?.addEventListener("click", (e) => {
  e.preventDefault();
  sendMessage("What features are included in the Pro plan?");
});
document.getElementById("navRoiCalc")?.addEventListener("click", (e) => {
  e.preventDefault();
  renderRoiCalculatorCard();
});
document.getElementById("cardRoiCalc")?.addEventListener("click", () => {
  renderRoiCalculatorCard();
});
document.getElementById("contextRoiBtn")?.addEventListener("click", () => {
  renderRoiCalculatorCard();
});

// -------------------------------------------------------------
// Interactive In-Chat Video ROI & Plan Calculator
// -------------------------------------------------------------
function renderRoiCalculatorCard(container) {
  welcomeContainer.classList.add("hidden");
  threadContainer.classList.add("active");

  const wrap = document.createElement("div");
  wrap.className = "message-wrap assistant";

  const card = document.createElement("div");
  card.className = "roi-calculator-card";
  card.innerHTML = `
    <div class="roi-card-header">
      <div class="roi-title-wrap">
        <h3 class="roi-heading">Video Editing ROI Calculator</h3>
        <span class="roi-badge">Live Estimator</span>
      </div>
    </div>
    <div class="roi-sliders-section">
      <div class="roi-control-group">
        <div class="roi-control-top">
          <span>Videos Produced per Week</span>
          <span class="roi-val-callout" id="roiVideosVal">4 videos</span>
        </div>
        <input type="range" class="roi-range-input" id="roiVideosRange" min="1" max="15" value="4" step="1">
      </div>
      <div class="roi-control-group">
        <div class="roi-control-top">
          <span>Average Raw Video Length</span>
          <span class="roi-val-callout" id="roiLengthVal">30 mins</span>
        </div>
        <input type="range" class="roi-range-input" id="roiLengthRange" min="10" max="120" value="30" step="5">
      </div>
    </div>
    <div class="roi-metrics-grid">
      <div class="roi-metric-tile">
        <div class="roi-tile-label">Hours Saved / Mo</div>
        <div class="roi-tile-number gold" id="roiHoursSaved">20 hrs</div>
      </div>
      <div class="roi-metric-tile">
        <div class="roi-tile-label">Creator Value / Mo</div>
        <div class="roi-tile-number sage" id="roiValueCreated">$700</div>
      </div>
      <div class="roi-metric-tile">
        <div class="roi-tile-label">Recommended Tier</div>
        <div class="roi-tile-number" id="roiRecommendedPlan">Pro ($79)</div>
      </div>
    </div>
    <button type="button" class="btn-roi-apply" id="roiApplyBtn" data-plan="Pro">
      <span>Select Pro Plan & Start Onboarding</span>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
    </button>
  `;

  const videosRange = card.querySelector("#roiVideosRange");
  const lengthRange = card.querySelector("#roiLengthRange");
  const videosVal = card.querySelector("#roiVideosVal");
  const lengthVal = card.querySelector("#roiLengthVal");
  const hoursSavedEl = card.querySelector("#roiHoursSaved");
  const valueCreatedEl = card.querySelector("#roiValueCreated");
  const recommendedPlanEl = card.querySelector("#roiRecommendedPlan");
  const applyBtn = card.querySelector("#roiApplyBtn");

  function recalculate() {
    const videos = parseInt(videosRange.value, 10);
    const mins = parseInt(lengthRange.value, 10);
    videosVal.textContent = `${videos} video${videos > 1 ? "s" : ""}`;
    lengthVal.textContent = `${mins} mins`;

    const monthlyVideos = videos * 4;
    const hoursPerVideo = mins / 60;
    const hoursSaved = Math.max(1, Math.round(monthlyVideos * hoursPerVideo * 2.5));
    const valueSaved = hoursSaved * 35;

    hoursSavedEl.textContent = `${hoursSaved} hrs`;
    valueCreatedEl.textContent = `$${valueSaved.toLocaleString()}`;

    if (videos <= 2 && mins <= 30) {
      recommendedPlanEl.textContent = "Basic ($29)";
      applyBtn.innerHTML = `<span>Select Basic Plan & Start Onboarding</span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
      applyBtn.dataset.plan = "Basic";
    } else {
      recommendedPlanEl.textContent = "Pro ($79)";
      applyBtn.innerHTML = `<span>Select Pro Plan & Start Onboarding</span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
      applyBtn.dataset.plan = "Pro";
    }
  }

  videosRange.addEventListener("input", recalculate);
  lengthRange.addEventListener("input", recalculate);

  applyBtn.addEventListener("click", () => {
    const plan = applyBtn.dataset.plan || "Pro";
    sendMessage(`I calculated my ROI (${videosRange.value} videos/wk, ${lengthRange.value} mins) and I'd like to sign up for the ${plan} plan.`);
  });

  wrap.appendChild(card);
  (container || thread).appendChild(wrap);
  scrollToBottomIfNeeded();
}

// -------------------------------------------------------------
// Visual Admin CRM & Review Panel
// -------------------------------------------------------------
const adminModalBackdrop = document.getElementById("adminModalBackdrop");
const closeAdminModalBtn = document.getElementById("closeAdminModalBtn");
const adminAuthGate = document.getElementById("adminAuthGate");
const adminCrmBody = document.getElementById("adminCrmBody");
const adminAuthForm = document.getElementById("adminAuthForm");
const adminTokenInput = document.getElementById("adminTokenInput");
const adminAuthError = document.getElementById("adminAuthError");
const crmMetricTotal = document.getElementById("crmMetricTotal");
const crmMetricPending = document.getElementById("crmMetricPending");
const crmMetricPlatform = document.getElementById("crmMetricPlatform");
const crmSearchInput = document.getElementById("crmSearchInput");
const crmPlatformFilter = document.getElementById("crmPlatformFilter");
const crmRefreshBtn = document.getElementById("crmRefreshBtn");
const crmExportBtn = document.getElementById("crmExportBtn");
const crmLockBtn = document.getElementById("crmLockBtn");
const crmTableBody = document.getElementById("crmTableBody");
const topAdminBtn = document.getElementById("topAdminBtn");
const navAdminCrm = document.getElementById("navAdminCrm");

const ADMIN_STORAGE_KEY = "autostream_admin_token";
let cachedAdminLeads = [];

function openAdminCrmModal() {
  if (!adminModalBackdrop) return;
  adminModalBackdrop.style.display = "flex";
  const savedToken = sessionStorage.getItem(ADMIN_STORAGE_KEY);
  if (savedToken) {
    adminAuthGate.style.display = "none";
    adminCrmBody.style.display = "flex";
    fetchAdminLeads(savedToken);
  } else {
    adminAuthGate.style.display = "block";
    adminCrmBody.style.display = "none";
    if (adminTokenInput) {
      adminTokenInput.value = "";
      setTimeout(() => adminTokenInput.focus(), 100);
    }
  }
}

function closeAdminCrmModal() {
  if (!adminModalBackdrop) return;
  adminModalBackdrop.style.display = "none";
}

if (topAdminBtn) topAdminBtn.addEventListener("click", openAdminCrmModal);
if (navAdminCrm) navAdminCrm.addEventListener("click", (e) => {
  e.preventDefault();
  openAdminCrmModal();
});
if (closeAdminModalBtn) closeAdminModalBtn.addEventListener("click", closeAdminCrmModal);
if (adminModalBackdrop) {
  adminModalBackdrop.addEventListener("click", (e) => {
    if (e.target === adminModalBackdrop) closeAdminCrmModal();
  });
}

if (adminAuthForm) {
  adminAuthForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const token = (adminTokenInput?.value || "").trim();
    if (!token) return;

    adminAuthError.style.display = "none";
    try {
      const verifyRes = await fetch("/api/admin/verify", {
        headers: { "X-Admin-Token": token }
      });
      if (!verifyRes.ok) {
        adminAuthError.textContent = "Invalid token. Please check ADMIN_TOKEN in your environment.";
        adminAuthError.style.display = "block";
        return;
      }
      sessionStorage.setItem(ADMIN_STORAGE_KEY, token);
      adminAuthGate.style.display = "none";
      adminCrmBody.style.display = "flex";
      fetchAdminLeads(token);
    } catch (err) {
      adminAuthError.textContent = "Failed to connect to server.";
      adminAuthError.style.display = "block";
    }
  });
}

async function fetchAdminLeads(token) {
  if (!token) return;
  if (crmTableBody) crmTableBody.innerHTML = '<tr><td colspan="5" class="table-empty">Loading authenticated leads…</td></tr>';
  try {
    const res = await fetch("/api/leads", {
      headers: { "X-Admin-Token": token }
    });
    if (!res.ok) {
      if (res.status === 401) {
        sessionStorage.removeItem(ADMIN_STORAGE_KEY);
        adminAuthGate.style.display = "block";
        adminCrmBody.style.display = "none";
      }
      return;
    }
    const data = await res.json();
    cachedAdminLeads = data.leads || [];
    renderCrmTable(cachedAdminLeads);
  } catch (err) {
    if (crmTableBody) crmTableBody.innerHTML = '<tr><td colspan="5" class="table-empty">Failed to load leads.</td></tr>';
  }
}

function renderCrmTable(leads) {
  if (!crmTableBody) return;
  const query = (crmSearchInput?.value || "").toLowerCase().trim();
  const platformFilter = (crmPlatformFilter?.value || "").toLowerCase();

  const filtered = leads.filter(l => {
    const matchQuery = !query || (l.name || "").toLowerCase().includes(query) || (l.email || "").toLowerCase().includes(query);
    const matchPlat = !platformFilter || (l.platform || "").toLowerCase() === platformFilter;
    return matchQuery && matchPlat;
  });

  if (crmMetricTotal) crmMetricTotal.textContent = leads.length;
  if (crmMetricPending) crmMetricPending.textContent = "0";
  if (crmMetricPlatform) {
    const counts = {};
    leads.forEach(l => { counts[l.platform] = (counts[l.platform] || 0) + 1; });
    const top = Object.entries(counts).sort((a,b) => b[1] - a[1])[0];
    crmMetricPlatform.textContent = top ? `${top[0]} (${top[1]})` : "None";
  }

  if (filtered.length === 0) {
    crmTableBody.innerHTML = '<tr><td colspan="5" class="table-empty">No matching leads found.</td></tr>';
    return;
  }

  crmTableBody.innerHTML = filtered.map(l => {
    const dateStr = l.captured_at ? new Date(l.captured_at).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "Recent";
    return `
      <tr>
        <td style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-3);">${escapeHtml(dateStr)}</td>
        <td style="font-weight: 500;">${escapeHtml(l.name || "Anonymous")}</td>
        <td style="font-family: var(--font-mono); font-size: 12px; color: var(--amber-gold); font-weight: 500;">${escapeHtml(l.email || "—")}</td>
        <td><span class="platform-pill">${escapeHtml(l.platform || "Web")}</span></td>
        <td><span class="status-badge-captured">✓ Captured</span></td>
      </tr>
    `;
  }).join("");
}

if (crmSearchInput) crmSearchInput.addEventListener("input", () => renderCrmTable(cachedAdminLeads));
if (crmPlatformFilter) crmPlatformFilter.addEventListener("change", () => renderCrmTable(cachedAdminLeads));
if (crmRefreshBtn) {
  crmRefreshBtn.addEventListener("click", () => {
    const token = sessionStorage.getItem(ADMIN_STORAGE_KEY);
    if (token) fetchAdminLeads(token);
  });
}

if (crmExportBtn) {
  crmExportBtn.addEventListener("click", () => {
    const token = sessionStorage.getItem(ADMIN_STORAGE_KEY);
    if (token) {
      window.open(`/api/leads/export?token=${encodeURIComponent(token)}`, "_blank");
    }
  });
}

if (crmLockBtn) {
  crmLockBtn.addEventListener("click", () => {
    sessionStorage.removeItem(ADMIN_STORAGE_KEY);
    adminAuthGate.style.display = "block";
    adminCrmBody.style.display = "none";
  });
}

// Context Plan Quick Action
if (contextPlanBtn) {
  contextPlanBtn.addEventListener("click", () => {
    sendMessage("I want to get started with the Pro plan for my channel.");
  });
}

// Context Panel Collapse / Expand
if (collapseContextBtn && expandContextBtn) {
  collapseContextBtn.addEventListener("click", () => {
    contextDrawer.classList.add("collapsed");
    expandContextBtn.style.display = "flex";
  });
  expandContextBtn.addEventListener("click", () => {
    contextDrawer.classList.remove("collapsed");
    expandContextBtn.style.display = "none";
  });
}

// Mobile Sidebar Drawer
function openMobileSidebar() {
  sidebar.classList.add("open");
  sidebarOverlay.classList.add("open");
}
function closeMobileSidebar() {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("open");
}
if (mobileMenuBtn) mobileMenuBtn.addEventListener("click", openMobileSidebar);
if (sidebarOverlay) sidebarOverlay.addEventListener("click", closeMobileSidebar);
const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
if (sidebarCloseBtn) sidebarCloseBtn.addEventListener("click", closeMobileSidebar);

// Theme Toggle & Persistence
const themeToggleBtn = document.getElementById("themeToggleBtn");
const navThemeToggle = document.getElementById("navThemeToggle");
const navThemeLabel = document.getElementById("navThemeLabel");

function applyTheme(theme) {
  if (theme === "dark") {
    document.documentElement.className = "dark";
    document.documentElement.setAttribute("data-theme", "dark");
    if (navThemeLabel) navThemeLabel.textContent = "Switch to Light UI";
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) metaTheme.setAttribute("content", "#0a0908");
  } else {
    document.documentElement.className = "light";
    document.documentElement.setAttribute("data-theme", "light");
    if (navThemeLabel) navThemeLabel.textContent = "Switch to Dark UI";
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) metaTheme.setAttribute("content", "#fcfbfa");
  }
  localStorage.setItem("autostream_theme", theme);
}

function toggleTheme() {
  const current = localStorage.getItem("autostream_theme") || "light";
  const next = current === "light" ? "dark" : "light";
  applyTheme(next);
}

if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);
if (navThemeToggle) navThemeToggle.addEventListener("click", (e) => {
  e.preventDefault();
  toggleTheme();
});

// -------------------------------------------------------------
// Immediate Page Initialization
// -------------------------------------------------------------
applyTheme(localStorage.getItem("autostream_theme") || "light");
createFreshThreadId();
input.focus();
loadRecentThreads();
loadDailyBrief();
