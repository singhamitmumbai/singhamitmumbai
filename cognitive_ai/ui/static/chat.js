(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const form = $("composer");
  const input = $("input");
  const sendBtn = $("send");
  const statusLine = $("statusLine");
  const versionLabel = $("versionLabel");
  const llmPill = $("llmPill");
  const agiPill = $("agiPill");
  const historyList = $("historyList");
  const suggestions = $("suggestions");
  const modeSelect = $("modeSelect");
  const sidebar = $("sidebar");
  const backdrop = $("sidebarBackdrop");

  const state = {
    busy: false,
    history: [],
    theme: localStorage.getItem("complete_ai_theme") || "light",
  };

  // --- theme ---
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("complete_ai_theme", theme);
  }
  applyTheme(state.theme);

  // --- markdown-lite ---
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(text) {
    const raw = String(text || "");
    // Extract fenced code blocks first
    const blocks = [];
    let work = raw.replace(/```([\s\S]*?)```/g, (_, code) => {
      const i = blocks.length;
      blocks.push(`<pre><code>${escapeHtml(code.replace(/^\w*\n/, ""))}</code></pre>`);
      return `\u0000BLOCK${i}\u0000`;
    });
    work = escapeHtml(work);
    work = work.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    work = work.replace(/`([^`]+)`/g, "<code>$1</code>");
    work = work.replace(/^### (.+)$/gm, "<strong>$1</strong>");
    work = work.replace(/^\- (.+)$/gm, "• $1");
    work = work.replace(/^\d+\. (.+)$/gm, "• $1");
    // paragraphs
    work = work
      .split(/\n{2,}/)
      .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("");
    work = work.replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[Number(i)] || "");
    return work;
  }

  function addMsg(role, text, meta = {}) {
    // Remove welcome if present
    const welcome = messages.querySelector(".welcome");
    if (welcome) welcome.remove();

    const el = document.createElement("div");
    el.className = `msg ${role}`;

    const who = document.createElement("div");
    who.className = "who";
    const label = document.createElement("span");
    label.textContent = role === "user" ? "You" : "Complete AI";
    who.appendChild(label);

    if (role === "bot") {
      const actions = document.createElement("div");
      actions.className = "actions";
      const copy = document.createElement("button");
      copy.type = "button";
      copy.textContent = "Copy";
      copy.title = "Copy reply";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(text);
          copy.textContent = "Copied";
          setTimeout(() => (copy.textContent = "Copy"), 1200);
        } catch {
          copy.textContent = "Failed";
        }
      });
      actions.appendChild(copy);
      who.appendChild(actions);
    }
    el.appendChild(who);

    const body = document.createElement("div");
    body.className = "body";
    if (role === "bot") body.innerHTML = renderMarkdown(text);
    else body.textContent = text;
    el.appendChild(body);

    if (role === "bot" && (meta.thinkingMarkdown || meta.thinkingBrief || meta.thinking)) {
      const details = document.createElement("details");
      details.className = "thinking";
      const summary = document.createElement("summary");
      summary.textContent = meta.thinkingBrief
        ? `Thinking · ${meta.thinkingBrief}`
        : "Thinking";
      details.appendChild(summary);
      const thinkBody = document.createElement("div");
      thinkBody.className = "thinking-body";
      if (meta.thinkingMarkdown) {
        thinkBody.innerHTML = renderMarkdown(meta.thinkingMarkdown);
      } else if (meta.thinking && Array.isArray(meta.thinking.plan)) {
        const plan = (meta.thinking.plan || [])
          .map((s, i) => `${i + 1}. ${s}`)
          .join("\n");
        thinkBody.innerHTML = renderMarkdown(
          `**Plan**\n${plan || "(none)"}\n\n_strategy=\`${meta.thinking.strategy || "?"}\`_`
        );
      } else {
        thinkBody.textContent = "Deliberation attached.";
      }
      details.appendChild(thinkBody);
      el.appendChild(details);
    }

    if (role === "bot" && (meta.route || meta.llm || meta.confidence != null || meta.thinkingBrief)) {
      const metaEl = document.createElement("div");
      metaEl.className = "meta";
      if (meta.route) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = `route=${meta.route}`;
        metaEl.appendChild(t);
      }
      if (meta.llm) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = "llm";
        metaEl.appendChild(t);
      }
      if (meta.confidence != null) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = `conf=${Number(meta.confidence).toFixed(2)}`;
        metaEl.appendChild(t);
      }
      if (meta.proto_agi) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = "proto-agi";
        metaEl.appendChild(t);
      }
      if (meta.thinkingBrief) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = meta.thinkingBrief;
        metaEl.appendChild(t);
      }
      el.appendChild(metaEl);
    }

    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function showTyping(on) {
    const existing = messages.querySelector(".typing");
    if (existing) existing.remove();
    if (!on) return;
    const el = document.createElement("div");
    el.className = "typing";
    el.innerHTML = "<span></span><span></span><span></span>";
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  function showWelcome() {
    messages.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "welcome";
    wrap.innerHTML = `
      <h2>How can Complete AI help?</h2>
      <p>Every reply starts with think→plan. Chat, Finderbay, vision, coding, research, and proto-AGI goals — without claiming human-level AGI.</p>
      <div class="welcome-grid">
        <button type="button" data-prompt="Explain gravity in simple terms"><strong>Explain</strong><span>Gravity in simple terms</span></button>
        <button type="button" data-prompt="Show unpaid invoices for Project A"><strong>Business</strong><span>Unpaid invoices for Project A</span></button>
        <button type="button" data-prompt="multi-step research photosynthesis then verify"><strong>Research</strong><span>Multi-step research + verify</span></button>
        <button type="button" data-prompt="Write a poem about Mars"><strong>Create</strong><span>A short poem about Mars</span></button>
      </div>
    `;
    messages.appendChild(wrap);
    wrap.querySelectorAll("[data-prompt]").forEach((btn) => {
      btn.addEventListener("click", () => sendText(btn.dataset.prompt));
    });
  }

  function renderHistory() {
    historyList.innerHTML = "";
    if (!state.history.length) {
      historyList.innerHTML = `<p class="muted empty-hist">No saved turns yet</p>`;
      return;
    }
    state.history
      .slice()
      .reverse()
      .slice(0, 30)
      .forEach((item) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "history-item";
        b.textContent = item.user;
        b.title = item.user;
        b.addEventListener("click", () => {
          input.value = item.user;
          input.focus();
          closeSidebar();
        });
        historyList.appendChild(b);
      });
  }

  function setSuggestions(list) {
    suggestions.innerHTML = "";
    (list || []).slice(0, 6).forEach((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = s;
      b.addEventListener("click", () => sendText(s));
      suggestions.appendChild(b);
    });
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body !== undefined ? "POST" : "GET",
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText || "request failed");
    return data;
  }

  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      const name = st.name || "Complete AI";
      const ver = st.version || "";
      const llmOn = !!(st.llm_at_runtime || st.chatgpt_class_general_chat);
      statusLine.textContent = `${name} v${ver} · chat=${(st.modes && st.modes.general_chat) || "symbolic"} · session=${st.session_turns ?? 0}`;
      versionLabel.textContent = `${name} v${ver}`;
      llmPill.textContent = llmOn ? "llm:on" : "llm:off";
      agiPill.textContent = "agi=false";
    } catch {
      statusLine.textContent = "offline — start the chat server";
      llmPill.textContent = "llm:?";
    }
  }

  async function sendText(text) {
    const trimmed = (text || "").trim();
    if (!trimmed || state.busy) return;
    state.busy = true;
    sendBtn.disabled = true;
    addMsg("user", trimmed);
    input.value = "";
    autosize();
    showTyping(true);
    closeSidebar();

    try {
      if (trimmed === "/status") {
        const st = await api("/api/status");
        showTyping(false);
        addMsg("bot", "```\n" + JSON.stringify(st, null, 2).slice(0, 3500) + "\n```");
      } else if (trimmed === "/diag" || trimmed === "/diagnostics") {
        const d = await api("/api/diagnostics");
        showTyping(false);
        addMsg("bot", `Diagnostics **ok=${d.ok}** · passed **${d.passed}/${d.total}** · human_level_agi=false`);
      } else if (trimmed === "/completeness") {
        const d = await api("/api/completeness");
        showTyping(false);
        addMsg(
          "bot",
          `Completeness **ok=${d.ok}** · complete_code=**${d.complete_code}** · modules=${(d.results || []).length}`
        );
      } else if (trimmed === "/reset") {
        await api("/api/reset", {});
        showTyping(false);
        state.history = [];
        renderHistory();
        showWelcome();
        addMsg("bot", "Session reset. Ready for a new conversation.");
      } else {
        const data = await api("/api/chat", {
          message: trimmed,
          mode: modeSelect.value || "auto",
        });
        showTyping(false);
        addMsg("bot", data.reply || "(empty reply)", {
          route: data.route,
          llm: data.llm_at_runtime,
          confidence: data.confidence,
          proto_agi: data.proto_agi,
          thinking: data.thinking,
          thinkingBrief: data.thinking_brief,
          thinkingMarkdown: data.thinking_markdown,
        });
        state.history.push({ user: trimmed, bot: data.reply || "", route: data.route });
        renderHistory();
        if (Array.isArray(data.suggestions) && data.suggestions.length) {
          setSuggestions(data.suggestions);
        } else {
          setSuggestions([
            "Explain that simply",
            "Give more detail",
            "Search the web for related info",
          ]);
        }
      }
      await refreshStatus();
    } catch (e) {
      showTyping(false);
      addMsg("bot", `**Error:** ${e.message}`);
    } finally {
      state.busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(160, Math.max(44, input.scrollHeight)) + "px";
  }

  function openSidebar() {
    sidebar.classList.add("open");
    backdrop.hidden = false;
  }
  function closeSidebar() {
    sidebar.classList.remove("open");
    backdrop.hidden = true;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendText(input.value);
  });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText(input.value);
    }
  });

  document.querySelectorAll("[data-prompt]").forEach((btn) => {
    btn.addEventListener("click", () => sendText(btn.dataset.prompt));
  });
  document.querySelectorAll("[data-cmd]").forEach((btn) => {
    btn.addEventListener("click", () => sendText(btn.dataset.cmd));
  });

  $("btnNewChat").addEventListener("click", () => {
    state.history = [];
    renderHistory();
    showWelcome();
    closeSidebar();
    input.focus();
  });
  $("btnTheme").addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });
  $("btnOpenSidebar").addEventListener("click", openSidebar);
  $("btnCloseSidebar").addEventListener("click", closeSidebar);
  backdrop.addEventListener("click", closeSidebar);

  (async () => {
    showWelcome();
    try {
      const hello = await api("/api/hello");
      // Keep welcome; append greeting as first bot note under it optionally
      if (hello.greeting) {
        // soft status only
        statusLine.textContent = "Ready";
      }
      await refreshStatus();
    } catch (e) {
      addMsg("bot", `Could not reach Complete AI API: ${e.message}`);
    }
    input.focus();
  })();
})();
