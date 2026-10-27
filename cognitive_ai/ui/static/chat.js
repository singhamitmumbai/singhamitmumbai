(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusPill = document.getElementById("statusPill");
  const mindStats = document.getElementById("mindStats");

  function addMsg(role, text, meta) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "user" ? "You" : "Complete AI";
    el.appendChild(who);
    el.appendChild(document.createTextNode(text));
    if (meta && meta.route) {
      const tag = document.createElement("div");
      tag.className = "hint";
      tag.textContent = `route=${meta.route}` + (meta.llm ? " · llm" : "");
      el.appendChild(tag);
    }
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      const name = st.name || "Complete AI";
      const ver = st.version || "";
      const llm = (st.llm_at_runtime || st.chatgpt_class_general_chat) ? "llm:on" : "llm:off";
      statusPill.textContent = `${name} ${ver} · ${llm}`;
      const modes = st.modes || {};
      mindStats.innerHTML = `
        <dt>product</dt><dd>${name}</dd>
        <dt>version</dt><dd>${ver}</dd>
        <dt>chat</dt><dd>${modes.general_chat || "?"}</dd>
        <dt>session</dt><dd>${st.session_turns ?? "—"}</dd>
        <dt>tools</dt><dd>${(st.tools || []).join(", ") || "—"}</dd>
        <dt>agi</dt><dd>${st.human_level_agi === false ? "false" : st.human_level_agi}</dd>
      `;
    } catch (e) {
      statusPill.textContent = "offline";
    }
  }

  async function sendText(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    addMsg("user", trimmed);
    input.value = "";
    sendBtn.disabled = true;
    try {
      if (trimmed === "/status") {
        const st = await api("/api/status");
        addMsg("bot", JSON.stringify(st, null, 2).slice(0, 2000));
      } else if (trimmed === "/diag" || trimmed === "/diagnostics") {
        const d = await api("/api/diagnostics");
        addMsg("bot", `diagnostics ok=${d.ok} passed=${d.passed}/${d.total}`);
      } else if (trimmed === "/reset") {
        await api("/api/reset", {});
        addMsg("bot", "Session reset.");
      } else {
        const data = await api("/api/chat", { message: trimmed });
        addMsg("bot", data.reply || "(empty)", {
          route: data.route,
          llm: data.llm_at_runtime,
        });
      }
      await refreshStatus();
    } catch (e) {
      addMsg("bot", `Error: ${e.message}`);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendText(input.value);
  });

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

  (async () => {
    try {
      const hello = await api("/api/hello");
      addMsg("bot", hello.greeting);
      await refreshStatus();
    } catch (e) {
      addMsg("bot", `Could not reach Complete AI: ${e.message}`);
    }
  })();
})();
