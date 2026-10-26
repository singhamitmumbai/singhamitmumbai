(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusPill = document.getElementById("statusPill");
  const mindStats = document.getElementById("mindStats");

  function addMsg(role, text) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "user" ? "You" : "Brain";
    el.appendChild(who);
    el.appendChild(document.createTextNode(text));
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
      statusPill.textContent = `${st.domain} · ${st.facts} facts · goals ${st.open_goals}`;
      mindStats.innerHTML = `
        <dt>domain</dt><dd>${st.domain}</dd>
        <dt>facts</dt><dd>${st.facts}</dd>
        <dt>beliefs</dt><dd>${st.beliefs}</dd>
        <dt>rules</dt><dd>${st.rules}</dd>
        <dt>open goals</dt><dd>${st.open_goals}</dd>
        <dt>cycles</dt><dd>${st.think_cycles}</dd>
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
      const data = await api("/api/chat", { message: trimmed });
      addMsg("bot", data.reply || "(empty)");
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
  document.querySelectorAll("[data-domain]").forEach((btn) => {
    btn.addEventListener("click", () => sendText(`domain ${btn.dataset.domain}`));
  });

  (async () => {
    try {
      const hello = await api("/api/hello");
      addMsg("bot", hello.greeting);
      await refreshStatus();
    } catch (e) {
      addMsg("bot", `Could not reach brain: ${e.message}`);
    }
  })();
})();
