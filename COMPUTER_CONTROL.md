# Detailed full computer control

Gated **local** desktop + filesystem automation for Complete AI / Finderbay Assistant.

> Controls **your own machine** only.  
> **No keylogging. No background clipboard monitoring. No remote unauthorized access.**  
> `human_level_agi=false`

---

## Enable

```bash
# Required for any side-effect action
export COMPLETE_AI_COMPUTER_CONTROL=1

# Optional: allow allowlisted shell (still needs confirm=True)
export COMPLETE_AI_COMPUTER_SHELL=1

# Optional: extra sandbox roots (pathsep-separated)
export COMPLETE_AI_COMPUTER_ROOT="$HOME/Projects:$HOME/Documents"
```

Or in code:

```python
from cognitive_ai.assistant import FinderbayAssistant
a = FinderbayAssistant(allow_computer_control=True, allow_shell=False)
a.handle("computer help")
```

---

## What you can do

| Category | Actions |
|----------|---------|
| Meta | `status` `help` `policy` `env` `screen` `cwd` `audit` `enable`/`disable` |
| Open / launch | `open_url` `open_path` `launch calculator\|browser\|terminal\|text_editor\|file_manager` |
| Files (sandboxed) | `list_dir` `read_file` `write_file` `mkdir` `copy` `move` `delete` |
| Desktop | `screenshot` `notify` `clipboard_get`/`set` *(explicit, confirm)* |
| Input *(confirm)* | `type` `press` `hotkey` `click` `mouse_move` `scroll` `wait` |
| Shell *(extra gate)* | `run_command` + denylist (`rm -rf /`, `mkfs`, shutdown, …) |

### Natural language examples

```text
computer help
computer status
computer open https://example.com
computer screenshot
computer list files in data
computer read file README.md
computer write_file data/note.txt :: hello from Complete AI
computer mkdir data/tmp_demo
computer copy data/note.txt to data/note2.txt
computer launch calculator
computer notify Build finished
computer type Hello world          # confirm implied via CLI path
computer press ctrl+s
computer click 120 240
computer env
computer screen
computer processes
computer policy
computer audit
computer run echo hello            # needs COMPLETE_AI_COMPUTER_SHELL=1
```

Assistant / API / tools:

```bash
python -m cognitive_ai assistant --ask "computer help"
# HTTP POST /computer  {"action":"list_dir","path":"data"}
# TOOL computer {"action":"screenshot"}
```

---

## Safety model

1. **Off by default** — refuses with `computer_control_disabled`  
2. **Allowlist only** — unknown actions rejected  
3. **Sandbox** — reads/writes must resolve under roots (cwd, `data/`, Documents/Downloads/Desktop, `COMPLETE_AI_COMPUTER_ROOT`)  
4. **Forbidden prefixes** — `/etc`, `/boot`, `C:\Windows`, …  
5. **Confirm required** for delete/write/input/clipboard/shell/move  
6. **Shell denylist** even when shell enabled  
7. **Full audit log** — `computer audit`  
8. **Clipboard** is one-shot explicit get/set only — never a keylogger or background siphon  

---

## Optional Python deps

```bash
pip install Pillow pyautogui pyperclip psutil
# screenshots / typing / clipboard / process list
```

Without them, related actions return clear `*_unavailable` errors (never fake success).

---

## Verify

```bash
export COMPLETE_AI_COMPUTER_CONTROL=1
python - <<'PY'
from cognitive_ai.assistant.computer import ComputerControl
cc = ComputerControl(allow=True)
print(cc.run("help")["result"]["examples"][:3])
print(cc.handle_text("computer list files in data")["ok"])
print(cc.run("open_url", url="https://example.com")["ok"])
PY
pytest tests/test_computer_control_v123.py -q
```
