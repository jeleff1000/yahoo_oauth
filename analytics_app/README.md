### KMFFL Stats

This is the project that contains scripts for the stats of the Kemp Mill Fantasy Football League

### Summary



### Migration to Claude Agent SDK

This repository has been prepared to use the renamed Claude Agent SDK (replacement for the older Claude Code SDK).

- Python dependency added to `streamlit_ui/requirements.txt`: `claude-agent-sdk==0.1.0`.
- An example helper was added at `streamlit_ui/tools/claude_agent_example.py` showing the new imports and usage pattern.

Notes:
- The Agent SDK uses `ClaudeAgentOptions` (replaces `ClaudeCodeOptions`).
- The SDK no longer applies the old system prompt by default. If you want the previous behaviour, set `systemPrompt` in the options to `{ type: "preset", preset: "claude_code" }`.

How to install the new dependency into your virtualenv (example):

```cmd
cd streamlit_ui
python -m pip install -r requirements.txt
```

Or install the new package only:

```cmd
python -m pip install claude-agent-sdk==0.1.0
```
