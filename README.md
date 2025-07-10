# Finder Agent
This is a minimal example of using MCP Protocol for searching file system and browsing the internet.

Adapted from: [mcp-basic-agent](https://github.com/lastmile-ai/mcp-agent/tree/main/examples/basic/mcp_basic_agent)

Modifications:
- Using Local Ollama instead of OpenAI/Anthropic API Calls
- Adding a minimal streamlit chat interface

# Ollama Server
You need to install [ollama](https://github.com/ollama/ollama) and download the [qwen3:8b](https://ollama.com/library/qwen3:8b) weights.

Run:
- `ollama serve`, to start the ollama Server
- `uv pip install -r requirements.txt`
- `streamlit run main.py`
