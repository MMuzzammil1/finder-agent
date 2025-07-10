import asyncio
import os
import time
import streamlit as st

from mcp import ListToolsResult
from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM


# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="mcp_basic_agent")  # settings=settings)

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ""
    for tool in list_tools_result.tools:
        res += f"- **{tool.name}**: {tool.description}\n\n"
    return res

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        async with finder_agent:
            llm = await finder_agent.attach_llm(OllamaAugmentedLLM)

            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "How can I help you?"}
                ]

            logger.info("finder: Connected to server, calling list_tools...")

            if "tools_str" not in st.session_state:
                tools = await finder_agent.list_tools()
                st.session_state["tools_str"] = format_list_tools_result(tools)
            with st.expander("View Tools"):
                st.markdown(st.session_state["tools_str"])
            result = await finder_agent.list_tools()

            for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Type your message here..."):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                with st.chat_message("assistant"):
                    response = ""
                    with st.spinner("Thinking..."):
                        response = await llm.generate_str(
                            message=prompt,
                            request_params=RequestParams(
                                use_history=True,
                                parallel_tool_calls=False,
                            )
                        )
                        
                        placeholder = st.empty()

                        placeholder.write(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                                    
                st.rerun()


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")