import asyncio
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import hooks

# ==========================================
# BIMExtract: Heavy Preprocessing Agent
# ==========================================

@hooks.on_session_start
async def on_start():
    print("[BIMExtract] Initializing Contextual Enrichment and PISCO compression pipelines.")

@hooks.pre_tool_call_decide
async def pre_tool(data: types.ToolCall) -> types.HookResult:
    print(f"[BIMExtract] Authorizing VLM parsing tool: {data.name}")
    return types.HookResult(allow=True)

async def run_preprocessing_agent():
    config = LocalAgentConfig(
        capabilities=types.CapabilitiesConfig(enable_subagents=True, enable_tools=True),
        skills_paths=["./skills"],
        hooks=[on_start, pre_tool]
    )

    async with Agent(config) as agent:
        print("[BIMExtract] Ingestion Engine Ready.")
        # Goal: Execute heavy preprocessing moat
        response = await agent.chat(
            "Use subagents to ingest documents via PaddleOCR-VL and Docling. "
            "Pass the chunks to the Contextual Enrichment skill and pre-materialize the KV-cache."
        )
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(run_preprocessing_agent())
