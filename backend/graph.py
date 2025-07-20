"""Define a custom Reasoning and Action agent.

This sets up a ReAct-style LangGraph agent with tool support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from backend.configuration import Configuration
from backend.state import InputState, State
from backend.tools import TOOLS
from backend.utils import load_chat_model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM and bind it to the available tools."""
    configuration = Configuration.from_context()

    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not complete your request in the given steps.",
                )
            ]
        }

    return {"messages": [response]}


# --- Graph Construction ---

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Register core nodes
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point
builder.add_edge("__start__", "call_model")

# Conditional logic: tool call or end?
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "__end__"

builder.add_conditional_edges("call_model", route_model_output)

# Loop back to model after tool use
builder.add_edge("tools", "call_model")

# Compile the graph
graph = builder.compile(name="ReAct Agent")

# Confirm tools loaded
print("Registered tools:", [tool.name for tool in TOOLS])
