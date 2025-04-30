"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""
import os
print("API KEY from env:", os.getenv("OPENAI_API_KEY"))

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS  # <- this picks up your resume generator tool
from react_agent.utils import load_chat_model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our 'agent'.

    This function prepares the prompt, initializes the model, and processes the response.
    """
    configuration = Configuration.from_context()

    # Initialize the model and bind it to the available tools
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt with current timestamp
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Invoke the model with current chat state
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # If it's the last step and tool calls still exist, exit with a fallback message
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response
    return {"messages": [response]}


# --- Build the agent's state graph ---

# Initialize graph with state definition
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the model node and the tool node
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Start with the model
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Decide whether to finish or call a tool, based on model output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # Finish if no tool call requested
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise call tools
    return "tools"


# Conditionally transition from model output
builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

# Always return to model after tool execution
builder.add_edge("tools", "call_model")

# Compile graph
graph = builder.compile(name="ReAct Agent")
