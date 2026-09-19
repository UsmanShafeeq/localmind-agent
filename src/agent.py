"""Agentic execution loop using LangGraph + Ollama."""

from functools import lru_cache
from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT_PATH,
    RAG_PROMPT_PATH,
    load_prompt,
)
from src.tools import ALL_TOOLS
from src.vectorstore import search


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _build_graph():
    llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)

    def agent_node(state: AgentState) -> AgentState:
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=system_prompt), *msgs]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        return "tools" if tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(question: str, max_turns: int = 8) -> str:
    """Execute the ReAct loop and return the final answer text."""
    app = _build_graph()
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": max_turns * 2},
    )
    final = result["messages"][-1]
    return final.content if isinstance(final.content, str) else str(final.content)


def synthesize_answer(question: str, k: int = 4) -> str:
    """Retrieve chunks then synthesize a grounded answer using rag_prompt.txt."""
    docs = search(question, k=k)
    if not docs:
        return "The provided documents do not contain sufficient information to answer this query."

    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}"
        + (
            f" p.{d.metadata['page'] + 1}"
            if isinstance(d.metadata.get("page"), int)
            else ""
        )
        + f"]\n{d.page_content.strip()}"
        for d in docs
    )

    prompt = load_prompt(RAG_PROMPT_PATH).format(context=context, question=question)
    response = get_llm().invoke([HumanMessage(content=prompt)])
    return (
        response.content if isinstance(response.content, str) else str(response.content)
    )
