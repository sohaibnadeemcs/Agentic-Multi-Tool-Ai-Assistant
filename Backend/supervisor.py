"""
Supervisor: the LangGraph orchestration layer.

This module defines the core StateGraph:

                        +------------------+
                        |  classify_intent  |
                        +------------------+
                                 |
                (conditional edge on state["intent"])
                 /       |        |        |        \\        \\
          research  code_fix  summarize  image_gen  rag_qa  general_chat
                \\       |        |        |        /        /
                        +------------------+
                                 |
                                END

`classify_intent` calls Groq with a strict instruction to output exactly one
of the six supported labels. Each sub-agent node below wraps the streaming
implementation in `backend/agents/*.py` so the compiled graph can be invoked
end-to-end (e.g. for tests, CLI usage, or non-streaming callers).

For the live token-by-token streaming used by the FastAPI `/chat` endpoint,
`ROUTES` exposes the same sub-agent generators directly — the endpoint calls
`classify_intent` once, then streams from the matching generator. This keeps
a single source of truth for intent classification and per-agent logic while
still allowing real HTTP streaming (LangGraph's own streaming API adds
complexity here that isn't needed for a linear supervisor -> single-agent
routing graph).
"""
from typing import Iterator, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from backend.logger import get_logger
from backend.llm import chat_completion
from backend.agents.research import run_research
from backend.agents.code_fix import run_code_fix
from backend.agents.summarize import run_summarize
from backend.agents.image_gen import run_image_gen
from backend.agents.rag_qa import run_rag_qa
from backend.agents.general_chat import run_general_chat

logger = get_logger(__name__)

VALID_INTENTS = [
    "research",
    "code_fix",
    "summarize",
    "image_gen",
    "rag_qa",
    "general_chat",
]

IntentType = Literal[
    "research", "code_fix", "summarize", "image_gen", "rag_qa", "general_chat"
]

CLASSIFIER_SYSTEM_PROMPT = (
    "You are an intent classification module. Given a user's message, decide "
    "which ONE of the following categories it belongs to. Respond with ONLY "
    "the category label, lowercase, nothing else — no punctuation, no "
    "explanation.\n\n"
    "Categories:\n"
    "- research: the user is asking a factual/current-events/multi-step "
    "question that likely benefits from a live web search.\n"
    "- code_fix: the user has pasted code and wants bugs found/fixed, or is "
    "describing a coding error to debug.\n"
    "- summarize: the user wants a block of text condensed into a summary.\n"
    "- image_gen: the user wants an image, picture, illustration, or artwork "
    "generated from a text description.\n"
    "- rag_qa: the user is asking a question about a PDF/document they "
    "uploaded (e.g. mentions 'the document', 'the PDF', 'the file I "
    "uploaded').\n"
    "- general_chat: anything conversational that doesn't clearly fit the "
    "above (greetings, opinions, small talk, general knowledge Q&A that "
    "doesn't need a live search).\n"
)


class AgentState(TypedDict, total=False):
    session_id: str
    message: str
    memory_context: List[str]
    intent: Optional[IntentType]
    response: str


# ---------------------------------------------------------------------------
# Node: intent classification
# ---------------------------------------------------------------------------
def classify_intent_node(state: AgentState) -> AgentState:
    intent = classify_intent(state["message"])
    logger.info("Classified intent=%s for session=%s", intent, state.get("session_id"))
    return {**state, "intent": intent}


def classify_intent(message: str) -> IntentType:
    """Standalone classifier — used by both the graph node and the streaming API."""
    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
        )
        cleaned = raw.strip().lower().strip(".\"'")
        if cleaned in VALID_INTENTS:
            return cleaned  # type: ignore
        logger.warning("Classifier returned unexpected label %r; defaulting to general_chat", raw)
        return "general_chat"
    except Exception:
        logger.exception("Intent classification failed; defaulting to general_chat")
        return "general_chat"


def _route_from_classification(state: AgentState) -> str:
    return state.get("intent", "general_chat")


# ---------------------------------------------------------------------------
# Sub-agent nodes (used for full graph.invoke() calls — collects the full
# streamed output into `response`). The FastAPI layer instead streams
# directly from the ROUTES generators for real-time token delivery.
# ---------------------------------------------------------------------------
def _collect(gen: Iterator[str]) -> str:
    return "".join(gen)


def research_node(state: AgentState) -> AgentState:
    text = _collect(run_research(state["message"], state.get("memory_context", [])))
    return {**state, "response": text}


def code_fix_node(state: AgentState) -> AgentState:
    text = _collect(run_code_fix(state["message"], state.get("memory_context", [])))
    return {**state, "response": text}


def summarize_node(state: AgentState) -> AgentState:
    text = _collect(run_summarize(state["message"], state.get("memory_context", [])))
    return {**state, "response": text}


def image_gen_node(state: AgentState) -> AgentState:
    text = _collect(run_image_gen(state["message"]))
    return {**state, "response": text}


def rag_qa_node(state: AgentState) -> AgentState:
    text = _collect(
        run_rag_qa(state["session_id"], state["message"], state.get("memory_context", []))
    )
    return {**state, "response": text}


def general_chat_node(state: AgentState) -> AgentState:
    text = _collect(run_general_chat(state["message"], state.get("memory_context", [])))
    return {**state, "response": text}


# ---------------------------------------------------------------------------
# Graph definition: nodes + conditional edges
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("research", research_node)
    graph.add_node("code_fix", code_fix_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("image_gen", image_gen_node)
    graph.add_node("rag_qa", rag_qa_node)
    graph.add_node("general_chat", general_chat_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_from_classification,
        {
            "research": "research",
            "code_fix": "code_fix",
            "summarize": "summarize",
            "image_gen": "image_gen",
            "rag_qa": "rag_qa",
            "general_chat": "general_chat",
        },
    )

    for node_name in VALID_INTENTS:
        graph.add_edge(node_name, END)

    return graph.compile()


# Compiled graph, ready for direct .invoke()/.stream() calls if needed
compiled_graph = build_graph()

# Streaming generator functions per intent, for the FastAPI live endpoint.
# Each takes (session_id, message, memory_context) for a uniform call signature.
ROUTES = {
    "research": lambda session_id, message, memory_context: run_research(message, memory_context),
    "code_fix": lambda session_id, message, memory_context: run_code_fix(message, memory_context),
    "summarize": lambda session_id, message, memory_context: run_summarize(message, memory_context),
    "image_gen": lambda session_id, message, memory_context: run_image_gen(message),
    "rag_qa": lambda session_id, message, memory_context: run_rag_qa(session_id, message, memory_context),
    "general_chat": lambda session_id, message, memory_context: run_general_chat(message, memory_context),
}
