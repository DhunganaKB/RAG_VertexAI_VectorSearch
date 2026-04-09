"""
Airbnb Rental Agent — Gemini Native Function Calling
=====================================================
An agentic loop built on Vertex AI Gemini's native function-calling capability.

Why NOT google-adk directly:
  google-adk 1.x depends on google-genai which conflicts with pydantic v2
  in the mcp environment. We get identical agentic behaviour using Vertex AI's
  GenerativeModel.Tool / FunctionDeclaration primitives directly.

Architecture:
  User query
      ↓
  Gemini 2.5 Flash  (with find_rentals function declaration)
      ↓ [if the model wants to call a tool]
  find_rentals(query, filters...)   ← VS2.0 semantic search
      ↓
  Tool result fed back to Gemini
      ↓ [repeat until no more tool calls]
  Final natural-language response

The agent can make multiple tool calls in one conversation turn if it
decides to refine the search (e.g. first try, then broaden/narrow).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

from agent.tools import find_rentals

# ── Agent system instruction ───────────────────────────────────────────────────
AGENT_INSTRUCTION = """You are an expert Airbnb rental assistant for Austin, Texas.
Your goal is to help users find the perfect short-term rental that matches their needs.

You have access to a search tool called find_rentals that queries a database of
real Airbnb listings in Austin using semantic vector search.

Available tool parameters:
  - query:            Natural language description (e.g. "cozy loft near music venues")
  - max_price:        Maximum price per night (USD)
  - min_price:        Minimum price per night (USD)
  - room_type:        "Entire home/apt", "Private room", "Shared room", "Hotel room"
  - min_bedrooms:     Minimum number of bedrooms (integer)
  - neighbourhood:    Zip code or neighborhood name (e.g. "78702", "East Downtown")
  - instant_bookable: true to only show instantly bookable listings
  - accommodates:     Minimum guests the listing must accommodate (integer)
  - top_k:            How many results to return (default 8, max 20)

Guidelines:
1. Extract constraints from the user's message and pass them as filters.
2. Always include a descriptive query so semantic search works well.
3. If the first search returns no results, retry with relaxed filters.
4. Present results clearly: name, location, price, size, rating, URL.
5. Highlight the best 2-3 options and explain why they match the request.
6. If the user asks follow-up questions, refine the search accordingly.
7. Be conversational and helpful — you're an expert local guide.
"""

# ── Function declaration for Gemini ───────────────────────────────────────────
FIND_RENTALS_DECLARATION = FunctionDeclaration(
    name="find_rentals",
    description=(
        "Search for Airbnb rentals in Austin, TX using semantic vector search "
        "and optional metadata filters. Returns a list of matching listings."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of the ideal rental "
                    "(e.g. 'quiet studio near downtown', 'family-friendly house with pool')"
                ),
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price per night in USD",
            },
            "min_price": {
                "type": "number",
                "description": "Minimum price per night in USD",
            },
            "room_type": {
                "type": "string",
                "description": "Type of room",
                "enum": ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
            },
            "min_bedrooms": {
                "type": "integer",
                "description": "Minimum number of bedrooms",
            },
            "neighbourhood": {
                "type": "string",
                "description": "Neighborhood name or zip code (e.g. '78702', 'East Downtown')",
            },
            "instant_bookable": {
                "type": "boolean",
                "description": "Set to true to only return instantly bookable listings",
            },
            "accommodates": {
                "type": "integer",
                "description": "Minimum number of guests the listing must accommodate",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 20)",
            },
        },
        "required": ["query"],
    },
)

FIND_RENTALS_TOOL = Tool(function_declarations=[FIND_RENTALS_DECLARATION])


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> str:
    """Route a function call to the correct Python function."""
    if name == "find_rentals":
        return find_rentals(**args)
    return f"Unknown tool: {name}"


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    max_turns: int = 5,
) -> tuple[str, list[dict]]:
    """
    Run the Airbnb agent for a single user query.

    Args:
        query:     The user's natural-language request.
        max_turns: Maximum agentic loop iterations (prevents infinite loops).

    Returns:
        (answer, tool_calls_log)
        - answer:         Final natural-language response from the agent.
        - tool_calls_log: List of dicts describing each tool call made.
    """
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)

    model = GenerativeModel(
        model_name=config.GEMINI_MODEL,
        tools=[FIND_RENTALS_TOOL],
        system_instruction=AGENT_INSTRUCTION,
        generation_config=GenerationConfig(
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        ),
    )

    # response_validation=False: Gemini 2.5 function-call turns can have
    # finish_reason=2 which the SDK incorrectly treats as a safety block.
    # Disabling validation lets the agentic loop handle these normally.
    chat            = model.start_chat(response_validation=False)
    response        = chat.send_message(query)
    tool_calls_log: list[dict] = []

    # ── Agentic loop ──────────────────────────────────────────────────────────
    for _turn in range(max_turns):
        if not response.candidates:
            break

        candidate  = response.candidates[0]
        fn_calls   = [
            part.function_call
            for part in candidate.content.parts
            if (
                hasattr(part, "function_call")
                and part.function_call is not None
                and getattr(part.function_call, "name", None)
            )
        ]

        if not fn_calls:
            break   # No more tool calls — model has produced the final answer

        # Execute each function call and collect results
        result_parts: list[Part] = []
        for fc in fn_calls:
            args   = dict(fc.args)
            result = _execute_tool(fc.name, args)

            tool_calls_log.append({
                "tool":   fc.name,
                "args":   args,
                "result": result[:300] + "…" if len(result) > 300 else result,
            })

            result_parts.append(
                Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

        # Feed results back to the model
        response = chat.send_message(result_parts)

    # ── Extract final text ────────────────────────────────────────────────────
    final_text = ""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                final_text += part.text

    return final_text.strip(), tool_calls_log


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = (
        sys.argv[1] if len(sys.argv) > 1
        else "Find me a cozy private room under $80 per night in Austin"
    )
    print(f"\nQuery: {test_query}\n{'─' * 60}")
    answer, calls = run_agent(test_query)
    print("\n[Tool calls made]")
    for c in calls:
        print(f"  → {c['tool']}({json.dumps({k: v for k, v in c['args'].items() if k != 'query'}, indent=None)})")
    print(f"\n[Agent response]\n{answer}")
