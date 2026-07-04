"""
The agent under test.

DESIGN NOTE: A deliberately minimal tool-use loop. Why minimal?
  - Eval harnesses measure the AGENT, but they're really measuring
    (agent + scaffolding). A flashy scaffolding can mask a weak model;
    a weak scaffolding can mask a strong one. The cleanest experimental
    design fixes scaffolding at a known baseline.
  - Real scaffoldings (react, plan-and-execute, multi-agent) deserve their
    own eval comparing scaffold vs scaffold with the same model.

This loop is roughly the standard agentic pattern:
  1. Send messages to model with tool schemas
  2. If model returns tool_use blocks, execute them and append results
  3. If model returns just text (no tool_use), stop
  4. Hard cap on iterations to prevent infinite loops
"""
import os
import time
import anthropic
from environment import Environment, TOOL_SCHEMAS
from mock_llm import MockClient


def _get_client(mock_policy: str = None):
    """Return real Anthropic client, or a mock if no API key / policy override."""
    if mock_policy or not os.environ.get("ANTHROPIC_API_KEY"):
        return MockClient(policy=mock_policy or "competent")
    return anthropic.Anthropic()


SYSTEM_PROMPT = """You are a customer support agent for an online retailer.
You help customers with their orders, including processing refunds when appropriate.

You have access to tools to look up orders, check policy, and issue refunds.
Always verify policy before issuing refunds. Be professional and concise.
"""


def run_agent(user_message: str, model: str, max_steps: int = 10,
              mock_policy: str = None) -> dict:
    """Run one agent rollout. Returns a structured trace.

    The return dict is deliberately rich — every metric the harness needs
    is captured here. This is the 'instrumentation' principle: log everything
    at agent run-time, decide what to score at grading time.

    If `mock_policy` is set, or no ANTHROPIC_API_KEY is available, uses the
    mock LLM (see mock_llm.py). Useful for CI and harness self-tests.
    """
    client = _get_client(mock_policy)
    env = Environment()
    messages = [{"role": "user", "content": user_message}]

    # Tracking
    start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    steps = 0
    stop_reason = None
    error = None

    try:
        while steps < max_steps:
            steps += 1
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                stop_reason = "end_turn"
                break

            if response.stop_reason == "tool_use":
                # Execute every tool_use block in the response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_fn = getattr(env, block.name, None)
                        if tool_fn is None:
                            result = {"ok": False, "error": f"Unknown tool: {block.name}"}
                        else:
                            try:
                                result = tool_fn(**block.input)
                            except TypeError as e:
                                # Catches parameter-blind selection failures
                                result = {"ok": False, "error": f"Bad args: {e}"}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Any other stop reason
            stop_reason = response.stop_reason
            break
        else:
            # While loop exited without break → hit max_steps
            stop_reason = "max_steps_exceeded"

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        stop_reason = "error"

    elapsed = time.time() - start

    return {
        "user_message": user_message,
        "model": model,
        "messages": messages,
        "env_state": {
            "tool_calls": env.state.tool_calls,
            "refunds_issued": env.state.refunds_issued,
        },
        "env": env,  # for graders to inspect
        "metrics": {
            "steps": steps,
            "stop_reason": stop_reason,
            "elapsed_seconds": elapsed,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "error": error,
        },
    }
