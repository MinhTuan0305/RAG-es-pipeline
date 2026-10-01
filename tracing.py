"""
Optional Langfuse tracing for the agent.

Tracing is enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and
the integration imports cleanly (it needs both `langfuse` and `langchain` installed).
Otherwise every helper here is a no-op and the app runs exactly as without tracing.
"""

import logging
import os

from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

_warned = False


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def make_callback_handler():
    """A fresh Langfuse handler for ONE run (the handler keeps per-run bookkeeping,
    so it isn't shared across concurrent questions), or None if tracing is off."""
    global _warned
    if not tracing_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as e:
        if not _warned:
            logger.warning("Langfuse keys are set but tracing is disabled: %s", e)
            _warned = True
        return None
    return CallbackHandler()


def flush():
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


def traced_step(name: str, inputs: dict, fn, summarize):
    """Run `fn()` as a named child step of the current run, so it shows up as its own
    span in the trace with real timing.

    The span records `inputs` and `summarize(result)` -- small, readable dicts -- rather
    than the raw objects (full Elasticsearch hits with whole passages), while the
    caller still gets the full, unsummarized result back. Outside a traced run this is
    just a plain function call.

    No config is passed on purpose: while a tool executes, LangChain puts the tool's
    own run config into a context variable, and invoke() without a config picks it up,
    nesting the step under the tool. The `config` argument a tool function receives is
    its *parent's* config -- passing that here would attach the step to the parent.
    """
    holder = {}

    def run(_inputs):
        holder["result"] = fn()
        return summarize(holder["result"])

    RunnableLambda(run, name=name).invoke(inputs)
    return holder["result"]
