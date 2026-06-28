"""LLM API clients used across experiments.

Currently provides one helper: an OpenAI-compatible client pointed at
NVIDIA NIM (https://build.nvidia.com/). Anything else (Anthropic, OpenAI,
self-hosted vLLM) is constructed directly inside the notebook that needs it.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# NVIDIA NIM
# -----------------------------------------------------------------------------

# Hosted NIM catalog endpoint. Override via $NIM_BASE_URL to point at a
# self-hosted NIM container (e.g. http://my-nim:8000/v1).
NIM_BASE_URL_DEFAULT = "https://integrate.api.nvidia.com/v1"

# Small curated list of NIM models that have been around long enough to be
# stable references. Edit freely — the catalog at build.nvidia.com moves.
NIM_DEFAULT_MODELS: list[str] = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "qwen/qwen2.5-7b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]


def _read_nim_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
    if not key:
        raise RuntimeError(
            "No NIM API key found. Set NVIDIA_API_KEY (preferred) or "
            "NIM_API_KEY in your env before calling get_nim_client()."
        )
    return key


def get_nim_client(*, base_url: str | None = None):
    """Return an OpenAI-compatible client pointed at NVIDIA NIM.

    NIM exposes a `/v1/chat/completions` endpoint that's wire-compatible with
    the OpenAI SDK, so we just construct an `openai.OpenAI` with the NIM
    base URL and API key swapped in.

    Args:
        base_url: Override the endpoint. Defaults to $NIM_BASE_URL if set,
            otherwise the hosted NVIDIA endpoint.

    Returns:
        An `openai.OpenAI` instance ready to call `chat.completions.create`.
    """
    from openai import OpenAI

    if base_url is None:
        base_url = os.environ.get("NIM_BASE_URL", NIM_BASE_URL_DEFAULT)
    return OpenAI(base_url=base_url, api_key=_read_nim_api_key())
