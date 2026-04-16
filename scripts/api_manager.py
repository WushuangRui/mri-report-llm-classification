"""
api_manager.py
Manages Azure OpenAI API client setup and rate-limited, retry-safe calls.
"""

import os
import time
import logging
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)


def get_client() -> AzureOpenAI:
    """Initialise Azure OpenAI client from environment variables."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key:
        raise EnvironmentError("AZURE_OPENAI_API_KEY is not set.")
    if not endpoint:
        raise EnvironmentError("AZURE_OPENAI_ENDPOINT is not set.")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_api(client: AzureOpenAI, deployment: str, messages: list, temperature: float = 0.0) -> str:
    """
    Send a chat completion request with automatic retry on transient errors.

    Args:
        client:      Initialised AzureOpenAI client.
        deployment:  Azure deployment name (e.g. "gpt-4o").
        messages:    List of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature (default 0 for deterministic output).

    Returns:
        The assistant message content as a string.
    """
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
