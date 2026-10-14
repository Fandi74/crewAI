"""Tests for Sign in with ChatGPT request constraints and local LLM routing."""

import pytest
from pydantic import ValidationError

from crewai import LLM
from crewai.llms.providers.openai.completion import OpenAICompletion


def build_oauth_llm(**kwargs) -> OpenAICompletion:
    return OpenAICompletion(model="gpt-6.1-sol", chatgpt_oauth=True, **kwargs)


def test_chatgpt_oauth_llm_initializes_without_api_key_or_saved_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    llm = build_oauth_llm()

    assert llm.api == "responses"
    assert llm.store is False
    assert llm.stream is True
    assert llm._client is None
    assert llm._async_client is None


def test_chatgpt_oauth_uses_public_responses_endpoint_and_never_serializes_token() -> None:
    llm = build_oauth_llm()

    params = llm._get_client_params(access_token="oauth-test-token")

    assert params["api_key"] == "oauth-test-token"
    assert params["base_url"] == "https://api.openai.com/v1"
    assert params["max_retries"] == 0
    assert "oauth-test-token" not in repr(llm.to_config_dict())


def test_chatgpt_oauth_strips_unsupported_responses_fields() -> None:
    llm = build_oauth_llm(temperature=0.2, top_p=0.5, max_tokens=128)

    params = llm._prepare_responses_params(
        messages=[{"role": "user", "content": "hello"}]
    )

    assert params["stream"] is True
    assert params["store"] is False
    assert "temperature" not in params
    assert "top_p" not in params
    assert "max_output_tokens" not in params


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"api": "completions"}, "Responses API"),
        ({"base_url": "https://proxy.example/v1"}, "only supports"),
        ({"store": True}, "store=False"),
        ({"client_params": {"http_client": object()}}, "Custom HTTP clients"),
    ],
)
def test_chatgpt_oauth_rejects_incompatible_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        build_oauth_llm(**kwargs)


def test_chatgpt_oauth_routes_model_slugs_to_native_openai() -> None:
    llm = LLM(model="openai/gpt-6.1-sol", chatgpt_oauth=True)

    assert isinstance(llm, OpenAICompletion)
    assert llm.chatgpt_oauth is True
    assert llm.api == "responses"
