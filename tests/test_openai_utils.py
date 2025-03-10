import os
from unittest.mock import MagicMock

import pytest

from src.utils.openai_budget_forcing import OpenAIUtils


@pytest.fixture
def mock_openai_utils():
    """Fixture to create an OpenAIUtils instance with mocked OpenAI API."""
    api_key = "test_key"
    base_url = "https://test.api"
    model_name = "test_model"

    os.environ['OPENAI_RETRY_SLEEP'] = '0.1'

    openai_utils = OpenAIUtils(api_key, base_url, model_name)
    openai_utils.client.chat.completions.create = MagicMock()

    return openai_utils


def test_generate_thinking(mock_openai_utils):
    """Test thinking phase response generation."""
    mock_openai_utils.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Thinking response..."))]
    )

    messages = [{"role": "user", "content": "Test prompt"}]
    response = mock_openai_utils.generate_thinking(messages)

    assert "Thinking response..." in response


def test_generate_final_answer(mock_openai_utils):
    """Test final answer generation."""
    mock_openai_utils.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Final response."))]
    )

    messages = [{"role": "user", "content": "Test prompt"}]
    response = mock_openai_utils.generate_final_answer(messages, "Thinking content")

    assert response == "Final response."


def test_process_prompt(mock_openai_utils):
    """Test full prompt processing workflow."""
    mock_openai_utils.client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Thinking response..."))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Thinking response..."))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Final response."))])
    ]

    thinking, final_answer = mock_openai_utils.process_prompt("Test prompt", 1)

    assert "Thinking response..." in thinking
    assert final_answer == "Final response."


def test_generate_final_answer_max_attempts(mock_openai_utils):
    """Test handling of failed final answer attempts."""
    mock_openai_utils.client.chat.completions.create.side_effect = Exception("API failure")

    messages = [{"role": "user", "content": "Test prompt"}]
    response = mock_openai_utils.generate_final_answer(messages, "Thinking content")

    assert response is None
