"""Utility — thin wrapper around the Claude API with model selection, retries, and max_tokens=1000."""
import anthropic
import dotenv
dotenv.load_dotenv()

client = anthropic.Anthropic()

def call_llm(system_prompt: str, user_message: str, model: str, max_tokens: int = 1000) -> str:
    response = client.messages.create(
        model = model,
        max_tokens = max_tokens,
        system = system_prompt,
        messages = [{
            'role': 'user',
            'content': user_message,
        }]
    )

    return response.content[0].text