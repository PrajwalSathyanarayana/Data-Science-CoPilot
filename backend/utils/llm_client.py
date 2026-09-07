"""Utility — thin wrapper around the Claude API with model selection, retries, and max_tokens=1000."""
import anthropic
import dotenv
import time
import logging

dotenv.load_dotenv()

client = anthropic.Anthropic()


logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Custom exception for LLM call errors."""


def call_llm(system_prompt: str, user_message: str, model: str, max_tokens: int = 1000) -> str:

    for attempt in range(3):
        try:
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

        except anthropic.RateLimitError:
            logger.warning("A 429 status code was received; we should back off a bit.")
            time.sleep(2 ** attempt)
            continue
        except anthropic.APIConnectionError as e:
            logger.error("The server could not be reached")
            logger.error(e.__cause__) 
            time.sleep(2 ** attempt)
            continue
        except anthropic.APIStatusError as e:
            logger.error("Another non-200-range status code was received")
            logger.error(e.status_code)
            logger.error(e.response)
            time.sleep(2 ** attempt)
            continue

    raise LLMCallError("Error: Failed to get a response from the LLM after 3 attempts.")
