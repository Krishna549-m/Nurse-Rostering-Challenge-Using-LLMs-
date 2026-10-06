from typing import Optional
from anthropic import Anthropic

API_KEY = ""
MODEL = "claude-opus-4-8"
TEMPERATURE = 0.2
TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 2

def create_client() -> Anthropic:
    return Anthropic(
        api_key=API_KEY,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )

def call_llm(prompt: str) -> str:
    client = create_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=20000,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "".join(text_blocks)