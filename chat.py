"""
Interactive chat application for experimenting with Gen-AI models via OpenAI API.

Usage:
    python chat.py

Environment variables:
    OPENAI_API_KEY  - Your OpenAI API key (required)
    OPENAI_MODEL    - Model to use (default: gpt-4o-mini)
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def chat(model: str | None = None) -> None:
    """Run an interactive chat session with the specified model."""
    client = get_client()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    conversation: list[dict[str, str]] = []

    print(f"Chat experiment — model: {model}")
    print("Type your message and press Enter. Type 'exit' or 'quit' to end the session.")
    print("Type 'reset' to clear the conversation history.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        if user_input.lower() == "reset":
            conversation.clear()
            print("Conversation history cleared.")
            continue

        conversation.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error communicating with API: {exc}")
            conversation.pop()  # remove the failed user message
            continue

        assistant_message = response.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": assistant_message})

        print(f"\nAssistant: {assistant_message}")


if __name__ == "__main__":
    chat()
