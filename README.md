# Gen-AI

Interactive chat application for experimenting with generative AI models.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**

   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. *(Optional)* **Choose a model** (defaults to `gpt-4o-mini`)

   ```bash
   export OPENAI_MODEL=gpt-4o
   ```

## Usage

```bash
python chat.py
```

- Type a message and press **Enter** to send it.
- Type `reset` to clear the conversation history and start fresh.
- Type `exit` or `quit` (or press **Ctrl+C**) to end the session.

## Example

```
Chat experiment — model: gpt-4o-mini
Type your message and press Enter. Type 'exit' or 'quit' to end the session.
Type 'reset' to clear the conversation history.
------------------------------------------------------------

You: What is a large language model?
Assistant: A large language model (LLM) is a type of AI trained on vast amounts of text...

You: exit
Session ended.
```
