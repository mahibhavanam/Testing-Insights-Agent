from typing import List, Optional
from openai import OpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class OpenAIChat:
    """
    Minimal wrapper that lets you use OpenAI chat like `chat.invoke([...messages])`
    similar to your ProxiedGPTChat usage.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 1800):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages: List[BaseMessage]) -> HumanMessage:
        payload = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "user"
            payload.append({"role": role, "content": m.content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return HumanMessage(content=text)
