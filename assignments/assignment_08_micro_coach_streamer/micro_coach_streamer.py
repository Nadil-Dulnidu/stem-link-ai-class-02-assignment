"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

class PrintTokens(BaseCallbackHandler):
    """Minimal callback-like interface for printing tokens.

    Implement compatibility with LangChain callback protocol if desired.
    """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")


class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders.

        Provide:
        - `system_prompt` motivating but practical tone
        - `user_prompt` with variables {goal}, {time_available}
        - `self.llm_streaming` and `self.llm_plain` placeholders (None), with TODOs
        - `self.stream_prompt` and `self.plain_prompt` placeholders (None), with TODOs
        """
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic and brief."
        )
        self.user_prompt = "Goal: {goal}\nTime: {time_available}\nReturn a 3-step plan."

        # TODO: Build prompts and LLMs (streaming and non-streaming)
        self.llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[PrintTokens()])
        self.llm_plain = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.stream_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])
        self.plain_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])
        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path.

        Implement:
        - If `stream=True`, attach a token printer callback and stream output.
        - Else, return a compact non-streamed plan string.
        """
        if stream:
            return self.stream_chain.invoke({"goal": goal, "time_available": time_available})
        else:
            return self.plain_chain.invoke({"goal": goal, "time_available": time_available})


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
