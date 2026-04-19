"""
Lightweight Ollama API client for Remote Gemma4 + nomic-embed-text.
No heavy dependencies (langchain, torch, etc.) — just requests.
"""

import requests
import base64
import json
import time
from typing import Optional


class OllamaClient:

    def __init__(self, host: str = "192.168.31.63", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"[OllamaClient] Connected to {self.base_url}")
            print(f"[OllamaClient] Available models: {models}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {e}"
            )

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def embed_batch(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def caption_image(
        self,
        image_bytes: bytes,
        model: str = "gemma4:e4b",
        prompt: Optional[str] = None,
    ) -> str:
        if prompt is None:
            prompt = (
                "Please describe in detail what you see in this image. "
                "Focus on people, objects, environmental features, "
                "activities, and spatial layout. Be specific and concise."
            )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [b64_image]}
                ],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat(
        self,
        messages: list[dict],
        model: str = "gemma4:e4b",
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str:
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str = "gemma4:e4b",
        temperature: float = 0.0,
    ) -> dict:
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]
