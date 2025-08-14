import os
import requests
from langchain_core.language_models import LLM

class FriendliLLM(LLM):
    endpoint_id: str
    api_key: str = os.getenv("FRIENDLI_API_KEY")

    @property
    def _llm_type(self) -> str:
        return "friendli"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        url = f"https://api.friendli.ai/dedicated/{self.endpoint_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": prompt}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        # Friendli usually returns something like {"output_text": "..."}
        return data.get("output_text", str(data))
