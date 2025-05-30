# utils/llm_utils.py
import requests
import time
import logging

logger = logging.getLogger(__name__)


def call_ollama(prompt: str, model: str = "llama3:8b", url: str = "http://localhost:11434",
                max_tokens: int = 512, temperature: float = 0.7) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_tokens,
            "repeat_penalty": 1.1
        }
    }
    try:
        start_time = time.time()
        response = requests.post(f"{url}/api/generate", json=payload, timeout=120)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            return {
                "success": True,
                "response": result,
                "response_time": end_time - start_time,
                "tokens_estimated": len(result.split()),
                "error": None
            }
        else:
            return {"success": False, "response": "", "response_time": 0, "tokens_estimated": 0,
                    "error": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.exception("Error calling Ollama")
        return {"success": False, "response": "", "response_time": 0, "tokens_estimated": 0, "error": str(e)}
