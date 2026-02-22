import httpx

from rag_lab.core.config import settings


class ChatServiceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def generate_answer(message: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "prompt": message,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            response = await client.post(settings.ollama_url, json=payload)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ChatServiceError(504, "Ollama request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ChatServiceError(
            502,
            f"Ollama returned HTTP {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise ChatServiceError(503, "Failed to reach Ollama service") from exc

    data = response.json()
    answer = data.get("response")
    if not isinstance(answer, str):
        raise ChatServiceError(502, "Invalid response from Ollama service")

    return answer
