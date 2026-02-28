"""
Сервер для распознавания достопримечательностей по фото.
Принимает изображение (base64), отправляет в Ollama, возвращает результат.

Запуск: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import json
import re
import base64
from io import BytesIO

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ========= НАСТРОЙКИ =========
OLLAMA_MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_NUM_THREADS = 6
LLM_NUM_CTX = 512
LLM_NUM_PREDICT = 48
MAX_IMAGE_WIDTH = 256
JPEG_QUALITY = 50


app = FastAPI(title="Landmark Facts API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    """Запрос: изображение в base64 (JPEG/PNG)."""
    image_base64: str


class LandmarkResponse(BaseModel):
    """Ответ: название, локация и факт."""
    text: str  # "Название, город, страна\n! факт"
    name: str  # первая строка
    fact: str  # вторая строка (без '! ')


def _process_ollama_response(text: str) -> str:
    """Нормализует ответ модели в формат 'название\\n! факт'."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return (
            "Неизвестное место\n"
            "! Модель не смогла сгенерировать ответ по этой фотографии."
        )

    def _strip_enumeration(s: str) -> str:
        s = s.lstrip("-•–— \t")
        s = re.sub(r"^\d+\s*[\.\)]\s*", "", s)
        return s.strip()

    first = _strip_enumeration(lines[0])
    second_raw = _strip_enumeration(lines[1]) if len(lines) >= 2 else "Не удалось уверенно определить место по фотографии."
    second = second_raw.lstrip()
    if not second.startswith("!"):
        second = "! " + second.lstrip("! ").lstrip()

    return f"{first}\n{second}"


def _ask_ollama(image_b64: str) -> str:
    """Отправляет картинку в Ollama и возвращает результат."""
    prompt = (
        "Ты — локальный ИИ, запущенный через Ollama. "
        "По фотографии определи известную достопримечательность и её местоположение.\n\n"
        "Ответь СТРОГО в двух строках на русском языке БЕЗ нумерации и маркеров:\n"
        "1) Первая строка: только 'Название, город, страна(локация)'. Без лишних слов, без '1)', '1.' и т.п.\n"
        "2) Вторая строка: начинается с символа '! ' (восклицательный знак и пробел), "
        "после него — один короткий интересный факт об этом месте, не более двух предложений. "
        "Тоже без '2)', '2.' и других номеров.\n"
        "Не добавляй никаких комментариев, пояснений, заголовков, JSON, тегов и т.п. "
        "Верни ТОЛЬКО эти две строки."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
            "num_thread": LLM_NUM_THREADS,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except requests.exceptions.Timeout:
        return (
            "Неизвестное место\n"
            "! Модель слишком долго обрабатывала изображение и была остановлена по таймауту."
        )
    except requests.exceptions.ConnectionError:
        return (
            "Неизвестное место\n"
            "! Не удалось подключиться к Ollama. Убедись, что он запущен (ollama serve)."
        )
    except Exception:
        return "Неизвестное место\n! Произошла непредвиденная ошибка при запросе к модели."

    if not resp.ok:
        if resp.status_code >= 500:
            return (
                "Неизвестное место\n"
                "! Модель не смогла обработать изображение (внутренняя ошибка сервера)."
            )
        return (
            "Неизвестное место\n"
            f"! Сервер Ollama вернул ошибку {resp.status_code}."
        )

    try:
        data = resp.json()
        text = (data.get("response") or "").strip()
    except json.JSONDecodeError:
        text = resp.text.strip()

    if not text:
        return (
            "Неизвестное место\n"
            "! Модель не смогла сгенерировать ответ по этой фотографии."
        )

    return _process_ollama_response(text)


@app.get("/")
def root():
    """Проверка работы сервера."""
    return {
        "service": "Landmark Facts API",
        "status": "ok",
        "endpoints": {
            "POST /recognize": "Отправить изображение в base64 (JSON: {image_base64: \"...\"})",
            "GET /health": "Проверка здоровья сервера",
        },
    }


@app.get("/health")
def health():
    """Health check для мониторинга."""
    return {"status": "ok"}


@app.post("/recognize", response_model=LandmarkResponse)
def recognize_landmark(req: ImageRequest):
    """
    Принимает изображение в base64 и возвращает название достопримечательности + факт.
    """
    if not req.image_base64 or not req.image_base64.strip():
        raise HTTPException(status_code=400, detail="image_base64 не может быть пустым")

    try:
        # Декодируем для проверки
        raw = base64.b64decode(req.image_base64, validate=True)
        if len(raw) > 5 * 1024 * 1024:  # 5 MB
            raise HTTPException(status_code=400, detail="Изображение слишком большое (макс 5 MB)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некорректный base64: {e}")

    result_text = _ask_ollama(req.image_base64)
    lines = result_text.strip().split("\n")
    name = lines[0] if lines else "Неизвестное место"
    fact = lines[1].lstrip("! ").strip() if len(lines) > 1 else ""

    return LandmarkResponse(text=result_text, name=name, fact=fact)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
