import logging

import gradio as gr
import httpx

from config import INGEST_URL
from rag import answer_question

log = logging.getLogger(__name__)


def read_status():
    try:
        response = httpx.get(f"{INGEST_URL}/status", timeout=5, trust_env=False)
        response.raise_for_status()
        state = response.json()
    except httpx.HTTPError:
        label = "Сервис индексации временно недоступен"
        message = ""
        ready = False
    else:
        corpus = state["corpus"] or "пока не загружен"
        label = f"Активный корпус: {corpus} · Документов: {state['documents']}"
        message = state["message"]
        ready = not state["running"]
    buttons = [gr.update(interactive=ready) for _ in range(3)]
    return label, message, *buttons


def start_rebuild(mode):
    try:
        response = httpx.post(f"{INGEST_URL}/rebuild/{mode}", timeout=5, trust_env=False)
        if response.status_code != 409:
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise gr.Error("Не удалось запустить индексацию. Проверьте сервис ingest.") from error
    return read_status()


def load_english():
    return start_rebuild("en")


def load_combined():
    return start_rebuild("both")


def load_russian():
    return start_rebuild("ru")


def ask(question, history, top_k):
    question = (question or "").strip()
    if not question:
        return gr.skip(), "", gr.skip(), gr.skip()
    try:
        answer, sources, timing = answer_question(question, int(top_k))
    except httpx.HTTPStatusError as error:
        log.exception("Local service rejected a request")
        message = ("Проверьте, загружена ли модель Qwen 3 в Ollama."
                   if error.request.url.path == "/api/chat" and error.response.status_code == 404
                   else "Проверьте журналы локальных сервисов.")
        raise gr.Error(message) from error
    except (httpx.RequestError, ValueError) as error:
        log.exception("Could not complete the request")
        raise gr.Error("Не удалось получить ответ от локальных сервисов.") from error
    messages = [
        *(history or []),
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return messages, "", sources, timing


theme = gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["monospace"])

with gr.Blocks(title="Enterprise Private GPT", analytics_enabled=False, theme=theme) as demo:
    gr.Markdown("# Enterprise Private GPT")
    with gr.Row():
        load_en = gr.Button("Только EN", interactive=False)
        load_both = gr.Button("EN+RU", interactive=False)
        load_ru = gr.Button("Только RU", interactive=False)
    active = gr.Textbox(label="База знаний", interactive=False)
    progress = gr.Textbox(label="Индексация", interactive=False)
    chat = gr.Chatbot(type="messages", label="Корпоративный чат", height=400, render_markdown=False)
    with gr.Row():
        question = gr.Textbox(label="Вопрос", placeholder="Задайте вопрос по документам", scale=5)
        send = gr.Button("Отправить", variant="primary")
    with gr.Accordion("Источники и параметры поиска", open=True):
        top_k = gr.Radio([1, 3, 5], value=3, label="Top-K")
        sources = gr.Textbox(label="Использованные фрагменты", lines=5, interactive=False)
        timing = gr.Textbox(label="Время обработки", interactive=False)

    status_outputs = [active, progress, load_en, load_both, load_ru]
    demo.load(read_status, outputs=status_outputs, queue=False, api_name="index_status")
    gr.Timer(2).tick(read_status, outputs=status_outputs, queue=False, api_name=False)
    load_en.click(load_english, outputs=status_outputs, api_name="load_en")
    load_both.click(load_combined, outputs=status_outputs, api_name="add_ru")
    load_ru.click(load_russian, outputs=status_outputs, api_name="load_ru")
    gr.on([question.submit, send.click], ask, inputs=[question, chat, top_k],
          outputs=[chat, question, sources, timing], api_name="ask", concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
