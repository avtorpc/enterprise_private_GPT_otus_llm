import logging
from threading import Lock
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COLLECTION_ALIAS, DOCUMENT_ROOT, QDRANT_URL, VECTOR_SIZE, active_collection, embeddings
from documents import FORMATS, sections

api = FastAPI()
log = logging.getLogger(__name__)
lock = Lock()
job = {"running": False, "message": ""}
CORPORA = {"en": ("en",), "both": ("en", "ru"), "ru": ("ru",)}
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)


def set_job(running, message):
    with lock:
        job.update(running=running, message=message)


def qdrant(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()["result"]


def rebuild(mode):
    new_collection = None
    with httpx.Client(base_url=QDRANT_URL, timeout=30, trust_env=False) as client:
        try:
            previous = active_collection(client)
            languages = CORPORA[mode]
            files = [
                (language, path)
                for language in languages
                for path in sorted((DOCUMENT_ROOT / language).iterdir())
                if path.is_file() and path.suffix.lower() in FORMATS
            ]
            if any(not any(lang == language for lang, _ in files) for language in languages):
                raise ValueError("Один из выбранных каталогов не содержит документов.")

            new_collection = f"knowledge_{mode}_{len(files)}_{uuid4().hex}"
            qdrant(client, "PUT", f"/collections/{new_collection}", json={
                "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"},
            })
            total_chunks = 0
            for index, (language, path) in enumerate(files, start=1):
                set_job(True, f"Индексация {index}/{len(files)} · {path.name}")
                chunks = [
                    {"text": chunk, "section": section, "file": path.name, "language": language}
                    for section, text in sections(path)
                    for chunk in splitter.split_text(text)
                ]
                if not chunks:
                    raise ValueError(f"Файл {path.name} не содержит извлекаемого текста.")
                for start in range(0, len(chunks), 64):
                    batch = chunks[start:start + 64]
                    vectors = embeddings([chunk["text"] for chunk in batch], "passage")
                    points = [
                        {"id": str(uuid4()), "vector": vector, "payload": chunk}
                        for chunk, vector in zip(batch, vectors, strict=True)
                    ]
                    qdrant(client, "PUT", f"/collections/{new_collection}/points",
                           params={"wait": "true"}, json={"points": points})
                total_chunks += len(chunks)

            count = qdrant(client, "POST", f"/collections/{new_collection}/points/count",
                           json={"exact": True})["count"]
            if count != total_chunks:
                raise RuntimeError("Число записей в Qdrant не совпало с результатом индексации.")
            actions = []
            if previous:
                actions.append({"delete_alias": {"alias_name": COLLECTION_ALIAS}})
            actions.append({"create_alias": {
                "collection_name": new_collection, "alias_name": COLLECTION_ALIAS,
            }})
            qdrant(client, "POST", "/collections/aliases", json={"actions": actions})
        except Exception as error:
            log.exception("Index build failed")
            if new_collection:
                try:
                    if active_collection(client) != new_collection:
                        client.delete(f"/collections/{new_collection}")
                except httpx.HTTPError:
                    log.warning("Could not verify or remove the temporary collection", exc_info=True)
            message = str(error) if isinstance(error, ValueError) else "Проверьте журнал сервиса ingest."
            set_job(False, f"Ошибка индексации. {message}")
            return

        if previous and previous.startswith("knowledge_"):
            try:
                qdrant(client, "DELETE", f"/collections/{previous}")
            except httpx.HTTPError:
                log.warning("Could not delete previous collection", exc_info=True)
        set_job(False, f"Индекс готов · {len(files)} документов · {total_chunks} чанков")


@api.get("/health")
def health():
    return {"ok": True}


@api.get("/status")
def status():
    with lock:
        state = job.copy()
    try:
        with httpx.Client(base_url=QDRANT_URL, timeout=3, trust_env=False) as client:
            collection = active_collection(client)
    except httpx.HTTPError as error:
        raise HTTPException(503, "Qdrant временно недоступен.") from error
    if collection:
        _, mode, count, _ = collection.split("_", maxsplit=3)
        state.update(corpus=" + ".join(language.upper() for language in CORPORA[mode]),
                     documents=int(count))
    else:
        state.update(corpus=None, documents=0)
    return state


@api.post("/rebuild/{mode}", status_code=202)
def start_rebuild(mode: Literal["en", "both", "ru"], tasks: BackgroundTasks):
    with lock:
        if job["running"]:
            raise HTTPException(409, "Индексация уже выполняется.")
        job.update(running=True, message="Подготовка индексации")
    tasks.add_task(rebuild, mode)
    return {"running": True}
