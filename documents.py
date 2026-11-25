from email import policy
from email.parser import BytesParser
from pathlib import Path

from docx import Document
from docx.table import Table
from openpyxl import load_workbook
from pptx import Presentation
from pypdf import PdfReader

FORMATS = {".pdf", ".docx", ".xlsx", ".pptx", ".eml", ".txt"}


def pdf_sections(path):
    for number, page in enumerate(PdfReader(path).pages, start=1):
        yield f"Страница {number}", page.extract_text() or ""


def word_sections(path):
    heading = "Документ"
    paragraphs = []
    for block in Document(path).iter_inner_content():
        if isinstance(block, Table):
            paragraphs.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
        elif block.style and block.style.name.startswith("Heading"):
            if paragraphs:
                yield heading, "\n".join(paragraphs)
            heading = block.text
            paragraphs = [block.text]
        else:
            paragraphs.append(block.text)
    if paragraphs:
        yield heading, "\n".join(paragraphs)


def workbook_sections(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in book:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(cell) for cell in row if cell is not None]
                if values:
                    rows.append(" | ".join(values))
            yield f"Лист {sheet.title}", "\n".join(rows)
    finally:
        book.close()


def slide_sections(path):
    for number, slide in enumerate(Presentation(path).slides, start=1):
        paragraphs = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                paragraphs.append(shape.text)
            if shape.has_table:
                paragraphs.extend(
                    " | ".join(cell.text for cell in row.cells) for row in shape.table.rows
                )
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            paragraphs.append(slide.notes_slide.notes_text_frame.text)
        yield f"Слайд {number}", "\n".join(paragraphs)


def email_sections(path):
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    body = message.get_body(preferencelist=("plain",))
    yield "Письмо", "\n".join(
        [f"Тема: {message['Subject'] or ''}", body.get_content() if body else ""]
    )


def sections(path: Path):
    extractors = {
        ".pdf": pdf_sections,
        ".docx": word_sections,
        ".xlsx": workbook_sections,
        ".pptx": slide_sections,
        ".eml": email_sections,
    }
    if path.suffix.lower() == ".txt":
        yield "Документ", path.read_text(encoding="utf-8-sig")
    else:
        yield from extractors[path.suffix.lower()](path)
