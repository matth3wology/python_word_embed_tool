from pypdf import PdfReader
import re


def clean_text(text: str) -> str:
    text = text.replace("■", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)   # rejoin hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path: str) -> str:
    """Read a PDF into text"""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Generate Chunks from the text with overlapping for embedding"""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks
