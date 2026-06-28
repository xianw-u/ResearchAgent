from dataclasses import dataclass
from pathlib import Path

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import truncate_text


@dataclass
class PaperChunk:
    page: int
    text: str


@dataclass
class PaperIndex:
    chunks: list[PaperChunk]
    vectorizer: TfidfVectorizer
    matrix: object


def extract_pdf_text(pdf_path: str | Path) -> list[dict[str, str | int]]:
    pages: list[dict[str, str | int]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_index, "text": text})
    if not pages:
        raise ValueError("No extractable text was found. Scanned PDFs need OCR before indexing.")
    return pages


def build_paper_index(
    pages: list[dict[str, str | int]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> PaperIndex:
    chunks: list[PaperChunk] = []
    for page in pages:
        page_num = int(page["page"])
        text = str(page["text"])
        for chunk_text in split_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(PaperChunk(page=page_num, text=chunk_text))

    if not chunks:
        raise ValueError("No chunks were built from PDF text.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return PaperIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def answer_question(index: PaperIndex, question: str, top_k: int = 3) -> dict[str, object]:
    if not question:
        raise ValueError("Question cannot be empty.")

    query_vec = index.vectorizer.transform([question])
    scores = cosine_similarity(query_vec, index.matrix).flatten()
    if scores.max() <= 0:
        ranked_indices = _fallback_indices(index, top_k)
    else:
        ranked_indices = scores.argsort()[::-1][:top_k]

    sources = []
    for idx in ranked_indices:
        score = float(scores[idx])
        chunk = index.chunks[int(idx)]
        sources.append({
            "page": chunk.page,
            "score": score,
            "text": truncate_text(chunk.text, max_chars=700),
        })

    if not sources:
        return {
            "answer": "No chunks were available in the paper index. Please rebuild the paper index.",
            "sources": [],
        }

    cited_pages = ", ".join(f"p.{source['page']}" for source in sources)
    best_score = max(source["score"] for source in sources)
    if best_score <= 0:
        answer = (
            "The lexical match is weak, but ResearchAgent still returned the top PDF chunks "
            f"as fallback context from {cited_pages}. Try asking with paper-specific terms, "
            "or enable LLM generation to summarize these retrieved snippets."
        )
    else:
        answer = (
            "Based on the retrieved paper context, the most relevant evidence appears in "
            f"{cited_pages}. Review the cited snippets below for the grounded answer."
        )
    return {"answer": answer, "sources": sources}


def _fallback_indices(index: PaperIndex, top_k: int) -> list[int]:
    priority_keywords = (
        "abstract",
        "introduction",
        "method",
        "approach",
        "conclusion",
        "summary",
    )
    scored: list[tuple[int, int]] = []
    for idx, chunk in enumerate(index.chunks):
        text = chunk.text.lower()
        keyword_score = sum(1 for keyword in priority_keywords if keyword in text)
        page_bonus = max(0, 8 - chunk.page)
        scored.append((idx, keyword_score * 10 + page_bonus))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]
