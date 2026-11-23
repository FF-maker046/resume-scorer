"""
Resume document parser: supports plain text, PDF, and DOCX.
"""
import io
from typing import Optional


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT with multiple encoding fallbacks."""
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16", "latin1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using pypdf or PyMuPDF."""
    text_chunks = []
    # Try pypdf first
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_chunks.append(extracted)
        if text_chunks:
            return "\n".join(text_chunks).strip()
    except Exception:
        pass

    # Try pymupdf (fitz) as backup
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text_chunks.append(page.get_text())
        if text_chunks:
            return "\n".join(text_chunks).strip()
    except Exception:
        pass

    return "\n".join(text_chunks).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs).strip()
    except Exception as e:
        return f"[DOCX 解析失败: {e}]"


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Detect file type by extension and extract resume text."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".docx") or lower_name.endswith(".doc"):
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)
