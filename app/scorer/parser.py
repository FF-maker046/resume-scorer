"""
Comprehensive resume document parser:
Supports Word (.docx, .doc), PDF (text & OCR scanned), Plain text, and Images (.jpg, .png).
Handles Word Drawing Canvas, Shapes, Grouped Text Boxes, and Tables.
"""
import io
import os
import re
import tempfile
from typing import Optional


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT with multiple encoding fallbacks."""
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16", "latin1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX using python-docx.
    Traverses standard paragraphs, tables, and XML drawing text boxes (w:txbxContent).
    """
    texts = []
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        for p in doc.paragraphs:
            if p.text.strip():
                texts.append(p.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    texts.append(row_text)

        # Extract text boxes inside shapes and drawings (w:txbxContent)
        for elem in doc.element.iter():
            if elem.tag.endswith("txbxContent"):
                for p in elem.iter():
                    if p.tag.endswith("}t") and p.text and p.text.strip():
                        texts.append(p.text.strip())

        if texts:
            return "\n".join(texts).strip()
    except Exception as e:
        pass

    return "\n".join(texts).strip()


def extract_text_from_doc(file_bytes: bytes) -> str:
    """
    Extract text from legacy Word 97-2003 (.doc) binary OLE files.
    On Windows, leverages Word or WPS COM automation to read paragraphs, tables,
    and grouped shapes / text boxes (Drawing Canvas).
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_resume_{os.getpid()}_{os.urandom(4).hex()}.doc")

    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # 1. Try Windows COM (Word or WPS)
        word_app = None
        doc = None
        try:
            import win32com.client
            for prog_id in ("Word.Application", "KWPS.Application"):
                try:
                    word_app = win32com.client.Dispatch(prog_id)
                    word_app.Visible = False
                    break
                except Exception:
                    word_app = None

            if word_app:
                doc = word_app.Documents.Open(os.path.abspath(temp_path))
                texts = []

                # Body paragraphs
                for p in doc.Paragraphs:
                    t = p.Range.Text.strip()
                    if t:
                        texts.append(t)

                # Tables
                for t in doc.Tables:
                    for r in t.Rows:
                        cells = [c.Range.Text.strip().replace("\r\x07", "") for c in r.Cells if c.Range.Text.strip()]
                        if cells:
                            texts.append(" | ".join(cells))

                # Recursive Shape & TextFrame extraction (essential for styled resumes)
                def get_shape_text(s):
                    sub_texts = []
                    try:
                        if s.Type == 6:  # msoGroup
                            for i in range(1, s.GroupItems.Count + 1):
                                sub_texts.extend(get_shape_text(s.GroupItems(i)))
                        elif s.TextFrame and s.TextFrame.HasText:
                            st = s.TextFrame.TextRange.Text.strip()
                            if st:
                                sub_texts.append(st)
                    except Exception:
                        pass
                    return sub_texts

                for i in range(1, doc.Shapes.Count + 1):
                    texts.extend(get_shape_text(doc.Shapes(i)))

                doc.Close(False)
                doc = None
                word_app.Quit()
                word_app = None

                if texts:
                    return "\n".join(texts).strip()
        except Exception:
            if doc:
                try:
                    doc.Close(False)
                except Exception:
                    pass
            if word_app:
                try:
                    word_app.Quit()
                except Exception:
                    pass

        # 2. Fallback: Pure python OLE stream text extraction
        try:
            import olefile
            if olefile.isOleFile(temp_path):
                ole = olefile.OleFileIO(temp_path)
                if ole.exists("WordDocument"):
                    stream = ole.openstream("WordDocument").read()
                    # Extract continuous ascii and utf-16 text chunks
                    chunks = []
                    # Simple decode scan
                    for enc in ("utf-16le", "gb18030", "gbk"):
                        try:
                            decoded = stream.decode(enc, errors="ignore")
                            words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9@._:\-\s]{4,}", decoded)
                            if len(words) > 5:
                                chunks.extend(words)
                        except Exception:
                            pass
                    if chunks:
                        return "\n".join(chunks)
        except Exception:
            pass

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF / pypdf, with OCR fallback for image-based PDFs."""
    text_chunks = []

    # 1. Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            t = page.get_text()
            if t and t.strip():
                text_chunks.append(t.strip())

        # If text is sufficient, return
        combined = "\n".join(text_chunks).strip()
        if len(combined) >= 50:
            return combined

        # Otherwise: Image-only / Scanned PDF fallback via RapidOCR
        ocr_chunks = []
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                ocr_result, _ = ocr(img_bytes)
                if ocr_result:
                    for line in ocr_result:
                        ocr_chunks.append(line[1])
            if ocr_chunks:
                return "\n".join(ocr_chunks).strip()
        except Exception:
            pass

        if combined:
            return combined
    except Exception:
        pass

    # 2. Try pypdf backup
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                text_chunks.append(t.strip())
        if text_chunks:
            return "\n".join(text_chunks).strip()
    except Exception:
        pass

    return "\n".join(text_chunks).strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from JPG/PNG images using RapidOCR."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        ocr_result, _ = ocr(file_bytes)
        if ocr_result:
            return "\n".join(line[1] for line in ocr_result).strip()
    except Exception:
        pass
    return ""


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Detect file type by extension and extract resume text cleanly."""
    lower = filename.lower()
    if lower.endswith(".doc"):
        return extract_text_from_doc(file_bytes)
    elif lower.endswith(".docx"):
        extracted = extract_text_from_docx(file_bytes)
        if not extracted or len(extracted) < 10:
            extracted = extract_text_from_doc(file_bytes)
        return extracted
    elif lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return extract_text_from_image(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)
