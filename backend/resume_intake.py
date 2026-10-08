"""Resume intake helpers that need no web framework: which kind of file was uploaded, and plain-text (.txt) extraction.

PDF parsing stays in main.extract_resume_text (PyPDF2). Supported formats are PDF and plain UTF-8 text; anything else
(including DOCX) is refused. Standard library plus resume_profile.clean_resume_text only."""
from resume_profile import clean_resume_text

KIND_PDF = "pdf"
KIND_TEXT = "text"

UNSUPPORTED_FORMAT_MESSAGE = "Only PDF or plain-text (.txt) resumes are supported"


class ResumeFormatError(ValueError):
    """The upload is not a usable resume; `status` is the HTTP status the endpoint should answer with."""
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def detect_resume_kind(filename: str | None, content_type: str | None, head: bytes) -> str:
    """'pdf' (the file starts like a PDF), 'text' (a .txt name or a text/plain type, and not a PDF) or raise 415."""
    if b"%PDF-" in head[:1024]:
        return KIND_PDF
    name = (filename or "").lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    if name.endswith(".txt") or ctype == "text/plain":
        return KIND_TEXT
    raise ResumeFormatError(UNSUPPORTED_FORMAT_MESSAGE, 415)


def extract_text_resume(data: bytes, min_chars: int) -> str:
    """Plain UTF-8 text -> cleaned resume text. Binary content, non-UTF-8 bytes and near-empty files are refused."""
    if b"\x00" in data[:4096]:
        raise ResumeFormatError("This file looks like a binary file, not plain text. Please upload a PDF or a UTF-8 .txt resume.", 415)
    try:
        text = data.decode("utf-8-sig")             # tolerates the BOM that Windows editors add
    except UnicodeDecodeError:
        raise ResumeFormatError("The text resume must be saved as UTF-8. Please re-save it as UTF-8 or upload a PDF.", 422)
    text = clean_resume_text(text)
    if len(text) < min_chars:
        raise ResumeFormatError("No readable text was found in the resume. Please upload a resume with more content.", 422)
    return text
