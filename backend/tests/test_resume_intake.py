"""Resume intake: PDF (unchanged) or plain UTF-8 text; nothing else. Standard library plus the app's own modules only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, check, finish  # noqa: E402

import resume_intake as ri  # noqa: E402
from resume_profile import build_candidate_profile  # noqa: E402

MIN = 50
TXT = ("Aarav Sharma\naarav@example.com\n\nSKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP\n\n"
       "EXPERIENCE\nML Intern - Acme\n- Built a text classifier with scikit-learn\n- Trained a PyTorch model\n")


def kind(name, ctype, head=b"hello"):
    try:
        return ri.detect_resume_kind(name, ctype, head)
    except ri.ResumeFormatError as e:
        return ("error", e.status)


def extract(data):
    try:
        return ri.extract_text_resume(data, MIN)
    except ri.ResumeFormatError as e:
        return ("error", e.status, str(e))


print("== detecting the kind of upload ==")
check("a PDF is recognised by its content, whatever the name or type", kind("cv.pdf", "application/pdf", b"%PDF-1.7 ...") == ri.KIND_PDF and kind("resume", "application/octet-stream", b"%PDF-1.4") == ri.KIND_PDF)
check("a PDF renamed to .txt is still treated as a PDF (content wins, as before)", kind("cv.txt", "text/plain", b"%PDF-1.5\n...") == ri.KIND_PDF)
check("a .txt file is plain text", kind("cv.txt", "text/plain") == ri.KIND_TEXT and kind("CV.TXT", None) == ri.KIND_TEXT)
check("text/plain without a .txt name is plain text", kind("resume", "text/plain; charset=utf-8") == ri.KIND_TEXT)
check("DOCX is refused with 415", kind("cv.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"PK\x03\x04") == ("error", 415))
check("legacy .doc / images / unknown types are refused with 415", all(kind(n, t, b"\xd0\xcf\x11\xe0") == ("error", 415) for n, t in [("cv.doc", "application/msword"), ("me.png", "image/png"), ("cv", None)]))
check("the refusal message names the supported formats", "PDF" in ri.UNSUPPORTED_FORMAT_MESSAGE and ".txt" in ri.UNSUPPORTED_FORMAT_MESSAGE and "DOCX" not in ri.UNSUPPORTED_FORMAT_MESSAGE.upper())

print("== plain-text extraction and validation ==")
text = extract(TXT.encode("utf-8"))
check("a normal UTF-8 resume is extracted and cleaned", isinstance(text, str) and "PyTorch" in text and "ML Intern" in text, text)
check("a UTF-8 BOM (Windows editors) is tolerated", extract(b"\xef\xbb\xbf" + TXT.encode("utf-8")) == text)
check("CRLF line endings are normalised", "\r" not in extract(TXT.replace("\n", "\r\n").encode("utf-8")))
check("non-ASCII text (accents, curly quotes) survives", "José" in extract(("José García\nSKILLS\nPython, Machine Learning, TensorFlow, scikit-learn\n" * 2).encode("utf-8")))
check("too little text -> 422", extract(b"too short")[0:2] == ("error", 422) and extract(b"")[0:2] == ("error", 422) and extract(b"   \n\n  ")[0:2] == ("error", 422))
check("binary content (NUL bytes) -> 415", extract(b"PK\x03\x04\x00\x00" + b"x" * 200)[0:2] == ("error", 415))
check("non-UTF-8 bytes (Latin-1 / UTF-16) -> 422 with a 'save as UTF-8' hint", extract("José García, ML engineer with Python and more text to pass the length check".encode("latin-1"))[0:2] == ("error", 422)
      and "UTF-8" in extract("José García, ML engineer with Python and more text".encode("latin-1"))[2])
check("control characters are removed", "\x07" not in extract(("Jane Doe\x07\nSKILLS\nPython, Machine Learning, Pandas, SQL, scikit-learn\n" * 2).encode("utf-8")))
profile = build_candidate_profile(extract(TXT.encode("utf-8")))
check("the text resume feeds the same profile builder as a PDF resume", "Machine Learning" in profile["skills"] and any(t in profile["technologies"] for t in ("PyTorch", "scikit-learn")), profile["skills"])

print("== wiring in main.py ==")
src = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
start = src[src.index("async def start_interview"):src.index("@app.post(\"/api/interview/chat\")")]
check("the size limit is enforced before the format is examined", start.index("413") < start.index("detect_resume_kind"))
check("PDF path unchanged (extract_resume_text / PyPDF2 route), text path uses extract_text_resume", "extract_resume_text(resume_bytes) if resume_kind == KIND_PDF" in start and "extract_text_resume(resume_bytes, MIN_RESUME_TEXT_CHARS)" in start)
check("format errors become HTTP errors with their own status", "except ResumeFormatError as e:" in start and "status_code=e.status" in start)
check("the stored file keeps its real extension (.pdf or .txt) and the original bytes", "'pdf' if resume_kind == KIND_PDF else 'txt'" in start and "f.write(resume_bytes)" in start)
check("no DOCX parsing exists in the intake code (no docx / zipfile handling)", "import docx" not in start and "zipfile" not in start and "import docx" not in open(os.path.join(BACKEND, "resume_intake.py"), encoding="utf-8").read() and "zipfile" not in open(os.path.join(BACKEND, "resume_intake.py"), encoding="utf-8").read())

finish()
