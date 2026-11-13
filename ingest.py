"""
ingest.py - How documents get INTO the vector database.

Huzaifa builds this platform; Michael (PDFs), Fatema (web articles + YouTube videos) and
Dimitri (images) use these functions to load their own data. Main entry points:

    ingest_pdf("paper.pdf")                 # Michael - research papers (uses Marker)
    ingest_web("https://...")               # Fatema  - web articles
    ingest_youtube("https://youtube...")    # Fatema  - video transcripts
    ingest_image("diagram.png")             # Dimitri - diagrams / screenshots (described by GPT-4o-mini)
    add_text(text, {"source": "..."})       # anyone  - already-extracted text

Every document type is turned into TEXT first, then embedded the same way:
    PDF   -> Markdown (Marker) or plain text per page (pypdf)
    web   -> visible text of the page
    video -> caption transcript, with [mm:ss] timestamps
    image -> a written description from a vision model (+ OCR text if Tesseract is installed)

Before anything is stored, check_worth_adding() asks:
    1. Is it already in the DB?        (exact hash + near-duplicate check)
    2. Is it about Agentic RAG?        (quick LLM check)

Command line:
    python ingest.py paper.pdf diagram.png https://some-article.com https://youtu.be/VIDEO_ID
    python ingest.py --path data/pdfs --url https://... --youtube https://... [--reset] [--force]
"""
import argparse
import base64
import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from langchain_text_splitters import (Language, MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)
from pydantic import BaseModel

import config
import prompts
import vector_store

IMAGE_FOLDER = config.PROJECT_ROOT / "extracted_images"
PDF_TYPES = {".pdf"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
TEXT_TYPES = {".txt", ".md", ".markdown"}

# A "raw document" is a dict: {"text": str, "metadata": dict, "markdown": bool}
# Loaders return lists of raw documents; make_chunks() cuts them into chunks.


# ---------------------------------------------------------------------------
# PDF -> Markdown with Marker (or plain text per page with pypdf)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_marker_converter():
    """Load Marker's models once (slow the first time, then cached)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    # paginate_output adds a "{page}------" line between pages, so we keep page numbers
    return PdfConverter(artifact_dict=create_model_dict(), config={"paginate_output": True})


PAGE_BREAK = re.compile(r"\n*\{(\d+)\}-{20,}\n*")


def load_pdf(pdf_path: str, use_marker: bool = True) -> tuple[list[dict], list[tuple]]:
    """
    Turn a PDF into raw documents (one per page) plus its images.
    Marker keeps headings, tables and equations as Markdown.
    If Marker is not installed (or fails), pypdf extracts plain text per page.
    Returns (raw_docs, [(image_name, PIL.Image, page_number), ...]).
    """
    name = Path(pdf_path).name
    try:
        if not use_marker:
            raise RuntimeError("Marker switched off")
        from marker.output import text_from_rendered
        rendered = get_marker_converter()(pdf_path)
        markdown, _, images = text_from_rendered(rendered)
        parts = PAGE_BREAK.split(markdown)
        raw = []
        if parts[0].strip():
            raw.append({"text": parts[0], "metadata": {"source": name, "page": 1}, "markdown": True})
        for page_id, text in zip(parts[1::2], parts[2::2]):
            if text.strip():
                raw.append({"text": text, "markdown": True,
                            "metadata": {"source": name, "page": int(page_id) + 1}})
        found = []
        for image_name, image in images.items():
            match = re.search(r"page_(\d+)", image_name)
            found.append((image_name, image, int(match.group(1)) + 1 if match else None))
        return raw, found
    except Exception as error:   # Marker missing or crashed -> basic extraction
        if use_marker:
            print(f"  Marker not used ({type(error).__name__}) -> basic PDF text extraction.")

    # LangChain's PyPDFLoader: one Document per page (its "page" metadata starts at 0)
    from langchain_community.document_loaders import PyPDFLoader
    raw = [{"text": d.page_content, "markdown": False,
            "metadata": {"source": name, "page": d.metadata.get("page", 0) + 1}}
           for d in PyPDFLoader(pdf_path).load() if d.page_content.strip()]

    # Images are taken out with pypdf directly
    from pypdf import PdfReader
    found = []
    for number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        try:
            for image_file in page.images:
                found.append((f"page{number}_{image_file.name}", image_file.image, number))
        except Exception:
            pass  # some PDFs have images pypdf cannot decode; text is what matters most
    return raw, found


# ---------------------------------------------------------------------------
# Web articles and YouTube videos
# ---------------------------------------------------------------------------
def load_web(url: str) -> list[dict]:
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(url, header_template=config.HTTP_HEADERS)
    text = "\n".join(d.page_content for d in loader.load())
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()   # squash empty lines from the HTML
    return [{"text": text, "metadata": {"source": url, "url": url}, "markdown": False}]


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/")
    if "/shorts/" in parsed.path or "/embed/" in parsed.path:
        return parsed.path.rstrip("/").split("/")[-1]
    return parse_qs(parsed.query).get("v", [url])[0]


TRANSCRIPT_FOLDER = config.PROJECT_ROOT / "data" / "transcripts"


def _snippets_from_api(video: str) -> list[tuple[float, str]]:
    """Method 1: youtube-transcript-api (fast, but YouTube often blocks it)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    if hasattr(YouTubeTranscriptApi, "fetch"):                  # version 1.x
        return [(s.start, s.text) for s in YouTubeTranscriptApi().fetch(video, languages=["en"])]
    return [(s["start"], s["text"]) for s in                    # version 0.6.x
            YouTubeTranscriptApi.get_transcript(video, languages=["en"])]


def _ytdlp_options(**extra) -> dict:
    options = {"quiet": True, "no_warnings": True, "noprogress": True, **extra}
    browser = getattr(config, "YOUTUBE_COOKIES_BROWSER", "")    # e.g. "chrome" - optional
    if browser:
        options["cookiesfrombrowser"] = (browser,)
    return options


def _snippets_from_ytdlp_subtitles(video: str) -> tuple[list[tuple[float, str]], str]:
    """Method 2: yt-dlp downloads the (auto) captions. Returns (snippets, title)."""
    import tempfile
    import yt_dlp
    with tempfile.TemporaryDirectory() as folder:
        options = _ytdlp_options(skip_download=True, writesubtitles=True, writeautomaticsub=True,
                                 subtitleslangs=["en.*", "en"], subtitlesformat="json3",
                                 outtmpl=f"{folder}/%(id)s")
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=True)
        files = sorted(Path(folder).glob("*.json3"))
        if not files:
            raise RuntimeError("no English captions available")
        import json
        events = json.loads(files[0].read_text(encoding="utf-8")).get("events", [])
        snippets = []
        for event in events:
            text = "".join(seg.get("utf8", "") for seg in event.get("segs", []) or []).strip()
            if text:
                snippets.append((event.get("tStartMs", 0) / 1000, text))
        return snippets, info.get("title", video)


def _snippets_from_whisper(video: str) -> tuple[list[tuple[float, str]], str]:
    """Method 3: download the audio with yt-dlp and transcribe it locally with Whisper.
    Works even when the video has no captions. Slow the first time (model download)."""
    import tempfile
    import yt_dlp
    from faster_whisper import WhisperModel
    with tempfile.TemporaryDirectory() as folder:
        options = _ytdlp_options(format="bestaudio[ext=m4a]/bestaudio", outtmpl=f"{folder}/%(id)s.%(ext)s")
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=True)
        audio = next(Path(folder).glob(f"{video}.*"))
        model = WhisperModel(getattr(config, "WHISPER_MODEL", "base.en"), compute_type="int8")
        segments, _ = model.transcribe(str(audio), language="en", vad_filter=True)
        return [(s.start, s.text.strip()) for s in segments], info.get("title", video)


def _oembed_title(url: str, fallback: str) -> str:
    try:
        import requests
        return requests.get("https://www.youtube.com/oembed",
                            params={"url": url, "format": "json"}, timeout=10).json()["title"]
    except Exception:
        return fallback


def fetch_youtube_snippets(video: str, url: str) -> tuple[list[tuple[float, str]], str]:
    """Get [(start_seconds, text)] and the title. Tries each method in turn and caches the
    result in data/transcripts/<id>.json so YouTube is only contacted once per video."""
    import json
    cache = TRANSCRIPT_FOLDER / f"{video}.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        return [tuple(s) for s in saved["snippets"]], saved["title"]

    errors = []
    methods = [("youtube-transcript-api", lambda: (_snippets_from_api(video), None)),
               ("yt-dlp captions", lambda: _snippets_from_ytdlp_subtitles(video)),
               ("yt-dlp audio + Whisper", lambda: _snippets_from_whisper(video))]
    for name, method in methods:
        try:
            snippets, title = method()
            if not snippets:
                raise RuntimeError("empty transcript")
            print(f"  transcript via {name}")
            title = title or _oembed_title(url, video)
            TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"title": title, "snippets": snippets}), encoding="utf-8")
            return snippets, title
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {str(error).splitlines()[0][:120] if str(error) else ''}")
            print(f"  {name} failed, trying next method...")
    raise RuntimeError("Could not get a transcript. " + " | ".join(errors))


def load_youtube(url: str) -> list[dict]:
    """Get the transcript of a video (with fallbacks, see fetch_youtube_snippets). Adds a
    [mm:ss] mark about every minute so answers can point to the moment in the video."""
    video = youtube_id(url)
    snippets, title = fetch_youtube_snippets(video, url)

    pieces, next_mark = [], 0
    for start, text in snippets:
        if start >= next_mark:
            pieces.append(f"\n[{int(start // 60):02d}:{int(start % 60):02d}]")
            next_mark = start + 60
        pieces.append(text.replace("\n", " "))
    transcript = " ".join(pieces).strip()
    return [{"text": transcript, "markdown": False,
             "metadata": {"source": f"YouTube: {title}", "url": f"https://www.youtube.com/watch?v={video}"}}]


# ---------------------------------------------------------------------------
# Images -> text (vision model description + optional OCR)
# ---------------------------------------------------------------------------
def describe_image(image) -> str:
    """Turn an image (PIL.Image or file path) into searchable text."""
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))       # smaller image = cheaper API call, same detail for diagrams

    parts = []
    if config.OPENAI_API_KEY:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        message = HumanMessage(content=[
            {"type": "text", "text": prompts.IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        parts.append(ChatOpenAI(model=config.VISION_MODEL, temperature=0).invoke([message]).content)

    try:   # OCR is optional: needs `pip install pytesseract` AND the Tesseract program
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        if text:
            parts.append(f"Text found in the image (OCR):\n{text}")
    except Exception:
        pass
    return "\n\n".join(parts)


def send_to_image_pipeline(images: list[tuple], source: str) -> list[dict]:
    """
    Images found inside a PDF: save them, then describe the biggest ones as text so
    figures (architecture diagrams, result charts) become searchable too.
    Dimitri: improve describe_image() if you want a different image method.
    """
    IMAGE_FOLDER.mkdir(exist_ok=True)
    big = [(n, img, p) for n, img, p in images
           if min(img.size) >= config.MIN_IMAGE_SIDE]
    big.sort(key=lambda item: item[1].size[0] * item[1].size[1], reverse=True)

    raw = []
    for number, (name, image, page) in enumerate(big[:config.MAX_IMAGES_PER_PDF], start=1):
        safe = re.sub(r"[^\w.-]", "_", f"{Path(source).stem}_{name}")
        image.convert("RGB").save(IMAGE_FOLDER / (Path(safe).stem + ".png"))
        try:
            description = describe_image(image)
        except Exception as error:
            print(f"  Could not describe {name}: {error}")
            continue
        if description:
            meta = {"source": f"{source} (figure {number})", "type": "image"}
            if page:
                meta["page"] = page
            raw.append({"text": description, "metadata": meta, "markdown": False})
    return raw


# ---------------------------------------------------------------------------
# Is this document worth adding?
# ---------------------------------------------------------------------------
class WorthAdding(BaseModel):
    worth_adding: bool
    reason: str


def file_hash(data: bytes) -> str:
    """A fingerprint of the file. Same file -> same hash."""
    return hashlib.sha256(data).hexdigest()


def check_worth_adding(text: str, hash_value: str | None = None,
                       min_length: int = 200) -> tuple[bool, str]:
    if len(text.strip()) < min_length:
        return False, "Document has almost no text."

    duplicate, reason = vector_store.is_duplicate(text, hash_value)
    if duplicate:
        return False, reason

    from agent import get_llm
    verdict = get_llm().with_structured_output(WorthAdding).invoke(
        prompts.WORTH_ADDING_PROMPT.format(text=text[:2000]))
    return verdict.worth_adding, verdict.reason


# ---------------------------------------------------------------------------
# Chunk and store
# ---------------------------------------------------------------------------
def make_chunks(raw_docs: list[dict], chunk_size: int = config.CHUNK_SIZE,
                chunk_overlap: int = config.CHUNK_OVERLAP) -> list[Document]:
    """
    Cut raw documents into chunks.
    Markdown (from Marker) is first split at headings, so every chunk remembers its
    section title; long sections are then cut further at paragraphs/sentences.
    """
    headings = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for raw in raw_docs:
        metadata = {key: value for key, value in raw["metadata"].items() if value is not None}
        if raw.get("markdown"):
            sections = headings.split_text(raw["text"])
            for section in sections:
                title = section.metadata.get("h3") or section.metadata.get("h2") or section.metadata.get("h1")
                section.metadata = dict(metadata, **({"section": title} if title else {}))
            chunks += markdown_splitter.split_documents(sections)
        else:
            chunks += text_splitter.create_documents([raw["text"]], metadatas=[metadata])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def store(raw_docs: list[dict], doc_type: str, hash_value: str, force: bool = False,
          min_length: int = 200) -> dict:
    """Run the worth-adding check, then chunk and save. Returns a small report."""
    raw_docs = [r for r in raw_docs if r["text"].strip()]
    if not raw_docs:
        return {"added": False, "chunks": 0, "reason": "No text could be extracted."}

    if not force:
        ok, reason = check_worth_adding("\n\n".join(r["text"] for r in raw_docs), hash_value, min_length)
        if not ok:
            return {"added": False, "chunks": 0, "reason": reason}

    for raw in raw_docs:
        raw["metadata"].setdefault("type", doc_type)
        raw["metadata"]["file_hash"] = hash_value
    chunks = make_chunks(raw_docs)
    vector_store.get_vector_store().add_documents(chunks)
    return {"added": True, "chunks": len(chunks), "reason": "Added to the database."}


def add_text(text: str, metadata: dict, markdown: bool = False) -> int:
    """Split already-extracted text into chunks and store them. Returns the number of chunks."""
    chunks = make_chunks([{"text": text, "metadata": dict(metadata), "markdown": markdown}])
    if chunks:
        vector_store.get_vector_store().add_documents(chunks)
    return len(chunks)


# ---------------------------------------------------------------------------
# One function per document type
# ---------------------------------------------------------------------------
def ingest_pdf(pdf_path: str, force: bool = False) -> dict:
    """Full PDF flow: Marker/pypdf -> worth-adding check -> figures described -> chunk -> store."""
    hash_value = file_hash(Path(pdf_path).read_bytes())
    if not force and vector_store.hash_exists(hash_value):
        return {"added": False, "chunks": 0, "images": 0, "reason": "Exact same file is already in the database."}

    raw, images = load_pdf(pdf_path)
    result = store(raw, "pdf", hash_value, force)
    result["images"] = 0
    if result["added"] and images:
        figures = send_to_image_pipeline(images, Path(pdf_path).name)
        if figures:
            result["chunks"] += store(figures, "image", hash_value, force=True)["chunks"]
            result["images"] = len(figures)
    return result


def ingest_web(url: str, force: bool = False) -> dict:
    """Load a web article and store it."""
    raw = load_web(url)
    return store(raw, "web", file_hash(raw[0]["text"].encode()), force)


def ingest_youtube(url: str, force: bool = False) -> dict:
    """Load a YouTube video's transcript and store it."""
    raw = load_youtube(url)
    return store(raw, "youtube", file_hash(raw[0]["text"].encode()), force)


def ingest_image(image_path: str, force: bool = False) -> dict:
    """Describe an image (diagram, screenshot, slide) and store the description."""
    hash_value = file_hash(Path(image_path).read_bytes())
    if not force and vector_store.hash_exists(hash_value):
        return {"added": False, "chunks": 0, "reason": "Exact same file is already in the database."}
    text = describe_image(image_path)
    raw = [{"text": text, "markdown": False,
            "metadata": {"source": f"Image: {Path(image_path).name}"}}]
    return store(raw, "image", hash_value, force, min_length=50)


def ingest_text_file(path: str, force: bool = False, source: str | None = None,
                     doc_type: str = "text") -> dict:
    """Store a .txt or .md file (Markdown is split at its headings)."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    raw = [{"text": text, "markdown": Path(path).suffix.lower() in {".md", ".markdown"},
            "metadata": {"source": source or Path(path).name}}]
    return store(raw, doc_type, file_hash(text.encode()), force, min_length=50)


def ingest_any(item: str, force: bool = False) -> dict:
    """Pick the right loader from the file type or URL."""
    if item.startswith("http"):
        if "youtube.com" in item or "youtu.be" in item:
            return ingest_youtube(item, force)
        return ingest_web(item, force)
    suffix = Path(item).suffix.lower()
    if suffix in PDF_TYPES:
        return ingest_pdf(item, force)
    if suffix in IMAGE_TYPES:
        return ingest_image(item, force)
    if suffix in TEXT_TYPES:
        return ingest_text_file(item, force)
    return {"added": False, "chunks": 0, "reason": f"Unsupported file type: {suffix}"}


def expand(path: str) -> list[str]:
    """A file -> [file]; a folder -> every supported file inside it."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    supported = PDF_TYPES | IMAGE_TYPES | TEXT_TYPES
    return [str(f) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in supported]


def main():
    parser = argparse.ArgumentParser(description="Add PDFs, images, text files, web pages or "
                                                 "YouTube videos to the knowledge base.")
    parser.add_argument("items", nargs="*", help="Files, folders or URLs")
    parser.add_argument("--path", action="append", default=[], help="A file or folder (repeatable)")
    parser.add_argument("--url", action="append", default=[], help="A web article URL (repeatable)")
    parser.add_argument("--youtube", action="append", default=[], help="A YouTube URL (repeatable)")
    parser.add_argument("--reset", action="store_true", help="Empty the database first")
    parser.add_argument("--force", action="store_true", help="Skip the duplicate/topic check")
    args = parser.parse_args()

    items = []
    for entry in args.items + args.path:
        items += [entry] if entry.startswith("http") else expand(entry)
    items += args.url + args.youtube
    if not items:
        parser.error("Give at least one file, folder or URL.")

    if args.reset:
        vector_store.reset_database()
        print("Database emptied.")

    for item in items:
        try:
            result = ingest_any(item, force=args.force)
            status = f"added {result['chunks']} chunks" if result["added"] else f"skipped: {result['reason']}"
        except Exception as error:
            status = f"ERROR: {error}"
        print(f"{item}: {status}")
    print(f"Total chunks in database: {vector_store.count_chunks()}")


if __name__ == "__main__":
    main()
