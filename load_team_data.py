"""
load_team_data.py - Load everyone's data into the vector database with one command.

Put the team's files here first:
    data/pdfs/       <- Michael's research papers (.pdf)  (tip: python download_papers.py)
    data/urls.txt    <- Fatema's web article links, one per line
    data/youtube.txt <- Fatema's YouTube video links, one per line (videos need captions)
    data/images/     <- Dimitri's diagrams/screenshots (.png/.jpg), described automatically,
                        OR one .txt file per image (first line = image name, rest = description)
    data/texts/      <- saved article text, same .txt format (for websites that block loading)

Then run:   python load_team_data.py          (add --reset to start from an empty database)
Already-added files are skipped automatically (duplicate check), so it is safe to run again.
"""
import argparse
from pathlib import Path

import config
import ingest
import vector_store

DATA = config.PROJECT_ROOT / "data"


def report(name, action):
    try:
        result = action()
        status = f"added {result['chunks']} chunks" if result["added"] else f"skipped: {result['reason']}"
    except Exception as error:
        status = f"ERROR: {error}"
    print(f"  {name}: {status}")


def read_list(file: Path) -> list[str]:
    if not file.exists():
        return []
    return [line.strip() for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def load_pdfs():
    folder = DATA / "pdfs"
    files = sorted(folder.glob("*.pdf")) if folder.exists() else []
    print(f"\n--- PDFs ({len(files)} found in {folder.name}/) ---")
    for pdf in files:
        report(pdf.name, lambda: ingest.ingest_pdf(str(pdf)))


def load_urls():
    urls = read_list(DATA / "urls.txt")
    print(f"\n--- Web articles ({len(urls)} found in urls.txt) ---")
    for url in urls:
        report(url, lambda: ingest.ingest_web(url))


def load_videos():
    urls = read_list(DATA / "youtube.txt")
    print(f"\n--- YouTube videos ({len(urls)} found in youtube.txt) ---")
    for url in urls:
        report(url, lambda: ingest.ingest_youtube(url))


def load_images():
    folder = DATA / "images"
    pictures = sorted(f for f in folder.iterdir() if f.suffix.lower() in ingest.IMAGE_TYPES) if folder.exists() else []
    print(f"\n--- Images ({len(pictures)} found in images/) ---")
    for picture in pictures:
        report(picture.name, lambda: ingest.ingest_image(str(picture)))
    load_text_files("images", "Image", "image")


def load_text_files(folder_name, label, doc_type):
    """Load .txt files: first line = name/source, the rest = the text."""
    folder = DATA / folder_name
    files = sorted(folder.glob("*.txt")) if folder.exists() else []
    if not files:
        return
    print(f"\n--- {label} descriptions / texts ({len(files)} .txt found in {folder_name}/) ---")
    for txt in files:
        lines = txt.read_text(encoding="utf-8").splitlines()
        source, text = (lines[0].strip(), "\n".join(lines[1:])) if len(lines) > 1 else (txt.stem, "\n".join(lines))
        raw = [{"text": text, "metadata": {"source": f"{label}: {source}"}, "markdown": False}]
        report(txt.name, lambda: ingest.store(raw, doc_type, ingest.file_hash(text.encode()), min_length=50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load all team data from the data/ folder.")
    parser.add_argument("--reset", action="store_true", help="Empty the database first")
    args = parser.parse_args()
    if args.reset:
        vector_store.reset_database()
        print("Database emptied.")

    load_pdfs()
    load_urls()
    load_videos()
    load_images()
    load_text_files("texts", "Article", "text")
    print(f"\nDone. Total chunks in database: {vector_store.count_chunks()}")
