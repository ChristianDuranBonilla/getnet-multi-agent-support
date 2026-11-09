"""Crawl Getnet public pages, clean them and build the Chroma index.

The crawl stores one cleaned text file per page under `data/raw/` so that the
index can be rebuilt offline, for example during a Docker build.
"""

import argparse
import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import llm
from app.config import get_settings

logger = logging.getLogger(__name__)

SEED_URLS: tuple[str, ...] = (
    "https://site.getnet.com.br/duvidas/get-smart/",
    "https://site.getnet.com.br/duvidas/get-tap/",
    "https://site.getnet.com.br/duvidas/pix/",
    "https://site.getnet.com.br/duvidas/pix-mais-respostas/",
    "https://site.getnet.com.br/duvidas/antecipacao-de-vendas/",
    "https://site.getnet.com.br/duvidas/crediario/",
    "https://site.getnet.com.br/duvidas/parcelado/",
    "https://site.getnet.com.br/duvidas/link-de-pagamento/",
    "https://site.getnet.com.br/duvidas/aluguel/",
    "https://site.getnet.com.br/duvidas/aluguel-zero/",
    "https://site.getnet.com.br/duvidas/taxas/",
    "https://site.getnet.com.br/duvidas/operacional/",
    "https://site.getnet.com.br/duvidas/bandeiras/",
    "https://site.getnet.com.br/duvidas/get-conta/",
    "https://site.getnet.com.br/duvidas/app-getnet/",
    "https://site.getnet.com.br/duvidas/cancelamento-descredenciamento/",
    "https://site.getnet.com.br/duvidas/split-de-pagamento/",
    "https://site.getnet.com.br/duvidas/pre-autorizacao/",
    "https://site.getnet.com.br/",
    "https://site.getnet.com.br/todas-as-maquininhas/",
    "https://site.getnet.com.br/get-tap/",
    "https://site.getnet.com.br/pix/",
    "https://site.getnet.com.br/crediario/",
    "https://site.getnet.com.br/link-de-pagamento/",
    "https://site.getnet.com.br/ecommerce/",
    "https://site.getnet.com.br/taxas-personalizadas/",
    "https://site.getnet.com.br/duvidas/",
    "https://site.getnet.com.br/get-ajuda/",
    "https://www.getnet.net/en/",
    "https://www.getnet.net/en/our-solutions/in-person-payments",
    "https://www.getnet.net/en/our-solutions/online-payments",
    "https://www.getnet.net/en/our-solutions/omnichannel",
    "https://www.getnet.net/en/our-solutions/value-added-solutions",
    "https://www.getnet.net/en/our-solutions/agentic-commerce",
    "https://www.getnet.net/en/corporate/about-us",
)
MAX_PAGES = 70
SKIP_PATTERNS = (
    "/legal/",
    "/privacy",
    "/cookies",
    "/press-room/",
    "wp-content",
    "/blog/",
    "/partners",
    "/resources/",
    "/your-business/",
    "/contact",
    "/developers",
    "/redirect",
    "/login",
    "/cadastro",
    "/carrinho",
    "?page_id=",
    "?modal_open=",
    "?ofertas=",
    "/imprensa",
    "/carreiras",
    "/patrocinios",
    "/sustentabilidade",
    "/report-vulnerability",
    "/politica-de-seguranca",
    "/parcerias",
)
SKIP_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png", ".svg", ".gif", ".zip", ".xml")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 Chrome/120 Safari/537.36"
)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def slugify(url: str) -> str:
    """Turn a URL into a safe file name.

    Args:
        url: Absolute URL.

    Returns:
        A lowercase slug made of the host and path.
    """
    parsed = urlparse(url)
    raw = f"{parsed.netloc}{parsed.path}".strip("/")
    return re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-") or "index"


def clean_html(html: str) -> tuple[str, str]:
    """Extract the title and the readable text of a page.

    All visible text is kept, including product cards built from `div` and `span`
    elements, with headings marked and duplicate lines removed.

    Args:
        html: Raw HTML.

    Returns:
        A tuple of (title, text) with navigation, scripts and banners removed.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "iframe", "svg"]):
        tag.decompose()
    for tag in soup.select("[class*=cookie], [id*=cookie], [class*=banner], [aria-hidden=true]"):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    headings = {tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2", "h3", "h4"])}
    body = soup.body or soup
    lines: list[str] = []
    seen: set[str] = set()
    for raw in body.get_text("\n", strip=True).splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < 3 or line in seen:
            continue
        seen.add(line)
        lines.append(f"\n## {line}" if line in headings else line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return title, text


def _allowed(url: str, host: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != host or parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path.lower()
    if path.endswith(SKIP_SUFFIXES) or any(p in url for p in SKIP_PATTERNS):
        return False
    # On the global site only the English section is relevant; other language roots
    # duplicate the same marketing pages.
    if parsed.netloc == "www.getnet.net":
        return path.startswith("/en/") or path.rstrip("/") == ""
    return True


def crawl(seeds: Iterable[str], max_pages: int = MAX_PAGES) -> dict[str, tuple[str, str]]:
    """Fetch the seed pages and the same-host links one level deep.

    Args:
        seeds: Starting URLs.
        max_pages: Upper bound on fetched pages.

    Returns:
        Mapping of URL to (title, text) for every page that produced text.
    """
    pages: dict[str, tuple[str, str]] = {}
    queue: list[tuple[str, int]] = [(url, 0) for url in seeds]
    seen: set[str] = set()
    with httpx.Client(
        timeout=15.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            normalised = url.split("#")[0].rstrip("/") + "/"
            if normalised in seen:
                continue
            seen.add(normalised)
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "fetch failed", extra={"extra_fields": {"url": url, "error": str(exc)}}
                )
                continue
            if "text/html" not in response.headers.get("content-type", ""):
                continue
            title, text = clean_html(response.text)
            if len(text) > 200:
                pages[url] = (title, text)
                logger.info("fetched %s (%d chars)", url, len(text))
            if depth >= 1:
                continue
            host = urlparse(url).netloc
            soup = BeautifulSoup(response.text, "lxml")
            for anchor in soup.find_all("a", href=True):
                link = urljoin(url, anchor["href"]).split("#")[0]
                if _allowed(link, host) and link.rstrip("/") + "/" not in seen:
                    queue.append((link, depth + 1))
    return pages


def save_raw(pages: dict[str, tuple[str, str]], raw_dir: Path) -> None:
    """Write one text file per page with a two-line header (url, title).

    Args:
        pages: Mapping of URL to (title, text).
        raw_dir: Destination directory, recreated on each call.
    """
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)
    for url, (title, text) in pages.items():
        path = raw_dir / f"{slugify(url)}.txt"
        path.write_text(f"url: {url}\ntitle: {title}\n\n{text}\n", encoding="utf-8")


def load_raw(raw_dir: Path) -> list[Document]:
    """Read the stored pages back as LangChain documents.

    Args:
        raw_dir: Directory written by `save_raw`.

    Returns:
        One document per file with `url`, `title` and `lang` metadata.
    """
    documents = []
    for path in sorted(raw_dir.glob("*.txt")):
        header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        meta = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
        url = meta.get("url", path.stem)
        lang = "pt" if "getnet.com.br" in url else "en"
        documents.append(
            Document(
                page_content=body.strip(),
                metadata={"url": url, "title": meta.get("title", ""), "lang": lang},
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks that keep their page metadata.

    Args:
        documents: Page-level documents.

    Returns:
        Chunk-level documents with a `section` metadata field when available.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        match = re.search(r"^## (.+)$", chunk.page_content, re.MULTILINE)
        chunk.metadata["section"] = match.group(1)[:120] if match else ""
    return chunks


def build_index(chunks: list[Document]) -> int:
    """Recreate the Chroma collection from the given chunks.

    Args:
        chunks: Chunk-level documents.

    Returns:
        Number of chunks stored.
    """
    import chromadb
    from langchain_chroma import Chroma

    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        logger.info("no previous collection to delete")
    store = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=llm.embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    batch = 64
    for start in range(0, len(chunks), batch):
        store.add_documents(chunks[start : start + batch])
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the Getnet knowledge index.")
    parser.add_argument(
        "--offline", action="store_true", help="Rebuild from data/raw without crawling."
    )
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raw_dir = get_settings().data_dir / "raw"
    if not args.offline:
        pages = crawl(SEED_URLS, args.max_pages)
        save_raw(pages, raw_dir)
        print(f"pages fetched: {len(pages)}")
    documents = load_raw(raw_dir)
    if not documents:
        print("no raw pages found; run without --offline first")
        return 1
    chunks = split_documents(documents)
    stored = build_index(chunks)
    print(f"pages indexed: {len(documents)}, chunks stored: {stored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
