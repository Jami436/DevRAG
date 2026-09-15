from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype, NavigableString, Tag

from app.domain.documents.entities import Document, DocumentPage

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_NO_CONTENT_TAGS = {
    "head",
    "iframe",
    "meta",
    "noscript",
    "script",
    "style",
    "template",
    "title",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class HTMLParser:
    """Parse HTML files into the DevRAG document domain model."""

    def parse(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() not in (".html", ".htm"):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        soup = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")

        title = self._extract_title(soup)
        pages = self._split_sections(soup)

        return Document(
            title=title or file_path.stem,
            source="html",
            pages=pages,
            metadata={
                "file_name": file_path.name,
                "section_count": str(len(pages)),
            },
        )

    def _split_sections(self, soup: BeautifulSoup) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        parts: list[str] = []
        heading: str | None = None
        heading_parts: list[str] = []
        in_heading = False
        previous_block: Tag | None = None
        page_number = 0

        def flush() -> None:
            nonlocal page_number, parts, heading, heading_parts, in_heading
            nonlocal previous_block
            if parts:
                page_number += 1
                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        content=self._assemble_page(heading, parts),
                        metadata={"heading": heading or ""},
                    )
                )
            parts = []
            heading = None
            heading_parts = []
            in_heading = False
            previous_block = None

        for descendant in soup.descendants:
            if not isinstance(descendant, NavigableString):
                continue
            if isinstance(descendant, (Comment, Doctype)):
                continue

            text = str(descendant)
            if not text or text.isspace():
                continue

            parent = descendant.parent
            if not isinstance(parent, Tag) or parent.name in _NO_CONTENT_TAGS:
                continue

            if self._in_heading(parent):
                if not in_heading:
                    flush()
                    in_heading = True
                heading_parts.append(text)
                continue

            if in_heading:
                heading = " ".join(heading_parts)
                heading_parts = []
                in_heading = False
                previous_block = None

            block = self._nearest_block(parent)
            if block is not previous_block and parts:
                parts.append("\n")
            previous_block = block
            parts.append(text)

        flush()

        return pages

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str | None:
        tag = soup.find("title")
        if not isinstance(tag, Tag):
            return None
        text = tag.get_text(" ", strip=True)
        return text or None

    @staticmethod
    def _in_heading(parent: Tag) -> bool:
        if parent.name in _HEADING_TAGS:
            return True
        return parent.find_parent(_HEADING_TAGS) is not None

    @staticmethod
    def _nearest_block(tag: Tag) -> Tag:
        current: Tag | None = tag
        while current is not None:
            if current.name in _BLOCK_TAGS:
                return current
            parent = current.parent
            current = parent if isinstance(parent, Tag) else None
        return tag

    @staticmethod
    def _assemble_page(heading: str | None, parts: list[str]) -> str:
        body = "".join(parts).strip()
        if not heading:
            return body
        if not body:
            return heading
        return f"{heading}\n{body}"