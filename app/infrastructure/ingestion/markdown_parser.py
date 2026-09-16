from pathlib import Path
from typing import ClassVar

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.domain.documents.entities import Document, DocumentPage

_TEXT_TOKEN_TYPES = ("text", "code_inline", "image")
_CONTENT_TOKEN_TYPES = ("code_block", "fence")


class MarkdownParser:
    """Parse Markdown files into the DevRAG document domain model."""

    SUPPORTED_EXTENSIONS: ClassVar[tuple[str, ...]] = (".md", ".markdown")

    def parse(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        markdown = file_path.read_text(encoding="utf-8")
        tokens = MarkdownIt("commonmark", {"html": True}).parse(markdown)

        pages = self._split_sections(tokens)

        return Document(
            title=file_path.stem,
            source="markdown",
            pages=pages,
            metadata={
                "file_name": file_path.name,
                "section_count": str(len(pages)),
            },
        )

    def _split_sections(self, tokens: list[Token]) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        parts: list[str] = []
        heading: str | None = None
        page_number = 0

        def flush() -> None:
            nonlocal page_number, parts, heading
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

        for index, token in enumerate(tokens):
            if token.type == "heading_open":
                flush()
                heading = self._inline_text(tokens[index + 1])
            elif token.type == "inline" and tokens[index - 1].type != "heading_open":
                parts.append(self._inline_text(token))
            elif token.type in _CONTENT_TOKEN_TYPES:
                parts.append(token.content)

        flush()

        return pages

    @staticmethod
    def _inline_text(token: Token) -> str:
        if not token.children:
            return token.content
        return "".join(
            child.content
            for child in token.children
            if child.type in _TEXT_TOKEN_TYPES
        )

    @staticmethod
    def _assemble_page(heading: str | None, parts: list[str]) -> str:
        body = "\n".join(part.strip() for part in parts).strip()
        if not heading:
            return body
        if not body:
            return heading
        return f"{heading}\n{body}"