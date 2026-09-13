from pathlib import Path

import pymupdf

from app.domain.documents.entities import Document, DocumentPage


class PDFParser:
    """Parse PDF files into the DevRAG document domain model."""

    def parse(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        pdf = pymupdf.open(file_path)
        pages: list[DocumentPage] = []

        try:
            for index in range(pdf.page_count):
                page = pdf.load_page(index)
                pages.append(
                    DocumentPage(
                        page_number=index + 1,
                        content=page.get_text().strip(),
                    )
                )
        finally:
            pdf.close()

        return Document(
            title=file_path.stem,
            source="pdf",
            pages=pages,
            metadata={
                "file_name": file_path.name,
                "page_count": str(len(pages)),
            },
        )
