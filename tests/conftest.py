from pathlib import Path

import pymupdf
import pytest

PAGE_TEXTS = [
    "Iris KNN Classification Pipeline\nAn end-to-end walkthrough of the\nproject.",
    "This section introduces the dataset and the KNN\nsupervised learning algorithm.",
    "Feature extraction and data preprocessing steps\nare detailed here.",
    "The implementation separates responsibilities\n"
    "across modules for maintainability.",
    "Model training, evaluation metrics, and results\nare covered in depth.",
    "The conclusion summarizes findings and next\nsteps for future work.",
]


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a known PDF fixture so tests never depend on external files."""
    pdf_path = tmp_path_factory.mktemp("pdf_fixtures") / (
        "Iris KNN Classification Pipeline.pdf"
    )

    doc = pymupdf.open()
    try:
        for _index, text in enumerate(PAGE_TEXTS):
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=12)
    finally:
        doc.save(pdf_path)
        doc.close()

    return pdf_path
