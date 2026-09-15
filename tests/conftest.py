from pathlib import Path

import pymupdf
import pytest

SAMPLE_MARKDOWN = """# Iris KNN Classification

An end-to-end walkthrough of this project.

## Dataset

This section introduces the **dataset** and the KNN supervised learning algorithm.

## Implementation

```python
class KNN:
    def predict(self, x):
        return self._nearest(x)
```

The implementation separates responsibilities across modules for maintainability.
"""

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Iris KNN Classification</title>
</head>
<body>
<main>
<h1>Iris KNN Classification</h1>
<p>An end-to-end walkthrough of this project.</p>
<h2>Dataset</h2>
<p>This section introduces the dataset and the
<strong>KNN</strong> supervised learning algorithm.</p>
<h2>Implementation</h2>
<pre><code>class KNN:
    def predict(self, x):
        return self._nearest(x)
</code></pre>
<p>The implementation separates responsibilities across modules for maintainability.</p>
</main>
</body>
</html>
"""

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


@pytest.fixture(scope="session")
def sample_markdown(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a known Markdown fixture so tests never depend on external files."""
    markdown_path = tmp_path_factory.mktemp("markdown_fixtures") / (
        "Iris KNN Classification.md"
    )
    markdown_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="session")
def sample_html(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a known HTML fixture so tests never depend on external files."""
    html_path = tmp_path_factory.mktemp("html_fixtures") / (
        "Iris KNN Classification.html"
    )
    html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    return html_path
