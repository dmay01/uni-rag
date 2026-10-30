"""Generate tiny, original teaching documents. Never imports or reads user sources."""

from pathlib import Path


def make_fixtures(directory: Path) -> None:
    import pymupdf
    from docx import Document
    from pptx import Presentation

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "notes.md").write_text(
        "---\ntitle: Synthetic course\nprivate_tag: hidden\n---\n# Retrieval\n"
        "Hybrid retrieval combines dense vectors and BM25 keywords.\n"
        "See [[evaluation|evaluation notes]]. ![[diagram.png]]\n"
        "## Evaluation\nMRR is the reciprocal rank of the first relevant result.\n",
        encoding="utf-8",
    )
    (directory / "schedule.txt").write_text(
        "The synthetic seminar takes place on Tuesday at 10:00 in room F42.\n"
    )
    (directory / "formula.tex").write_text(
        r"\section{Energy} Energy is the integral of power over time: $E=\int P(t)dt$."
    )
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Synthetic lecture: reciprocal rank fusion uses rank positions.")
    pdf.new_page()  # Synthetic blank page exercises no-text detection.
    pdf.save(directory / "lecture.pdf")
    pdf.close()
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Local RAG"
    slide.placeholders[1].text = "All inference runs on the local machine."
    slide.notes_slide.notes_text_frame.text = (
        "Speaker notes: bge-m3 supports multilingual retrieval."
    )
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Privacy"
    slide.placeholders[1].text = "Source documents are read only."
    deck.save(directory / "slides.pptx")
    doc = Document()
    doc.add_heading("Laboratory policy", level=1)
    doc.add_paragraph("The synthetic laboratory accepts reports in German or English.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Submission"
    table.cell(0, 1).text = "PDF"
    doc.save(directory / "policy.docx")


if __name__ == "__main__":
    make_fixtures(Path(__file__).parent / "fixtures")
