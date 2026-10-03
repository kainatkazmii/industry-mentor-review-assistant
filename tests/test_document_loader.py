import pytest

from src.document_loader import load_document, load_docx


SAMPLE_DOCX_PATH = "data/sample_reports/sample_report_1_pcb_defect_detection.docx"


def test_load_docx_returns_nonempty_text():
    text = load_docx(SAMPLE_DOCX_PATH)
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_document_dispatches_to_docx_loader():
    text = load_document(SAMPLE_DOCX_PATH)
    assert "defect" in text.lower()


def test_load_document_raises_on_unsupported_extension():
    with pytest.raises(ValueError):
        load_document("data/sample_reports/not_a_real_file.txt")


def test_load_document_raises_on_missing_file():
    with pytest.raises(Exception):
        load_document("data/sample_reports/does_not_exist.docx")