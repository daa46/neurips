from __future__ import annotations

from scripts.make_method_figure import save_method_figure


def test_method_figure_is_vector_and_anonymous(tmp_path):
    pdf, png = save_method_figure(tmp_path / "method", png_dpi=72)

    assert pdf.is_file() and pdf.stat().st_size > 1_000
    assert png.is_file() and png.stat().st_size > 1_000
    pdf_bytes = pdf.read_bytes().lower()
    assert b"/home/" not in pdf_bytes
    assert b"denali" not in pdf_bytes
    assert b"/subtype /type3" not in pdf_bytes
