"""Small, offline regression checks for the manuscript PDF renderer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest


@pytest.fixture
def report_builder(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/build_research_report.py"
    spec = importlib.util.spec_from_file_location("research_report_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", root)
    return module


@pytest.fixture
def manuscript(report_builder):
    path = report_builder.ROOT / "docs/note.md"
    path.write_text(
        """# A small research note

XML inequalities A < B & C > D remain text. **Audit outcome** and *prediction pool*
have distinct meanings; the inline residual is `Y - lambda F`.
See [`mean(Y_L)`](../reports/example/REPORT.md#target),
[API guide](API_GUIDE.md#means), and [external paper](https://example.org/paper?q=a&b=c).

## Equations

```text
theta(lambda) = mean(Y_L) + lambda * (mean(F_U) - mean(F_L))
Var = Var(Y - lambda F)/n + lambda^2 Var(F)/N
0 <= lambda <= 1; A < B & C > D
```

| Rule | Target | Error | Scope |
| --- | --- | --- | --- |
| Fixed PPI | Population mean | RMSE 0.031 | IID audit |
| Pool tuning | Realized pool | RMSE 0.032 | Marginal prediction |

- Both errors use the same observed draws.
  The hidden outcomes enter scoring only.

## References

[1] A. Author. A reproducible audit. 2026. [Source](https://example.org/reference).
""",
        encoding="utf-8",
    )
    return path


def _render(report_builder, manuscript, output):
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    output.mkdir()
    report_builder.render_pdf(manuscript, output)
    path = output / report_builder.PDF_NAME
    return path, pypdf.PdfReader(path)


def test_pdf_preserves_equations_text_tables_and_link_destinations(
    report_builder, manuscript, tmp_path
):
    _, reader = _render(report_builder, manuscript, tmp_path / "render")
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() for page in reader.pages))
    for expected in (
        "A small research note",
        "XML inequalities A < B & C > D remain text.",
        "Audit outcome and prediction pool have distinct meanings",
        "the inline residual is Y - lambda F",
        "mean(Y_L)",
        "theta(lambda) = mean(Y_L) + lambda * (mean(F_U) - mean(F_L))",
        "Var = Var(Y - lambda F)/n + lambda^2 Var(F)/N",
        "0 <= lambda <= 1; A < B & C > D",
        "Rule Target Error Scope",
        "Fixed PPI Population mean RMSE 0.031 IID audit",
        "Pool tuning Realized pool RMSE 0.032 Marginal prediction",
        "Both errors use the same observed draws. The hidden outcomes enter scoring only.",
        "[1] A. Author. A reproducible audit. 2026. Source",
    ):
        assert expected in text
    assert "<font" not in text
    assert "`mean(Y_L)`" not in text

    urls = {
        str(annotation.get_object()["/A"]["/URI"])
        for page in reader.pages
        for annotation in page.get("/Annots", [])
        if annotation.get_object().get("/A", {}).get("/S") == "/URI"
    }
    base = "https://github.com/Siquan-Wang/llm-judge-calibration/blob/"
    assert urls == {
        base + report_builder.SNAPSHOT + "/reports/example/REPORT.md#target",
        base + "main/docs/API_GUIDE.md#means",
        "https://example.org/paper?q=a&b=c",
        "https://example.org/reference",
    }


def test_pdf_bytes_do_not_depend_on_output_directory_or_repeat_render(
    report_builder, manuscript, tmp_path, monkeypatch
):
    pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    # Distinct wall-clock times make this catch timestamp regressions even when
    # both tiny renders complete within the same second. An environment-level
    # reproducible-build override must not conceal a missing renderer setting.
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr("reportlab.lib.utils.time.time", lambda: 1_700_000_000.0)
    first, _ = _render(report_builder, manuscript, tmp_path / "first")
    original = first.read_bytes()
    monkeypatch.setattr("reportlab.lib.utils.time.time", lambda: 1_800_000_000.0)
    second, _ = _render(report_builder, manuscript, tmp_path / "second")
    assert original.startswith(b"%PDF-")
    assert original == second.read_bytes()
    report_builder.render_pdf(manuscript, first.parent)
    assert original == first.read_bytes()


@pytest.mark.parametrize("target", ["../../outside.md", "../reports/../../../outside.md"])
def test_repository_link_escape_is_rejected(report_builder, manuscript, target):
    with pytest.raises(ValueError, match="stay inside the repository"):
        report_builder.link_url(target, manuscript)
    with pytest.raises(ValueError, match="stay inside the repository"):
        report_builder.inline(f"[untrusted link]({target})", manuscript)


def test_absolute_link_escape_is_rejected(report_builder, manuscript, tmp_path):
    with pytest.raises(ValueError, match="stay inside the repository"):
        report_builder.link_url((tmp_path / "outside.md").as_posix(), manuscript)


def test_local_links_preserve_fragments_and_quote_paths(report_builder, manuscript):
    base = "https://github.com/Siquan-Wang/llm-judge-calibration/blob/"
    assert report_builder.link_url("../reports/case file.csv#row-2", manuscript) == (
        base + report_builder.SNAPSHOT + "/reports/case%20file.csv#row-2"
    )
    assert report_builder.link_url("../reports/synthesis/report.pdf", manuscript) == (
        base + "main/reports/synthesis/report.pdf"
    )
    assert report_builder.link_url("mailto:author@example.org", manuscript) == (
        "mailto:author@example.org"
    )
