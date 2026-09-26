"""Render the evidence-pinned Markdown technical report and three study figures.

This creates a synthesis, not new experimental results. Install ``.[paper]``.
The small Markdown renderer intentionally supports the manuscript's documented
headings, paragraphs, lists, tables, code equations, links and local figures.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from html import escape
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import textwrap
from urllib.parse import quote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "b4bdeab08481a25b529d81fa909f96aab10f368f"
PDF_NAME = "judgecal_technical_report.pdf"
FIGURES = ("target_tradeoff", "proxy_targets", "coverage_width")
PLOT_INPUTS = ("reports/estimand/simulation_summary.csv", "reports/rewardbench-audit/summary.csv",
               "reports/finite-sample/summary.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "judgecal-synthesis-v1"})

    def save(fig, name):
        fig.savefig(output/f"{name}.png", dpi=200)
        fig.savefig(output/f"{name}.svg", metadata={"Date": None})
        path = output/f"{name}.svg"
        path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines())+b"\n")
        plt.close(fig)

    data = pd.read_csv(ROOT/PLOT_INPUTS[0])
    cell = data.loc[data.scenario == "iid_p050_positive_n0200_r005"].set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, target, label in zip(axes, ("population", "pool"), ("Population mean", "Random held-out mean")):
        values = cell.loc[["population_tuned", "pool_tuned"], f"{target}_rmse"]*100
        ax.bar([0, 1], values, color=["#416e9b", "#c46a3c"], width=.6)
        for x, value in enumerate(values):
            ax.text(x, value+.12, f"{value:.2f}", ha="center", fontsize=11)
        ax.set(title=label, xticks=[0, 1], xticklabels=["Population tuning", "Pool tuning"],
               ylabel="Target-specific RMSE (pp)", ylim=(0, 5.6))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A coefficient change has opposite effects on two named targets", fontsize=12)
    save(fig, "target_tradeoff")

    data = pd.read_csv(ROOT/PLOT_INPUTS[1]); data = data.loc[data.cohort == "NonLLMBar"]
    methods = ("raw_proxy", "human_only", "ppi", "ppi_tuned", "ppi_signed", "audit_residual")
    names = ("Raw agreement", "Reference audit", "Fixed coefficient 1", "Positive population",
             "Signed population", "Audit residual")
    colors = ("#ad6b35", "#5e6572", "#3575bb", "#009575", "#825bc2", "#ce4660")
    judges = ("openai/gpt-4o-2024-08-06", "openai/gpt-4o-mini-2024-07-18")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    for ax, judge, title in zip(axes, judges, ("GPT-4o (2024-08-06)", "GPT-4o mini (2024-07-18)")):
        for method, name, color in zip(methods, names, colors):
            rows = data.loc[(data.target_judge == judge) & (data.method == method)].sort_values("labeled_fraction")
            ax.plot(100*rows.labeled_fraction, 100*rows.mean_absolute_error, marker="o", color=color, label=name)
        ax.set(title=title, xticks=[20, 40, 60], xlabel="Audited groups (% of cohort)", ylabel="Held-out accuracy MAE (pp)", ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("RewardBench non-LLMBar component: compare methods within each target", fontsize=12)
    save(fig, "proxy_targets")

    data = pd.read_csv(ROOT/PLOT_INPUTS[2])
    cell = data.loc[data.scenario == "iid_rare_strong_n0020"].set_index("method").loc[
        ["human_normal", "ppi_tuned_normal", "human_hoeffding", "ppi_eb_grid"]]
    names = ["Audit\nnormal", "Tuned PPI\nnormal", "Audit\nHoeffding", "Finite-grid\nEB"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    x = np.arange(4); colors = ["#5e6572", "#ce4660", "#3575bb", "#009575"]
    error = np.maximum(0, np.array([cell.coverage-cell.coverage_mc_low, cell.coverage_mc_high-cell.coverage]))
    axes[0].bar(x, 100*cell.coverage, color=colors, yerr=100*error, capsize=3)
    axes[0].axhline(95, color="#252a30", linestyle="--", linewidth=1)
    axes[0].set(ylim=(0, 110), ylabel="Population coverage (%)", title="Nominal target: 95%")
    axes[1].bar(x, cell.mean_width, color=colors)
    axes[1].axhline(1, color="#252a30", linestyle=":", linewidth=1)
    axes[1].set(ylabel="Mean interval width (outcome units)", title="Intervals are not clipped to [0,1]")
    for ax in axes:
        ax.set(xticks=x, xticklabels=names); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Near-boundary mean (p=0.95), 20 audit labels: coverage and width", fontsize=12)
    save(fig, "coverage_width")


def link_url(target, manuscript):
    if target.startswith(("https://", "http://", "mailto:")):
        return target
    path, _, fragment = target.partition("#")
    local = (manuscript.parent/path).resolve()
    try:
        relative = local.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("manuscript links must stay inside the repository") from error
    # Historical result links stay pinned; this new report and its companion
    # documentation did not exist at the evidence snapshot.
    revision = SNAPSHOT if relative.startswith("reports/") and not relative.startswith("reports/synthesis/") else "main"
    return f"https://github.com/Siquan-Wang/llm-judge-calibration/blob/{revision}/{quote(relative)}" + (f"#{fragment}" if fragment else "")


def inline(text, manuscript):
    pattern = r"\[[^\]]+\]\([^\)]+\)|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*"
    parts, start = [], 0
    for match in re.finditer(pattern, text):
        parts.append(escape(text[start:match.start()])); token = match.group()
        if token.startswith("["):
            label, target = token[1:].split("](", 1)
            parts.append(f'<link href="{escape(link_url(target[:-1], manuscript), quote=True)}" color="#245889">{inline(label, manuscript)}</link>')
        elif token.startswith("`"):
            parts.append(f'<font face="Courier" size="9">{escape(token[1:-1])}</font>')
        elif token.startswith("**"):
            parts.append(f"<b>{escape(token[2:-2])}</b>")
        else:
            parts.append(f"<i>{escape(token[1:-1])}</i>")
        start = match.end()
    parts.append(escape(text[start:])); return "".join(parts)


def render_pdf(manuscript, output):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, Preformatted
    width = letter[0]-112
    styles = {
        "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=14.3, spaceAfter=7),
        "reference": ParagraphStyle("reference", fontName="Times-Roman", fontSize=10, leading=12.5, spaceAfter=5),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=13, textColor=colors.HexColor("#163449")),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.3, leading=14, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=11.5),
        "caption": ParagraphStyle("caption", fontName="Times-Italic", fontSize=9.3, leading=12, spaceAfter=10),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=11, leftIndent=9, spaceAfter=9),
        "bullet": ParagraphStyle("bullet", fontName="Times-Roman", fontSize=11, leading=14.3, leftIndent=12, firstLineIndent=-9, spaceAfter=5),
    }
    lines = manuscript.read_text(encoding="utf-8").splitlines(); story = []; index = 0
    in_references = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1; continue
        if line.startswith("```"):
            code = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.extend(textwrap.wrap(lines[index], width=96, replace_whitespace=False, drop_whitespace=False) or [""])
                index += 1
            story.append(Preformatted("\n".join(code), styles["code"])); index += 1; continue
        if line.startswith("#"):
            level = len(line)-len(line.lstrip("#")); title = line[level:].strip()
            if level == 2:
                in_references = title == "References"
            story.append(Paragraph(inline(title, manuscript), styles[f"h{min(level, 3)}"])); index += 1; continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            alt, source = image_match.groups(); local = (manuscript.parent/source).resolve()
            if local.stem in FIGURES:
                local = output/local.name
            picture = Image(str(local)); scale = min(width/picture.imageWidth, 265/picture.imageHeight)
            picture.drawWidth = picture.imageWidth*scale; picture.drawHeight = picture.imageHeight*scale
            block = [Spacer(1, 5), picture]
            next_line = index+1
            while next_line < len(lines) and not lines[next_line].strip():
                next_line += 1
            if next_line < len(lines) and re.match(r"\*?Figure\s+\d", lines[next_line].strip()):
                caption = []
                while next_line < len(lines) and lines[next_line].strip():
                    caption.append(lines[next_line].strip()); next_line += 1
                block.append(Paragraph(inline(" ".join(caption).strip("*"), manuscript), styles["caption"]))
                index = next_line
            else:
                block.append(Paragraph(escape(alt), styles["caption"])); index += 1
            story.append(KeepTogether(block)); continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [item.strip() for item in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append([Paragraph(inline(cell, manuscript), styles["cell"]) for cell in cells])
                index += 1
            count = len(rows[0]); fractions = [.18, .29, .28, .25] if count == 4 else [1/count]*count
            table = Table(rows, colWidths=[width*f for f in fractions], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#788795")),
                ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#d5dce2"))]))
            story += [table, Spacer(1, 9)]; continue
        if re.match(r"^[-*]\s+", line):
            text = line[2:]; index += 1
            while index < len(lines) and lines[index].strip() and not re.match(r"^[-*]\s+", lines[index].strip()):
                text += " "+lines[index].strip(); index += 1
            story.append(Paragraph("- "+inline(text, manuscript), styles["bullet"])); continue
        text = [line]; index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#|\||```|!\[|[-*]\s)", lines[index].strip()):
            text.append(lines[index].strip()); index += 1
        story.append(Paragraph(inline(" ".join(text), manuscript), styles["reference" if in_references else "body"]))

    def footer(pdf, document):
        pdf.saveState(); pdf.setFont("Helvetica", 8); pdf.setFillColor(colors.HexColor("#586775"))
        pdf.drawString(56, 29, f"judgecal | Retrospective technical report | Evidence {SNAPSHOT[:7]}")
        pdf.drawRightString(letter[0]-56, 29, str(document.page)); pdf.restoreState()

    document = SimpleDocTemplate(str(output/PDF_NAME), pagesize=letter, leftMargin=56, rightMargin=56,
                                 topMargin=47, bottomMargin=49, title="Auditing LLM Evaluation with Imperfect Proxies",
                                 author="Siquan Wang", subject="Targets, useful corrections, and failure modes", invariant=1)
    document.build(story, onFirstPage=footer, onLaterPages=footer,
                   canvasmaker=partial(canvas.Canvas, invariant=1, pageCompression=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=ROOT/"reports/synthesis")
    args = parser.parse_args(argv); output = args.output_directory
    subprocess.run([sys.executable, str(ROOT/"examples/research_evidence.py"), "--check"], check=True)
    artifacts = {PDF_NAME} | {f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg")}
    output.mkdir(parents=True, exist_ok=True)
    if {p.name for p in output.iterdir()} - artifacts - {"reproducibility.json"}:
        raise ValueError("output directory contains unrelated or stale artifacts")
    manuscript = ROOT/"docs/TECHNICAL_REPORT.md"
    make_figures(output); render_pdf(manuscript, output)
    from pypdf import PdfReader
    reader = PdfReader(output/PDF_NAME)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if any(len((page.extract_text() or "").strip()) < 100 for page in reader.pages):
        raise ValueError("unexpected blank or nearly empty report page")
    if any(token in text for token in ("PENDING_", "{{", "PLACEHOLDER")):
        raise ValueError("unresolved report placeholder")
    inputs = list(PLOT_INPUTS) + ["docs/TECHNICAL_REPORT.md", "docs/research_evidence.json", "docs/RESEARCH_EVIDENCE.md",
                                "docs/METHODS.md", "examples/research_evidence.py", "examples/build_research_report.py", "pyproject.toml"]
    manifest = {"kind": "retrospective synthesis; no new experiment", "evidence_snapshot": SNAPSHOT,
                "python": platform.python_version(), "platform": platform.system(),
                "packages": {name: version(name) for name in ("pandas", "matplotlib", "reportlab", "pypdf", "pillow")},
                "pages": len(reader.pages), "source_text_normalization": "CRLF to LF before SHA-256",
                "inputs_sha256": {name: hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in inputs},
                "artifacts_sha256": {name: digest(output/name) for name in sorted(artifacts)}}
    (output/"reproducibility.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode("utf-8"))
    print(f"Wrote {len(reader.pages)} report pages and three evidence-linked figures.")


if __name__ == "__main__":
    main()
