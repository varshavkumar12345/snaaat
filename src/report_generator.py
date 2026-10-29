"""
src/report_generator.py
------------------------
Generate academic deliverables:
  1. PDF report (ReportLab)
  2. PowerPoint presentation (python-pptx)

Usage
~~~~~
    from src import load_config
    from src.report_generator import ReportGenerator

    cfg = load_config()
    gen = ReportGenerator(cfg)
    gen.generate_pdf(network_stats, eval_df, rec_dfs, image_paths)
    gen.generate_pptx(network_stats, eval_df, rec_dfs, image_paths)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate PDF report and PowerPoint presentation."""

    def __init__(self, config: dict) -> None:
        self._config = config
        report_cfg = config.get("reports", {})
        self._reports_dir = Path(report_cfg.get("output_dir", "outputs/reports"))
        self._pptx_dir = Path(report_cfg.get("presentation_dir", "outputs/presentation"))
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._pptx_dir.mkdir(parents=True, exist_ok=True)
        self._author: str = report_cfg.get("author", "Student")
        self._institution: str = report_cfg.get("institution", "University")
        self._seed_user: str = config.get("github", {}).get("username", "")

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def generate_pdf(
        self,
        network_stats: dict,
        eval_df: pd.DataFrame,
        rec_dfs: Dict[int, pd.DataFrame],
        image_paths: Optional[List[Path]] = None,
    ) -> Path:
        """Generate a PDF report with ReportLab.

        Parameters
        ----------
        network_stats:
            Dict from ``compute_network_stats()``.
        eval_df:
            Algorithm evaluation DataFrame from ``Evaluator.evaluate()``.
        rec_dfs:
            Dict of {k: DataFrame} from ``RecommendationEngine.generate()``.
        image_paths:
            Optional list of paths to PNG/JPG figures to embed.

        Returns
        -------
        Path
            Path to the generated PDF.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Image,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            logger.error("reportlab is not installed. Run: pip install reportlab")
            return self._reports_dir / "report.pdf"

        out = self._reports_dir / "report.pdf"
        doc = SimpleDocTemplate(str(out), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        body = styles["BodyText"]
        body.leading = 14

        story = []

        # ── Title page ────────────────────────────────────────────────
        story.append(Spacer(1, 3*cm))
        story.append(Paragraph("GitHub Developer Collaboration Network", h1))
        story.append(Paragraph("& Link Prediction", h1))
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(f"Seed Username: <b>{self._seed_user}</b>", body))
        story.append(Paragraph(f"Author: {self._author}", body))
        story.append(Paragraph(f"Institution: {self._institution}", body))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body
        ))
        story.append(PageBreak())

        # ── Section 1: Network Statistics ─────────────────────────────
        story.append(Paragraph("1. Network Statistics", h2))
        story.append(Spacer(1, 0.3*cm))
        if network_stats:
            stat_data = [["Metric", "Value"]] + [
                [k.replace("_", " ").title(), str(v)]
                for k, v in network_stats.items()
            ]
            t = Table(stat_data, colWidths=[9*cm, 7*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f0f3f4"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("Network statistics not available.", body))
        story.append(Spacer(1, 0.5*cm))

        # ── Section 2: Algorithm Evaluation ───────────────────────────
        story.append(Paragraph("2. Link Prediction — Algorithm Evaluation", h2))
        story.append(Spacer(1, 0.3*cm))
        if not eval_df.empty:
            cols = ["auc_roc", "avg_precision", "precision", "recall", "f1"]
            headers = ["Algorithm", "AUC-ROC", "Avg Prec", "Precision",
                       "Recall", "F1"]
            eval_data = [headers]
            for algo, row in eval_df.iterrows():
                eval_data.append(
                    [algo] + [f"{row.get(c, 'N/A'):.4f}"
                               if isinstance(row.get(c), float) else "N/A"
                               for c in cols]
                )
            t2 = Table(eval_data, colWidths=[5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm, 2*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#eaf4fb"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 3: Recommendations ────────────────────────────────
        story.append(Paragraph("3. Top-K Collaboration Recommendations", h2))
        for k, rec_df in rec_dfs.items():
            story.append(Paragraph(f"Top-{k} Recommendations", styles["Heading3"]))
            story.append(Spacer(1, 0.2*cm))
            if not rec_df.empty:
                # Show top 10 rows per algorithm
                sub = rec_df[rec_df["rank"] <= min(k, 10)]
                rec_data = [["Rank", "Algorithm", "Candidate", "Score"]]
                for _, row in sub.iterrows():
                    rec_data.append([
                        str(row["rank"]), row["algorithm"],
                        row["candidate"], f"{row['score']:.6f}",
                    ])
                t3 = Table(rec_data, colWidths=[2*cm, 5*cm, 6*cm, 3.5*cm])
                t3.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#117a65")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#eafaf1"), colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(t3)
            story.append(Spacer(1, 0.4*cm))

        # ── Section 4: Figures ────────────────────────────────────────
        if image_paths:
            story.append(PageBreak())
            story.append(Paragraph("4. Figures", h2))
            for img_path in image_paths:
                img_path = Path(img_path)
                if img_path.exists():
                    story.append(Spacer(1, 0.3*cm))
                    story.append(Paragraph(img_path.stem.replace("_", " ").title(),
                                           styles["Heading3"]))
                    story.append(
                        Image(str(img_path), width=15*cm, height=10*cm,
                              kind="proportional")
                    )

        doc.build(story)
        logger.info("PDF report saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # PowerPoint presentation
    # ------------------------------------------------------------------

    def generate_pptx(
        self,
        network_stats: dict,
        eval_df: pd.DataFrame,
        rec_dfs: Dict[int, pd.DataFrame],
        image_paths: Optional[List[Path]] = None,
    ) -> Path:
        """Generate a PowerPoint presentation with python-pptx.

        Parameters
        ----------
        network_stats:
            Dict from ``compute_network_stats()``.
        eval_df:
            Algorithm evaluation DataFrame.
        rec_dfs:
            Recommendation DataFrames keyed by k.
        image_paths:
            List of figure paths to embed as slides.

        Returns
        -------
        Path
            Path to the saved .pptx file.
        """
        try:
            from pptx import Presentation
            from pptx.dml.color import RGBColor
            from pptx.util import Cm, Pt
            from pptx.enum.text import PP_ALIGN
        except ImportError:
            logger.error("python-pptx is not installed. Run: pip install python-pptx")
            return self._pptx_dir / "presentation.pptx"

        prs = Presentation()
        prs.slide_width = Cm(33.87)   # widescreen 16:9
        prs.slide_height = Cm(19.05)

        blank_layout = prs.slide_layouts[6]
        title_layout = prs.slide_layouts[0]

        # ── Helper utilities ──────────────────────────────────────────
        def add_text_box(slide, text, left, top, width, height,
                         font_size=18, bold=False,
                         colour=RGBColor(0x2c, 0x3e, 0x50)):
            txb = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
            tf = txb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = text
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.color.rgb = colour

        def bg_fill(slide, hex_col: str):
            from pptx.util import Pt
            fill = slide.background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor.from_string(hex_col)

        # ── Slide 1: Title ────────────────────────────────────────────
        slide = prs.slides.add_slide(title_layout)
        bg_fill(slide, "2C3E50")
        slide.shapes.title.text = "GitHub Developer Collaboration Network"
        slide.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = (
            RGBColor(0xFF, 0xFF, 0xFF)
        )
        if len(slide.placeholders) > 1:
            sub = slide.placeholders[1]
            sub.text = (
                f"Link Prediction & Social Network Analysis\n"
                f"Seed: {self._seed_user}  |  {self._author}  |  "
                f"{datetime.now().strftime('%Y-%m-%d')}"
            )

        # ── Slide 2: Network Statistics ───────────────────────────────
        slide2 = prs.slides.add_slide(blank_layout)
        bg_fill(slide2, "FDFEFE")
        add_text_box(slide2, "Network Statistics", 1, 0.5, 30, 1.5,
                     font_size=24, bold=True)
        if network_stats:
            lines = [f"• {k.replace('_',' ').title()}: {v}"
                     for k, v in network_stats.items()]
            add_text_box(slide2, "\n".join(lines), 1, 2, 30, 15, font_size=14)

        # ── Slide 3: Algorithm Evaluation ─────────────────────────────
        slide3 = prs.slides.add_slide(blank_layout)
        bg_fill(slide3, "FDFEFE")
        add_text_box(slide3, "Algorithm Evaluation Results", 1, 0.5, 30, 1.5,
                     font_size=24, bold=True)
        if not eval_df.empty:
            lines = []
            for algo, row in eval_df.iterrows():
                lines.append(
                    f"• {algo}: AUC-ROC={row.get('auc_roc','N/A'):.4f}  "
                    f"F1={row.get('f1','N/A'):.4f}  "
                    f"AP={row.get('avg_precision','N/A'):.4f}"
                )
            add_text_box(slide3, "\n".join(lines), 1, 2, 31, 15, font_size=14)

        # ── Slide 4: Top-10 Recommendations ───────────────────────────
        for k, rec_df in rec_dfs.items():
            slide_r = prs.slides.add_slide(blank_layout)
            bg_fill(slide_r, "FDFEFE")
            add_text_box(slide_r, f"Top-{k} Collaboration Recommendations",
                         1, 0.5, 31, 1.5, font_size=22, bold=True)
            if not rec_df.empty:
                sub = rec_df[rec_df["rank"] <= 5]
                lines = [
                    f"  #{int(r['rank'])} [{r['algorithm']}]  "
                    f"→ {r['candidate']}  (score: {r['score']:.4f})"
                    for _, r in sub.iterrows()
                ]
                add_text_box(slide_r, "\n".join(lines), 1, 2.5, 31, 14,
                             font_size=14)

        # ── Slides for figures ────────────────────────────────────────
        if image_paths:
            for img_path in image_paths:
                img_path = Path(img_path)
                if img_path.exists():
                    slide_f = prs.slides.add_slide(blank_layout)
                    bg_fill(slide_f, "FDFEFE")
                    add_text_box(slide_f,
                                 img_path.stem.replace("_", " ").title(),
                                 1, 0.3, 31, 1.2, font_size=20, bold=True)
                    try:
                        slide_f.shapes.add_picture(
                            str(img_path), Cm(1), Cm(1.8), Cm(31), Cm(16)
                        )
                    except Exception as exc:
                        logger.warning("Could not embed image %s: %s", img_path, exc)

        out = self._pptx_dir / "presentation.pptx"
        prs.save(str(out))
        logger.info("PowerPoint saved: %s", out)
        return out

