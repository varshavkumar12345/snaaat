"""
src/report_generator.py
------------------------
Phase 7: Academic PDF Report and PowerPoint Presentation Generation.

Produces:
  1. Full academic evaluation report (ReportLab PDF).
  2. Slide presentation deck (python-pptx).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates academic deliverables: PDF report and PowerPoint presentation."""

    def __init__(self, config: dict) -> None:
        self._config = config
        report_cfg = config.get("reports", {})
        self._reports_dir = Path(report_cfg.get("output_dir", "outputs/reports"))
        self._pptx_dir = Path(report_cfg.get("presentation_dir", "outputs/presentation"))
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._pptx_dir.mkdir(parents=True, exist_ok=True)
        self._seed_user = config.get("github", {}).get("username", "")

    def generate_pdf_report(
        self,
        network_stats: Dict[str, Any],
        eval_df: pd.DataFrame,
        recommendations_df: pd.DataFrame,
        image_paths: Optional[List[Path]] = None,
        filename: str = "collaboration_network_report.pdf",
    ) -> Path:
        """Generate comprehensive academic PDF report using ReportLab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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

        out_pdf = self._reports_dir / filename
        doc = SimpleDocTemplate(
            str(out_pdf), pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle", parent=styles["Heading1"],
            fontSize=22, leading=26, textColor=colors.HexColor("#2c3e50"),
            spaceAfter=15,
        )
        h2 = styles["Heading2"]
        h3 = styles["Heading3"]
        body = styles["BodyText"]
        body.leading = 14

        story = []

        # ── Title Block ──────────────────────────────────────────────
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph("GitHub Developer Collaboration Network & Link Prediction", title_style))
        story.append(Paragraph("<i>Academic Project Technical Evaluation Report</i>", h3))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"<b>Starting Seed Developer:</b> {self._seed_user}", body))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
        story.append(Spacer(1, 0.8 * cm))

        # ── Section 1: Methodology & Problem Statement ────────────────
        story.append(Paragraph("1. Research Objective & Methodology", h2))
        methodology_text = (
            "This project analyzes social network structures in public GitHub activity. "
            "Given a starting developer, the application constructs an ego-centered developer "
            "collaboration network based exclusively on verifiable, dyadic peer interactions: "
            "Pull Request reviews, inline review comments, issue discussion threads, PR mergers, "
            "and collaborator additions. Edges are weighted by interaction frequency and tracked with "
            "complete provenance (event ID, timestamp, and repository)."
        )
        story.append(Paragraph(methodology_text, body))
        story.append(Spacer(1, 0.4 * cm))

        # ── Section 2: Graph Statistics Table ─────────────────────────
        story.append(Paragraph("2. Collaboration Network Topology", h2))
        stat_rows = [["Network Metric", "Value"]]
        for k, v in network_stats.items():
            stat_rows.append([k.replace("_", " ").title(), str(v)])

        t_stats = Table(stat_rows, colWidths=[9 * cm, 7 * cm])
        t_stats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_stats)
        story.append(Spacer(1, 0.6 * cm))

        # ── Section 3: Link Prediction Benchmark ──────────────────────
        story.append(Paragraph("3. Link Prediction Algorithm Benchmark", h2))
        eval_intro = (
            "Four classical topological link prediction heuristics were evaluated on consistent "
            "leak-free test candidates (80% training / 20% testing split with balanced negative sampling): "
            "Common Neighbors (CN), Jaccard Coefficient (JC), Adamic-Adar (AA), and Preferential Attachment (PA)."
        )
        story.append(Paragraph(eval_intro, body))
        story.append(Spacer(1, 0.3 * cm))

        if not eval_df.empty:
            headers = ["Algorithm", "ROC-AUC", "PR-AUC", "Prec@10", "Rec@10", "F1"]
            eval_rows = [headers]
            for algo, row in eval_df.iterrows():
                eval_rows.append([
                    str(algo).replace("_", " ").title(),
                    f"{row.get('roc_auc', 0.0):.4f}",
                    f"{row.get('pr_auc', 0.0):.4f}",
                    f"{row.get('precision@10', 0.0):.2f}",
                    f"{row.get('recall@10', 0.0):.4f}",
                    f"{row.get('f1_score', 0.0):.4f}",
                ])

            t_eval = Table(eval_rows, colWidths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
            t_eval.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8f8f5"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t_eval)

        story.append(Spacer(1, 0.6 * cm))

        # ── Section 4: Top Recommendations & Explanations ─────────────
        story.append(Paragraph("4. Recommended Collaborators & Structural Rationale", h2))
        rec_intro = (
            "Recommendations exclude existing neighbors, self-loops, and automated bots. "
            "Scores represent topological proximity heuristics rather than guaranteed collaboration probabilities."
        )
        story.append(Paragraph(rec_intro, body))
        story.append(Spacer(1, 0.3 * cm))

        if not recommendations_df.empty:
            rec_headers = ["Rank", "Candidate", "Agreed Algos", "Structural Explanation"]
            rec_rows = [rec_headers]
            for _, r in recommendations_df.head(10).iterrows():
                rec_rows.append([
                    str(r.get("consensus_rank", "")),
                    str(r.get("candidate", "")),
                    str(r.get("algorithms_supporting", "")),
                    str(r.get("summary_rationale", ""))[:65] + "...",
                ])

            t_recs = Table(rec_rows, colWidths=[1.5 * cm, 3.5 * cm, 2.5 * cm, 9.5 * cm])
            t_recs.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ebf5fb"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_recs)

        story.append(Spacer(1, 0.6 * cm))

        # ── Section 5: Limitations ────────────────────────────────────
        story.append(Paragraph("5. Limitations & Threats to Validity", h2))
        limitations = (
            "1. <b>Temporal Truncation:</b> Hourly archives capture only real-time public actions; "
            "historical connections formed outside the collection window are missing without API backfills.<br/>"
            "2. <b>Sparsity:</b> New or quiet developers with few public interactions yield small ego graphs, "
            "which causes metrics like Average Clustering to reflect single triangles.<br/>"
            "3. <b>Unresolved Commit Logins:</b> Git commits contain local author emails rather than GitHub logins; "
            "only direct platform interactions were parsed to prevent false attribution."
        )
        story.append(Paragraph(limitations, body))

        # ── Section 6: Embedded Figures ───────────────────────────────
        if image_paths:
            story.append(PageBreak())
            story.append(Paragraph("6. Network Visualizations", h2))
            for img in image_paths:
                p = Path(img)
                if p.exists() and p.suffix.lower() == ".png":
                    story.append(Spacer(1, 0.3 * cm))
                    story.append(Paragraph(p.stem.replace("_", " ").title(), h3))
                    story.append(Image(str(p), width=16 * cm, height=11 * cm, kind="proportional"))

        doc.build(story)
        logger.info("PDF Report generated successfully at: %s", out_pdf)
        return out_pdf

    def generate_presentation(
        self,
        network_stats: Dict[str, Any],
        eval_df: pd.DataFrame,
        recommendations_df: pd.DataFrame,
        image_paths: Optional[List[Path]] = None,
        filename: str = "project_presentation.pptx",
    ) -> Path:
        """Generate academic slide deck using python-pptx."""
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Cm, Pt

        prs = Presentation()
        prs.slide_width = Cm(33.87)   # 16:9 widescreen
        prs.slide_height = Cm(19.05)
        blank_layout = prs.slide_layouts[6]

        def add_header(slide, title_text):
            tb = slide.shapes.add_textbox(Cm(1.5), Cm(0.8), Cm(30), Cm(1.5))
            tf = tb.text_frame
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

        # Slide 1: Title
        s1 = prs.slides.add_slide(blank_layout)
        tb1 = s1.shapes.add_textbox(Cm(2), Cm(5), Cm(29), Cm(7))
        tf1 = tb1.text_frame
        p1 = tf1.paragraphs[0]
        p1.text = "GitHub Developer Collaboration Network & Link Prediction"
        p1.font.size = Pt(28)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

        p1_sub = tf1.add_paragraph()
        p1_sub.text = f"Social Network Analysis (SNA) Academic Project | Seed User: {self._seed_user}"
        p1_sub.font.size = Pt(18)
        p1_sub.font.color.rgb = RGBColor(0x16, 0xA0, 0x85)

        # Slide 2: Problem & Methodology
        s2 = prs.slides.add_slide(blank_layout)
        add_header(s2, "Problem Statement & Interaction Provenance")
        tb2 = s2.shapes.add_textbox(Cm(2), Cm(2.8), Cm(29), Cm(14))
        tf2 = tb2.text_frame
        bullets = [
            "• Objective: Predict plausible future developer collaborations centered on a starting user.",
            "• Ground Truth Source: Public GitHub Archive (hourly compressed JSON-lines dumps).",
            "• Strict Edge Definition: Rejects arbitrary 'same-repo' commits. Focuses on dyadic actions:",
            "    - PR Reviews & Review Comments (code review interactions)",
            "    - Issue Discussion comments & assignments",
            "    - Pull request merges by repository maintainers",
            "• Provenance Tracking: Every edge retains source event ID, repository, timestamp, and relation.",
        ]
        for b in bullets:
            p = tf2.add_paragraph()
            p.text = b
            p.font.size = Pt(16)
            p.space_after = Pt(10)

        # Slide 3: Algorithms & Evaluation
        s3 = prs.slides.add_slide(blank_layout)
        add_header(s3, "Link Prediction Benchmark & Metrics")
        tb3 = s3.shapes.add_textbox(Cm(2), Cm(2.8), Cm(29), Cm(14))
        tf3 = tb3.text_frame
        p = tf3.add_paragraph()
        p.text = "Evaluated on leak-free 80/20 train-test splits with balanced negative non-edge samples:"
        p.font.size = Pt(16)
        p.space_after = Pt(12)

        if not eval_df.empty:
            for algo, r in eval_df.iterrows():
                p_algo = tf3.add_paragraph()
                p_algo.text = (
                    f"• {str(algo).replace('_',' ').title()}: "
                    f"ROC-AUC = {r.get('roc_auc',0.0):.4f} | "
                    f"PR-AUC = {r.get('pr_auc',0.0):.4f} | "
                    f"Prec@10 = {r.get('precision@10',0.0):.2f} | "
                    f"F1 = {r.get('f1_score',0.0):.4f}"
                )
                p_algo.font.size = Pt(15)
                p_algo.space_after = Pt(8)

        # Slide 4: Recommendations
        s4 = prs.slides.add_slide(blank_layout)
        add_header(s4, "Top-5 Collaboration Recommendations")
        tb4 = s4.shapes.add_textbox(Cm(2), Cm(2.8), Cm(29), Cm(14))
        tf4 = tb4.text_frame
        if not recommendations_df.empty:
            for _, r in recommendations_df.head(5).iterrows():
                p_r = tf4.add_paragraph()
                p_r.text = (
                    f"#{r.get('consensus_rank')} Candidate: {r.get('candidate')} "
                    f"({r.get('algorithms_supporting')} algorithms agreeing)\n"
                    f"   Rationale: {r.get('summary_rationale')}"
                )
                p_r.font.size = Pt(14)
                p_r.space_after = Pt(12)

        # Slide 5: Visualizations
        if image_paths:
            for img in image_paths:
                p = Path(img)
                if p.exists() and p.suffix.lower() == ".png":
                    s_img = prs.slides.add_slide(blank_layout)
                    add_header(s_img, p.stem.replace("_", " ").title())
                    try:
                        s_img.shapes.add_picture(str(p), Cm(2), Cm(2.5), Cm(29), Cm(15))
                    except Exception as exc:
                        logger.warning("Could not add slide picture %s: %s", p, exc)

        out_pptx = self._pptx_dir / filename
        prs.save(str(out_pptx))
        logger.info("PowerPoint deck generated at: %s", out_pptx)
        return out_pptx
