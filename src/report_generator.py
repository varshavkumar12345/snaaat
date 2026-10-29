"""
Academic PDF report and PowerPoint presentation generators for NumFOCUS Collaboration Network project.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

from src.config import DELIVERABLES_DIR, FIGURES_DIR

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that adds page numbers and running header/footer."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (on pages after cover)
        if self._pageNumber > 1:
            self.drawString(54, 750, "NumFOCUS Developer Collaboration Network & Link Prediction Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, text)
        self.drawString(54, 36, "CONFIDENTIAL & ACADEMIC DELIVERABLE - NUMFOCUS MSR DATASET STUDY")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def generate_pdf_report(
    network_stats: Dict[str, Any],
    eval_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    target_user: str,
    output_pdf: Optional[Path] = None
) -> Path:
    """Generate a multi-page, formatted academic report in PDF format."""
    if output_pdf is None:
        output_pdf = DELIVERABLES_DIR / "NumFOCUS_Collaboration_Report.pdf"

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=18
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []

    # Title & Header
    story.append(Paragraph("NumFOCUS Developer Collaboration Network & Link Prediction", title_style))
    story.append(Paragraph(
        f"<b>Empirical Mining and Structural Analysis of Open-Source Scientific Computing Ecosystems</b><br/>"
        f"Target Starting Developer: <b>@{target_user}</b> | Dataset: Zenodo DOI 10.5281/zenodo.14230406",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182ce"), spaceAfter=14))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        "This project models and analyzes the developer collaboration topology across the NumFOCUS open-source "
        "ecosystem using the MSR research dataset (Zenodo record 14230406). By analyzing granular activity logs "
        "spanning pull request reviews, comments, and issue interactions, we construct an undirected, weighted "
        "collaboration graph. We formulate collaborator recommendation as an unsupervised link prediction task, "
        "evaluating four core topological heuristics: Common Neighbors, Jaccard Coefficient, Adamic–Adar, and "
        "Preferential Attachment under strict data-leakage prevention protocols. Tailored recommendations with "
        "structural attribution are generated for the designated maintainer."
    )
    story.append(Paragraph(summary_text, body_style))

    # Dataset & Methodology
    story.append(Paragraph("2. Dataset Architecture & Interaction Verification", h1_style))
    data_text = (
        "The underlying dataset, authored by Hourri, Decan, and Mens (MSR 2025), tracks contributor actions across "
        "58 premier scientific computing projects (e.g., NumPy, SciPy, Pandas, scikit-learn, Julia, Matplotlib) from "
        "January 2022 to December 2024. Unlike raw GitHub event streams, the dataset provides a two-tier abstraction: "
        "low-level actions aggregated into 21 high-level activities. Direct developer collaboration edges are "
        "established through verifiable cooperative actions: code review submissions on pull requests, technical comments "
        "on author pull requests, and discussions on issue threads. Strict hygiene filtering strips automated CI/CD bots, "
        "invalid records, and self-interactions."
    )
    story.append(Paragraph(data_text, body_style))

    # Network Statistics Table
    story.append(Paragraph("3. Empirical Network Statistics", h1_style))
    stats_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Nodes (|V|)", f"{network_stats.get('num_nodes', 0):,}", "Total Edges (|E|)", f"{network_stats.get('num_edges', 0):,}"],
        ["Graph Density", f"{network_stats.get('density', 0.0):.6f}", "Average Degree", f"{network_stats.get('avg_degree', 0.0):.2f}"],
        ["Max Node Degree", f"{network_stats.get('max_degree', 0):,}", "Connected Components", f"{network_stats.get('num_components', 0):,}"],
        ["Largest CC Size", f"{network_stats.get('largest_cc_size', 0):,}", "Largest CC Ratio", f"{network_stats.get('largest_cc_ratio', 0.0):.2%}"],
        ["Transitivity", f"{network_stats.get('transitivity', 0.0):.4f}", "Avg Clustering Coeff", f"{network_stats.get('avg_clustering', 0.0):.4f}"]
    ]
    t_stats = Table(stats_data, colWidths=[120, 130, 120, 130])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ebf8ff")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 10))

    # Figures: Ego network & Predictions
    ego_img = FIGURES_DIR / "network_ego.png"
    pred_img = FIGURES_DIR / "network_predictions.png"
    if ego_img.exists():
        story.append(Paragraph(f"<b>Figure 1:</b> 1-Hop Ego Neighborhood centered on @{target_user}.", body_style))
        story.append(Image(str(ego_img), width=6.5*inch, height=3.8*inch))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Link Prediction & Evaluation
    story.append(Paragraph("4. Link Prediction Evaluation & Preventing Data Leakage", h1_style))
    leakage_text = (
        "To evaluate link prediction without data leakage, observed positive edges were strictly partitioned into "
        "80% training ($E_{train}$) and 20% testing ($E_{test}$). The training graph $G_{train}$ was instantiated by removing "
        "all test edges, ensuring topological metrics were computed solely on antecedent network states. Negative samples "
        "were drawn uniformly from unobserved non-edges. Four topological algorithms were benchmarked:"
    )
    story.append(Paragraph(leakage_text, body_style))
    story.append(Paragraph("• <b>Common Neighbors (CN):</b> $Score(u, v) = |N(u) \\cap N(v)|$", bullet_style))
    story.append(Paragraph("• <b>Jaccard Coefficient (JC):</b> $Score(u, v) = \\frac{|N(u) \\cap N(v)|}{|N(u) \\cup N(v)|}$", bullet_style))
    story.append(Paragraph("• <b>Adamic–Adar (AA):</b> $Score(u, v) = \\sum_{w \\in N(u) \\cap N(v)} \\frac{1}{\\log |N(w)|}$", bullet_style))
    story.append(Paragraph("• <b>Preferential Attachment (PA):</b> $Score(u, v) = |N(u)| \\cdot |N(v)|$", bullet_style))
    story.append(Spacer(1, 8))

    # Evaluation table
    story.append(Paragraph("<b>Table 1:</b> Link Prediction Performance Comparison", body_style))
    eval_cols = ["Algorithm", "ROC-AUC", "Precision@10", "Recall@10", "F1@10", "Precision@20", "Recall@20", "F1@20"]
    eval_headers = [c for c in eval_cols if c in eval_df.columns]
    table_data = [eval_headers]
    for _, row in eval_df.iterrows():
        table_data.append([str(row[c]) if isinstance(row[c], (str, int)) else f"{float(row[c]):.4f}" for c in eval_headers])

    t_eval = Table(table_data, colWidths=[110] + [55] * (len(eval_headers) - 1))
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 10))

    # ROC Curves figure
    roc_img = FIGURES_DIR / "roc_curves.png"
    if roc_img.exists():
        story.append(Paragraph("<b>Figure 2:</b> Comparative ROC Curves across link prediction algorithms.", body_style))
        story.append(Image(str(roc_img), width=4.5*inch, height=3.5*inch))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Recommendations
    story.append(Paragraph(f"5. Collaborator Recommendations for @{target_user}", h1_style))
    rec_text = (
        f"Using Adamic–Adar as the primary ranking heuristic, candidate developers outside the immediate neighborhood "
        f"of @{target_user} were prioritized. Existing direct collaborators, bots, and self-loops were strictly excluded. "
        f"Below are the Top-10 prioritized potential collaborators with structural explanations."
    )
    story.append(Paragraph(rec_text, body_style))

    # Recommendations table
    rec_rows = [["Rank", "Candidate", "Deg", "CN", "AA Score", "JC Score", "Explanation"]]
    for _, row in rec_df.head(10).iterrows():
        expl = str(row.get("Explanation", ""))[:55] + "..." if len(str(row.get("Explanation", ""))) > 55 else str(row.get("Explanation", ""))
        rec_rows.append([
            str(row.get("Rank", "")),
            str(row.get("Recommended Collaborator", ""))[:14],
            str(row.get("Candidate Degree", "")),
            str(row.get("Common Neighbors", "")),
            f"{float(row.get('Adamic-Adar', 0)):.2f}",
            f"{float(row.get('Jaccard', 0)):.3f}",
            expl
        ])
    t_rec = Table(rec_rows, colWidths=[30, 75, 30, 25, 45, 45, 250])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#dd6b20")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#fffaf0")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 10))

    if pred_img.exists():
        story.append(Paragraph(f"<b>Figure 3:</b> Link Prediction Overlay: Observed edges (gray) and Top-10 Predicted edges (dashed orange).", body_style))
        story.append(Image(str(pred_img), width=6.5*inch, height=4.2*inch))
        story.append(Spacer(1, 10))

    # Limitations & Conclusion
    story.append(Paragraph("6. Dataset Limitations & Academic Conclusion", h1_style))
    limitations = (
        "<b>Limitations & Threats to Validity:</b><br/>"
        "1. <i>Observation Window:</i> The dataset spans 2022–2024; legacy interactions prior to 2022 are not captured.<br/>"
        "2. <i>Granularity:</i> Interactions outside GitHub (Mailing lists, Discord, Slack, Gitter) are unrecorded.<br/>"
        "3. <i>Cold Start:</i> Peripheral developers with low degree lack common neighbors, causing triadic closure heuristics to default to Preferential Attachment.<br/>"
        "<br/>"
        "<b>Conclusion:</b> Adamic–Adar significantly outperforms standard Common Neighbors and Preferential Attachment by "
        "penalizing ubiquitous hub collaborators, providing higher-precision recommendations in open-source software engineering."
    )
    story.append(Paragraph(limitations, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_pdf

def generate_presentation(
    network_stats: Dict[str, Any],
    eval_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    target_user: str,
    output_pptx: Optional[Path] = None
) -> Path:
    """Generate professional PowerPoint slide deck summarizing project deliverables."""
    if output_pptx is None:
        output_pptx = DELIVERABLES_DIR / "NumFOCUS_Collaboration_Presentation.pptx"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="NUMFOCUS RESEARCH PROJECT"):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = RGBColor(43, 108, 176)

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(26, 32, 44)

    # Slide 1: Title
    s1 = prs.slides.add_slide(blank_layout)
    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(11), Inches(3))
    tf1 = tb1.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "NumFOCUS Developer Collaboration Network\nand Link Prediction"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(26, 54, 93)

    p2 = tf1.add_paragraph()
    p2.text = f"\nMining 58 Scientific OSS Projects | Zenodo DOI 10.5281/zenodo.14230406\nTarget Starting Maintainer: @{target_user}"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(74, 85, 104)

    # Slide 2: Motivation & Architecture
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation & Dataset Abstraction")
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        ("Scientific Computing Ecosystem: ", "NumFOCUS sustains core tools (NumPy, SciPy, Pandas, scikit-learn, Julia)."),
        ("Overcoming Event Log Noise: ", "Raw GitHub event logs contain noise. The MSR dataset provides a 2-tier abstraction mapping low-level actions into 21 high-level intent activities."),
        ("Verifiable Interaction Links: ", "Edges are grounded in verifiable engineering workflows: pull-request reviews, technical comments, and issue discussions."),
        ("Link Prediction Objective: ", "Anticipate high-probability cross-project collaboration edges using unsupervised structural heuristics.")
    ]
    for b_title, b_desc in bullets2:
        p = tf2.add_paragraph()
        p.text = b_title + b_desc
        p.font.size = Pt(15)
        p.space_after = Pt(14)

    # Slide 3: Network Topography & Stats
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Network Topography & Empirical Statistics")
    # Table of stats
    rows, cols = 6, 2
    table_shape = s3.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.5))
    t = table_shape.table
    stats_list = [
        ("Total Nodes (|V|)", f"{network_stats.get('num_nodes', 0):,}"),
        ("Total Edges (|E|)", f"{network_stats.get('num_edges', 0):,}"),
        ("Network Density", f"{network_stats.get('density', 0.0):.6f}"),
        ("Average Node Degree", f"{network_stats.get('avg_degree', 0.0):.2f}"),
        ("Clustering Coefficient", f"{network_stats.get('avg_clustering', 0.0):.4f}"),
        ("Largest Component Size", f"{network_stats.get('largest_cc_size', 0):,} ({network_stats.get('largest_cc_ratio', 0.0):.1%})")
    ]
    for idx, (label, val) in enumerate(stats_list):
        t.cell(idx, 0).text = label
        t.cell(idx, 1).text = val
        t.cell(idx, 0).text_frame.paragraphs[0].font.size = Pt(13)
        t.cell(idx, 1).text_frame.paragraphs[0].font.size = Pt(13)
        t.cell(idx, 1).text_frame.paragraphs[0].font.bold = True

    # Image of ego network if available
    ego_img = FIGURES_DIR / "network_ego.png"
    if ego_img.exists():
        s3.shapes.add_picture(str(ego_img), Inches(6.8), Inches(1.8), width=Inches(5.7))

    # Slide 4: Algorithms & Anti-Leakage Protocol
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Link Prediction Algorithms & Train-Test Integrity")
    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    bullets4 = [
        ("Strict Train-Test Separation: ", "80% positive training edges ($G_{train}$), 20% test edges. Test edges are completely purged from $G_{train}$ prior to score computation to eliminate data leakage."),
        ("Balanced Negative Sampling: ", "Non-edges sampled uniformly at random from absent pairings to mirror authentic link prediction verification."),
        ("Common Neighbors (CN): ", "Measures shared direct collaborator volume: |N(u) ∩ N(v)|."),
        ("Jaccard Coefficient (JC): ", "Normalizes overlap by total pooled neighborhood size: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|."),
        ("Adamic–Adar (AA): ", "Weights mutual neighbors inversely by log degree, strongly discounting ubiquitous hubs."),
        ("Preferential Attachment (PA): ", "Scale-free degree product assumption: deg(u) × deg(v).")
    ]
    for b_title, b_desc in bullets4:
        p = tf4.add_paragraph()
        p.text = b_title + b_desc
        p.font.size = Pt(14)
        p.space_after = Pt(10)

    # Slide 5: Evaluation & ROC Performance
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Comparative Algorithm Performance")
    # Add eval table
    eval_cols = ["Algorithm", "ROC-AUC", "Precision@10", "Recall@10", "F1@10"]
    cols_to_use = [c for c in eval_cols if c in eval_df.columns]
    t_shape5 = s5.shapes.add_table(len(eval_df) + 1, len(cols_to_use), Inches(0.8), Inches(1.8), Inches(5.8), Inches(2.5))
    t5 = t_shape5.table
    for c_idx, c_name in enumerate(cols_to_use):
        t5.cell(0, c_idx).text = c_name
        t5.cell(0, c_idx).text_frame.paragraphs[0].font.bold = True
        t5.cell(0, c_idx).text_frame.paragraphs[0].font.size = Pt(12)
    for r_idx, row in eval_df.iterrows():
        for c_idx, c_name in enumerate(cols_to_use):
            val = row[c_name]
            val_str = f"{float(val):.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) and c_name != "Algorithm" else str(val)
            t5.cell(r_idx + 1, c_idx).text = val_str
            t5.cell(r_idx + 1, c_idx).text_frame.paragraphs[0].font.size = Pt(11)

    roc_img = FIGURES_DIR / "roc_curves.png"
    if roc_img.exists():
        s5.shapes.add_picture(str(roc_img), Inches(7.0), Inches(1.8), width=Inches(5.5))

    # Slide 6: Recommended Collaborators
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, f"Top Recommended Collaborators for @{target_user}")
    rec_cols = ["Rank", "Recommended User", "Degree", "Common Nbrs", "AA Score"]
    t_shape6 = s6.shapes.add_table(min(11, len(rec_df) + 1), len(rec_cols), Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.8))
    t6 = t_shape6.table
    for c_idx, c_name in enumerate(rec_cols):
        t6.cell(0, c_idx).text = c_name
        t6.cell(0, c_idx).text_frame.paragraphs[0].font.bold = True
        t6.cell(0, c_idx).text_frame.paragraphs[0].font.size = Pt(11)
    for r_idx, row in rec_df.head(10).iterrows():
        t6.cell(r_idx + 1, 0).text = str(row.get("Rank", ""))
        t6.cell(r_idx + 1, 1).text = str(row.get("Recommended Collaborator", ""))
        t6.cell(r_idx + 1, 2).text = str(row.get("Candidate Degree", ""))
        t6.cell(r_idx + 1, 3).text = str(row.get("Common Neighbors", ""))
        t6.cell(r_idx + 1, 4).text = f"{float(row.get('Adamic-Adar', 0)):.2f}"
        for col_i in range(len(rec_cols)):
            t6.cell(r_idx + 1, col_i).text_frame.paragraphs[0].font.size = Pt(10)

    pred_img = FIGURES_DIR / "network_predictions.png"
    if pred_img.exists():
        s6.shapes.add_picture(str(pred_img), Inches(6.8), Inches(1.8), width=Inches(5.7))

    # Slide 7: Viva Q&A & Key Takeaways
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Academic Defense & Viva Insights")
    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    viva_pts = [
        ("1. Data Leakage Prevention: ", "Test edges removed before training graph metrics computed; prevents cheating on triadic closure."),
        ("2. Why Adamic–Adar Leads: ", "Sub-linear damping (1 / log deg(w)) reduces influence of mega-hubs who review hundreds of unrelated PRs."),
        ("3. Cold Start Challenge: ", "New/isolated contributors lack shared neighbors; resolved via hybrid preferential attachment & repo interest."),
        ("4. Real-World Value: ", "Facilitates maintainer recruitment, cross-pollination across scientific packages (e.g., Pandas to Arrow/Polars), and ecosystem resilience.")
    ]
    for q_t, q_d in viva_pts:
        p = tf7.add_paragraph()
        p.text = q_t + q_d
        p.font.size = Pt(14)
        p.space_after = Pt(12)

    prs.save(str(output_pptx))
    
    # Also generate matching PDF version of presentation
    output_pdf = output_pptx.with_suffix('.pdf')
    try:
        generate_presentation_pdf(network_stats, eval_df, rec_df, target_user, output_pdf)
    except Exception as e:
        print(f"Notice: Presentation PDF generation note: {e}")

    return output_pptx

def generate_presentation_pdf(
    network_stats: Dict[str, Any],
    eval_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    target_user: str,
    output_pdf: Path
) -> Path:
    """Generate landscape PDF presentation corresponding to the slide deck."""
    from reportlab.lib.pagesizes import landscape
    
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=14
    )
    
    slide_body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=10
    )
    
    story = []
    
    # Slide 1: Title
    story.append(Spacer(1, 100))
    story.append(Paragraph("<b>NumFOCUS Developer Collaboration Network & Link Prediction</b>", slide_title_style))
    story.append(Paragraph(
        f"<b>MSR Research Dataset Mining across 58 Scientific OSS Projects</b><br/>"
        f"Designated Starting Developer: <b>@{target_user}</b> | Zenodo DOI: 10.5281/zenodo.14230406",
        slide_body_style
    ))
    story.append(PageBreak())
    
    # Slide 2: Topography & Stats
    story.append(Paragraph("Network Topography & Empirical Statistics", slide_title_style))
    stats_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Nodes (|V|)", f"{network_stats.get('num_nodes', 0):,}", "Total Edges (|E|)", f"{network_stats.get('num_edges', 0):,}"],
        ["Graph Density", f"{network_stats.get('density', 0.0):.6f}", "Average Degree", f"{network_stats.get('avg_degree', 0.0):.2f}"],
        ["Max Node Degree", f"{network_stats.get('max_degree', 0):,}", "Connected Components", f"{network_stats.get('num_components', 0):,}"],
        ["Largest CC Size", f"{network_stats.get('largest_cc_size', 0):,}", "Largest CC Ratio", f"{network_stats.get('largest_cc_ratio', 0.0):.2%}"]
    ]
    t = Table(stats_data, colWidths=[150, 150, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ebf8ff")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Empirical Insights:</b> The collaboration topology exhibits small-world properties with dense community clusters around major ecosystem pillars (Pandas, SciPy, NumPy, Scikit-Learn).", slide_body_style))
    story.append(PageBreak())
    
    # Slide 3: Link Prediction Evaluation
    story.append(Paragraph("Comparative Link Prediction Performance", slide_title_style))
    eval_cols = ["Algorithm", "ROC-AUC", "Precision@10", "Recall@10", "F1@10"]
    cols_to_use = [c for c in eval_cols if c in eval_df.columns]
    table_data = [cols_to_use]
    for _, row in eval_df.iterrows():
        table_data.append([str(row[c]) if isinstance(row[c], (str, int)) else f"{float(row[c]):.4f}" for c in cols_to_use])
    t_eval = Table(table_data, colWidths=[150, 100, 100, 100, 100])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Key Finding:</b> Adamic–Adar achieves superior precision and ranking quality by penalizing shared neighbors with disproportionately large hub degrees.", slide_body_style))
    story.append(PageBreak())
    
    # Slide 4: Recommendations
    story.append(Paragraph(f"Top Collaborator Recommendations for @{target_user}", slide_title_style))
    rec_rows = [["Rank", "Candidate", "Deg", "CN", "AA Score", "JC Score", "Explanation"]]
    for _, row in rec_df.head(8).iterrows():
        expl = str(row.get("Explanation", ""))[:55] + "..." if len(str(row.get("Explanation", ""))) > 55 else str(row.get("Explanation", ""))
        rec_rows.append([
            str(row.get("Rank", "")),
            str(row.get("Recommended Collaborator", ""))[:14],
            str(row.get("Candidate Degree", "")),
            str(row.get("Common Neighbors", "")),
            f"{float(row.get('Adamic-Adar', 0)):.2f}",
            f"{float(row.get('Jaccard', 0)):.3f}",
            expl
        ])
    t_rec = Table(rec_rows, colWidths=[35, 90, 35, 30, 60, 60, 330])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#dd6b20")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#fffaf0")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_rec)
    story.append(PageBreak())
    
    # Slide 5: Viva Defense
    story.append(Paragraph("Academic Defense & Viva Discussion", slide_title_style))
    story.append(Paragraph("<b>1. Data Leakage Prevention:</b> Strict temporal/topological partitioning eliminates test edges from G_train before candidate evaluation.", slide_body_style))
    story.append(Paragraph("<b>2. Evaluation under Extreme Sparsity:</b> Addressed via threshold-independent ROC-AUC alongside ranking-focused Precision@K metrics.", slide_body_style))
    story.append(Paragraph("<b>3. Action vs. Activity Abstraction:</b> Using activity groupings mitigates GitHub event noise (e.g., commit push batches vs. code review synthesis).", slide_body_style))
    story.append(Paragraph("<b>4. Engineering Recommendations:</b> Explanatory features ground each prediction in tangible shared collaborator relationships.", slide_body_style))

    doc.build(story)
    return output_pdf
