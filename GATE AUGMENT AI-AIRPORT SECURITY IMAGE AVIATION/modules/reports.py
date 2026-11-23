from __future__ import annotations

import io


def build_pdf_report(report: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF reports require ReportLab. Install reportlab from requirements.txt.") from exc

    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, title="GateAugment AI report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GateAugment AI — Security Imaging Research Report", styles["Title"]),
        Spacer(1, 12),
    ]
    rows = [
        ["Field", "Value"],
        ["Experiment ID", str(report.get("experiment_id") or "Not trained")],
        ["Generated at (UTC)", str(report.get("generated_at", ""))],
        ["Images uploaded", str(report.get("total_uploaded_images", 0))],
        ["Scans completed", str(report.get("total_scans", 0))],
        ["Augmented images", str(report.get("total_augmented_images", 0))],
        ["Synthetic images", str(report.get("total_synthetic_images", 0))],
        ["Augmentation methods", ", ".join(report.get("augmentation_techniques", [])) or "None"],
        ["Generative AI model(s)", ", ".join(report.get("generative_ai_models", [])) or "None / not run"],
        ["Dataset", str(report.get("dataset", "Not configured"))],
        ["Detection model", str(report.get("detection_model", "Unavailable / not run"))],
    ]
    training = report.get("model_evaluation")
    if training:
        rows.extend([
            ["CNN architecture", str(report.get("model_details", {}).get("model_architecture", "CNN"))],
            ["Baseline accuracy", f"{training['baseline']['accuracy']:.4f}"],
            ["Augmented accuracy", f"{training['augmented']['accuracy']:.4f}"],
            ["Baseline macro precision / recall / F1",
             f"{training['baseline']['precision']:.4f} / {training['baseline']['recall']:.4f} / {training['baseline']['f1']:.4f}"],
            ["Augmented macro precision / recall / F1",
             f"{training['augmented']['precision']:.4f} / {training['augmented']['recall']:.4f} / {training['augmented']['f1']:.4f}"],
        ])
    else:
        rows.append(["CNN metrics", "Not available: no model evaluation has been completed."])
    table = Table(rows, colWidths=[205, 325], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2030")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#85A5B5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF5F8")]),
    ]))
    story.extend([table, Spacer(1, 16), Paragraph("Recent scan and processing history", styles["Heading2"])])
    history = [["Time", "Source", "Image", "Status", "Time (ms)"]]
    for item in report.get("scan_history", [])[:20]:
        history.append([
            str(item.get("created_at", ""))[:19],
            str(item.get("source", "")),
            str(item.get("filename", ""))[:30],
            str(item.get("status", "")),
            f"{float(item.get('processing_ms', 0)):.2f}",
        ])
    if len(history) == 1:
        history.append(["No scans recorded", "", "", "", ""])
    history_table = Table(history, colWidths=[95, 75, 170, 80, 65], repeatRows=1)
    history_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2030")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#85A5B5")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(history_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Research-only output. Model detections and generated imagery are not validated for operational airport screening.",
        styles["Italic"],
    ))
    document.build(story)
    return output.getvalue()
