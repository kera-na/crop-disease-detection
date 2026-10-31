from reportlab.pdfgen import canvas
from io import BytesIO


def create_pdf_report(
        disease,
        confidence,
        severity):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer
    )

    pdf.drawString(
        50,
        800,
        "Crop Disease Report"
    )

    pdf.drawString(
        50,
        770,
        f"Disease: {disease}"
    )

    pdf.drawString(
        50,
        740,
        f"Confidence: {confidence:.2f}%"
    )

    pdf.drawString(
        50,
        710,
        f"Severity: {severity}"
    )

    pdf.save()

    buffer.seek(0)

    return buffer