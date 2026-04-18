from fpdf import FPDF
import sys

try:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(200, 10, text="Diagnostic PDF Test", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(200, 10, text="If you can see this text, PDF generation is working.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.output("diagnostic.pdf")
    print("PDF generated successfully: diagnostic.pdf")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
