"""
IEEE-format paper generator for:
"GRAPES-SHAP: A Twelve-Stage Clinical QA Framework with
 Calibrated World-Model Reasoning and Shapley Source Attribution"
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, HRFlowable, KeepTogether,
    PageBreak, Flowable
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, Arrow
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
import os

# ── Colour palette ──────────────────────────────────────────────────────────
C_BLACK   = colors.black
C_WHITE   = colors.white
C_GRAY    = HexColor('#666666')
C_LGRAY   = HexColor('#CCCCCC')
C_LLGRAY  = HexColor('#F2F2F2')
C_BLUE    = HexColor('#003366')   # IEEE deep blue
C_LBLUE   = HexColor('#4472C4')
C_TEAL    = HexColor('#2F7A6E')
C_ORANGE  = HexColor('#C55A11')
C_RED     = HexColor('#C00000')

# ── Page geometry (IEEE double-column) ──────────────────────────────────────
PAGE_W, PAGE_H = letter          # 8.5 × 11 in
ML = 0.625*inch; MR = 0.625*inch
MT = 0.75*inch;  MB = 1.0*inch
COL_GAP  = 0.25*inch
COL_W    = (PAGE_W - ML - MR - COL_GAP) / 2   # ~3.5 in

# ── Style definitions ────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    # Paper title
    title = S('IEEETitle',
        fontName='Times-Bold', fontSize=24,
        leading=28, alignment=TA_CENTER,
        textColor=C_BLACK, spaceAfter=6)

    # Author block
    authors = S('IEEEAuthors',
        fontName='Times-Roman', fontSize=10,
        leading=13, alignment=TA_CENTER,
        textColor=C_BLACK, spaceAfter=2)

    # Affiliation
    affil = S('IEEEAffil',
        fontName='Times-Italic', fontSize=9,
        leading=11, alignment=TA_CENTER,
        textColor=C_GRAY, spaceAfter=6)

    # Section heading
    sec = S('IEEESec',
        fontName='Times-Bold', fontSize=10,
        leading=12, alignment=TA_CENTER,
        spaceBefore=8, spaceAfter=3,
        textColor=C_BLACK,
        borderPad=0)

    # Sub-section heading
    subsec = S('IEEESubsec',
        fontName='Times-Bold-Italic', fontSize=9.5,
        leading=12, alignment=TA_LEFT,
        spaceBefore=5, spaceAfter=2,
        textColor=C_BLACK)

    # Sub-sub-section (run-in)
    subsubsec = S('IEEESubsubsec',
        fontName='Times-Bold', fontSize=9,
        leading=11, alignment=TA_LEFT,
        spaceBefore=3, spaceAfter=1,
        textColor=C_BLACK)

    # Normal body text
    body = S('IEEEBody',
        fontName='Times-Roman', fontSize=9,
        leading=11, alignment=TA_JUSTIFY,
        spaceAfter=4)

    # Abstract text (smaller, indented)
    abst = S('IEEEAbstract',
        fontName='Times-Roman', fontSize=8.5,
        leading=10.5, alignment=TA_JUSTIFY,
        leftIndent=0.25*inch, rightIndent=0.25*inch,
        spaceAfter=4)

    # Keyword text
    kw = S('IEEEKeywords',
        fontName='Times-Italic', fontSize=8.5,
        leading=10.5, alignment=TA_JUSTIFY,
        leftIndent=0.25*inch, rightIndent=0.25*inch,
        spaceAfter=6)

    # Caption
    cap = S('IEEECaption',
        fontName='Times-Roman', fontSize=8,
        leading=10, alignment=TA_CENTER,
        textColor=C_BLACK, spaceAfter=4, spaceBefore=2)

    # Table header
    thdr = S('IEEETableHdr',
        fontName='Times-Bold', fontSize=8,
        leading=10, alignment=TA_CENTER,
        textColor=C_WHITE)

    # Table cell
    tcell = S('IEEETableCell',
        fontName='Times-Roman', fontSize=8,
        leading=10, alignment=TA_LEFT)

    # Reference
    ref = S('IEEERef',
        fontName='Times-Roman', fontSize=8,
        leading=9.5, alignment=TA_JUSTIFY,
        leftIndent=12, firstLineIndent=-12,
        spaceAfter=2)

    # Equation label
    eq = S('IEEEEq',
        fontName='Times-Roman', fontSize=9,
        leading=11, alignment=TA_CENTER,
        spaceAfter=3, spaceBefore=3)

    # Footer
    foot = S('IEEEFooter',
        fontName='Times-Roman', fontSize=7,
        leading=9, alignment=TA_CENTER,
        textColor=C_GRAY)

    return dict(title=title, authors=authors, affil=affil,
                sec=sec, subsec=subsec, subsubsec=subsubsec,
                body=body, abst=abst, kw=kw, cap=cap,
                thdr=thdr, tcell=tcell, ref=ref, eq=eq, foot=foot)


# ── Custom flowable: numbered section title with rule ──────────────────────
class SectionHeading(Flowable):
    def __init__(self, number, text, width=COL_W):
        super().__init__()
        self.number = number
        self.text   = text
        self._w     = width
        self._h     = 18

    def wrap(self, aW, aH):
        return self._w, self._h

    def draw(self):
        c = self.canv
        c.saveState()
        # Top rule
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(0.8)
        c.line(0, self._h-2, self._w, self._h-2)
        # Text
        c.setFont('Times-Bold', 10)
        c.setFillColor(C_BLACK)
        label = f"{self.number}. {self.text.upper()}"
        c.drawCentredString(self._w/2, self._h - 14, label)
        # Bottom rule
        c.setLineWidth(0.3)
        c.line(0, 0, self._w, 0)
        c.restoreState()


# ── Custom flowable: simple architecture diagram ────────────────────────────
class PipelineDiagram(Flowable):
    """12-stage inference pipeline block diagram."""
    def __init__(self, width=COL_W, height=1.6*inch):
        super().__init__()
        self._w = width
        self._h = height

    def wrap(self, aW, aH):
        return self._w, self._h

    def draw(self):
        c = self.canv
        c.saveState()

        stages = [
            ("Query\nExpand\n(HyDE)", C_TEAL),
            ("Hybrid\nRetrieval\nBM25+Dense", C_TEAL),
            ("Cross-Enc\nRe-rank\n+MMR", C_TEAL),
            ("Causal\nKG-GNN\nEncoder", C_LBLUE),
            ("Latent\nWorld\nModel", C_BLUE),
            ("ToT Beam\nPlanner\nW=8 H=4", C_BLUE),
            ("Deep\nEnsemble\n(M=5)", C_ORANGE),
            ("Hall.\nSelf-check", C_ORANGE),
            ("SHAP\nAttrib.\nϕ(d)", C_RED),
            ("LLM\nVerbal.\n(DeepSeek)", C_GRAY),
        ]

        n     = len(stages)
        bw    = (self._w - 0.05*inch) / n   # box width
        bh    = self._h * 0.62
        by    = self._h * 0.22
        arrow = 4

        for i, (label, col) in enumerate(stages):
            x = i * bw
            # Box
            c.setFillColor(col)
            c.setStrokeColor(C_WHITE)
            c.setLineWidth(0.5)
            c.roundRect(x+1, by, bw-2, bh, 3, fill=1, stroke=1)
            # Text
            c.setFillColor(C_WHITE)
            c.setFont('Helvetica-Bold', 5.0)
            lines = label.split('\n')
            lh = 6
            total = len(lines)*lh
            ty = by + bh/2 + total/2 - lh*0.8
            for ln in lines:
                c.drawCentredString(x + bw/2, ty, ln)
                ty -= lh
            # Arrow
            if i < n-1:
                ax = x + bw - 1
                ay = by + bh/2
                c.setStrokeColor(C_BLACK)
                c.setLineWidth(0.6)
                c.line(ax, ay, ax+2, ay)
                c.setFillColor(C_BLACK)
                c.setStrokeColor(C_BLACK)
                p = c.beginPath()
                p.moveTo(ax+2, ay+2)
                p.lineTo(ax+5, ay)
                p.lineTo(ax+2, ay-2)
                p.close()
                c.drawPath(p, fill=1, stroke=0)

        # Band labels below
        bands = [
            (0, 3, "Retrieval", C_TEAL),
            (3, 4, "Reasoning", C_LBLUE),
            (4, 6, "World-Model Planning", C_BLUE),
            (6, 8, "Uncertainty", C_ORANGE),
            (8, 10, "Explanation", C_RED),
        ]
        c.setFont('Helvetica', 4.8)
        for (s, e, lbl, col) in bands:
            x1 = s * bw + 1
            x2 = e * bw - 1
            c.setFillColor(col)
            c.rect(x1, by - 10, x2-x1, 7, fill=1, stroke=0)
            c.setFillColor(C_WHITE)
            c.drawCentredString((x1+x2)/2, by - 7, lbl)

        c.restoreState()


# ── Custom flowable: neural architecture block diagram ─────────────────────
class ArchDiagram(Flowable):
    """Simplified neural architecture overview."""
    def __init__(self, width=COL_W, height=2.0*inch):
        super().__init__()
        self._w = width
        self._h = height

    def wrap(self, aW, aH):
        return self._w, self._h

    def draw(self):
        c = self.canv
        c.saveState()

        W, H = self._w, self._h
        # Background
        c.setFillColor(C_LLGRAY)
        c.setStrokeColor(C_LGRAY)
        c.rect(0, 0, W, H, fill=1, stroke=1)

        def box(x, y, w, h, col, label, sublabel="", fontsize=6):
            c.setFillColor(col)
            c.setStrokeColor(C_WHITE)
            c.setLineWidth(0.5)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            c.setFillColor(C_WHITE)
            c.setFont('Helvetica-Bold', fontsize)
            c.drawCentredString(x+w/2, y+h/2+3, label)
            if sublabel:
                c.setFont('Helvetica', fontsize-1)
                c.drawCentredString(x+w/2, y+h/2-4, sublabel)

        def arrow_h(x1, y, x2):
            c.setStrokeColor(C_GRAY)
            c.setLineWidth(0.6)
            c.line(x1, y, x2-4, y)
            c.setFillColor(C_GRAY)
            p = c.beginPath()
            p.moveTo(x2-4, y+2); p.lineTo(x2, y); p.lineTo(x2-4, y-2); p.close()
            c.drawPath(p, fill=1, stroke=0)

        bh = 28; pad = 4
        # Row 1 – inputs (left column)
        bw1 = W*0.15
        box(pad, H-bh*1.1, bw1, bh, C_TEAL, "Context", "obs∈R^{T×64}", 5.5)
        box(pad, H-bh*2.3, bw1, bh, C_TEAL, "Action", "a∈{0..49}", 5.5)
        box(pad, H-bh*3.5, bw1, bh, C_TEAL, "KG", "20 nodes", 5.5)

        # Core blocks (middle)
        x2 = pad + bw1 + 8
        bw2 = W*0.14
        box(x2, H-bh*1.8, bw2, bh*1.2, C_LBLUE, "Causal", "GAT×3", 5.5)
        x3 = x2 + bw2 + 6
        box(x3, H*0.58, bw2, bh*1.5, C_BLUE, "Evidence", "Fusion Enc.", 5.5)
        x4 = x3 + bw2 + 6
        box(x4, H*0.55, bw2, bh*1.6, C_BLUE, "Causal", "Residual Δz", 5.5)
        x5 = x4 + bw2 + 6
        box(x5, H*0.52, bw2, bh*1.8, HexColor('#1a4a7a'), "GRU", "3-layer h=512", 5.5)

        # Output heads (right)
        x6 = x5 + bw2 + 8
        bw3 = W*0.12
        box(x6, H-bh*1.1,  bw3, bh*0.85, C_ORANGE,  "Decoder", "δ(64)", 5.5)
        box(x6, H-bh*2.1,  bw3, bh*0.85, C_ORANGE,  "σ-head",  "aleatoric", 5.5)
        box(x6, H-bh*3.1,  bw3, bh*0.85, C_RED,     "Reward",  "r(z,a)", 5.5)
        box(x6, H-bh*4.15, bw3, bh*0.85, C_RED,     "Ensemble","5×(μ,σ)", 5.5)

        # Bottom modules
        bw4 = W*0.27
        yb  = pad
        box(pad,     yb, bw4, 20, C_TEAL,  "Shapley Source Tracking  ϕᵢ per doc", fontsize=5.5)
        box(pad+bw4+6, yb, bw4, 20, C_BLUE, "Beam Planner (ToT)  W=8, H=4", fontsize=5.5)
        box(pad+2*bw4+12, yb, W-pad-2*bw4-16, 20, C_ORANGE, "Deep Ensemble  calibrated σ", fontsize=5.5)

        # Title
        c.setFont('Helvetica-Bold', 6)
        c.setFillColor(C_BLACK)
        c.drawCentredString(W/2, H-6, "Detailed Neural Architecture · Context Labels & Source Tracking")

        c.restoreState()


# ── Custom flowable: Results bar chart ─────────────────────────────────────
class ResultsChart(Flowable):
    def __init__(self, width=COL_W, height=1.5*inch):
        super().__init__()
        self._w = width
        self._h = height

    def wrap(self, aW, aH):
        return self._w, self._h

    def draw(self):
        c = self.canv
        c.saveState()

        metrics  = ["Concept\nCoverage", "Structure\nComplete.", "Evidence\nCitations/6"]
        baseline = [0.70, 0.50, 0.33]
        proposed = [0.97, 0.84, 0.85]

        W, H  = self._w, self._h
        ox, oy = 0.55*inch, 0.35*inch   # origin
        cw = (W - ox - 0.1*inch) / (3*2 + 3 + 1)  # cluster width
        bh_max = H - oy - 0.2*inch

        c.setFont('Helvetica', 4.5)
        # Y-axis ticks
        for v in [0, 0.25, 0.5, 0.75, 1.0]:
            y = oy + v * bh_max
            c.setStrokeColor(C_LGRAY)
            c.setLineWidth(0.3)
            c.line(ox, y, W-0.05*inch, y)
            c.setFillColor(C_GRAY)
            c.drawRightString(ox-3, y-2, f"{v:.2f}")

        # Bars
        gap_between = 6
        group_w = 2*cw*0.85 + gap_between
        total_groups = 3
        total_used = total_groups * group_w + (total_groups-1)*10

        start_x = ox + (W - ox - 0.05*inch - total_used) / 2

        for i, (m, bv, pv) in enumerate(zip(metrics, baseline, proposed)):
            gx = start_x + i*(group_w + 10)
            bw = cw * 0.82

            # Baseline bar
            bary = oy + bv * bh_max
            c.setFillColor(C_LGRAY)
            c.setStrokeColor(C_GRAY)
            c.setLineWidth(0.3)
            c.rect(gx, oy, bw, bv*bh_max, fill=1, stroke=1)

            # Proposed bar
            c.setFillColor(C_BLUE)
            c.rect(gx+bw+gap_between, oy, bw, pv*bh_max, fill=1, stroke=1)

            # Value labels
            c.setFont('Helvetica-Bold', 4.5)
            c.setFillColor(C_BLACK)
            c.drawCentredString(gx+bw/2, oy+bv*bh_max+2, f"{bv:.2f}")
            c.setFillColor(C_WHITE)
            c.drawCentredString(gx+bw+gap_between+bw/2, oy+pv*bh_max-7, f"{pv:.2f}")

            # X-label
            c.setFillColor(C_BLACK)
            c.setFont('Helvetica', 4.5)
            lines = m.split('\n')
            lx = gx + bw + gap_between/2
            for li, ln in enumerate(lines):
                c.drawCentredString(lx, oy - 10 + li*6, ln)

        # Axes
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(0.7)
        c.line(ox, oy, ox, H-0.15*inch)
        c.line(ox, oy, W-0.05*inch, oy)

        # Legend
        lx = ox
        ly = H - 0.14*inch
        c.setFillColor(C_LGRAY)
        c.rect(lx, ly, 8, 6, fill=1, stroke=0)
        c.setFillColor(C_BLACK)
        c.setFont('Helvetica', 4.5)
        c.drawString(lx+10, ly+1, "Baseline RAG")
        lx2 = lx + 70
        c.setFillColor(C_BLUE)
        c.rect(lx2, ly, 8, 6, fill=1, stroke=0)
        c.setFillColor(C_BLACK)
        c.drawString(lx2+10, ly+1, "Proposed Framework")

        c.restoreState()


# ── Custom flowable: Reliability diagram ──────────────────────────────────
class ReliabilityDiagram(Flowable):
    def __init__(self, width=COL_W*0.85, height=1.3*inch):
        super().__init__()
        self._w = width
        self._h = height

    def wrap(self, aW, aH):
        return self._w, self._h

    def draw(self):
        c = self.canv
        c.saveState()

        W, H  = self._w, self._h
        ox, oy = 0.45*inch, 0.35*inch
        pw = W - ox - 0.1*inch
        ph = H - oy - 0.1*inch

        # Axes
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(0.7)
        c.line(ox, oy, ox, oy+ph)
        c.line(ox, oy, ox+pw, oy)

        # Grid & ticks
        c.setFont('Helvetica', 4.2)
        for v in [0, 0.05, 0.10, 0.15, 0.20]:
            y = oy + (v/0.20)*ph
            x = ox + (v/0.20)*pw
            c.setStrokeColor(C_LGRAY)
            c.setLineWidth(0.25)
            c.line(ox, y, ox+pw, y)
            c.setFillColor(C_GRAY)
            c.drawRightString(ox-2, y-2, f"{v:.2f}")
            c.drawCentredString(x, oy-7, f"{v:.2f}")

        # Perfect calibration line (dashed)
        c.setStrokeColor(C_GRAY)
        c.setLineWidth(0.8)
        c.setDash([3, 3])
        c.line(ox, oy, ox+pw, oy+ph)
        c.setDash([])

        # Proposed data points
        pts = [(0.00,0.00),(0.03,0.025),(0.05,0.05),(0.07,0.065),
               (0.10,0.100),(0.15,0.138),(0.20,0.145)]
        c.setStrokeColor(C_BLUE)
        c.setLineWidth(1.0)
        path = c.beginPath()
        for i,(px,py) in enumerate(pts):
            xi = ox + (px/0.20)*pw
            yi = oy + (py/0.20)*ph
            if i==0: path.moveTo(xi, yi)
            else:    path.lineTo(xi, yi)
        c.drawPath(path)
        # Dots
        c.setFillColor(C_BLUE)
        for (px,py) in pts:
            xi = ox + (px/0.20)*pw
            yi = oy + (py/0.20)*ph
            c.circle(xi, yi, 2, fill=1, stroke=0)

        # Axis labels
        c.setFont('Helvetica', 4.5)
        c.setFillColor(C_BLACK)
        c.drawCentredString(ox+pw/2, oy-14, "Predicted uncertainty σ")
        c.saveState()
        c.translate(ox-28, oy+ph/2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Empirical |error|")
        c.restoreState()

        # Legend
        c.setFont('Helvetica', 4.2)
        c.setStrokeColor(C_GRAY); c.setLineWidth(0.8); c.setDash([3,3])
        c.line(ox+pw-60, oy+ph-5, ox+pw-50, oy+ph-5)
        c.setDash([])
        c.setFillColor(C_GRAY)
        c.drawString(ox+pw-48, oy+ph-7, "Perfect calib.")
        c.setStrokeColor(C_BLUE); c.setLineWidth(1.0)
        c.line(ox+pw-60, oy+ph-14, ox+pw-50, oy+ph-14)
        c.setFillColor(C_BLUE)
        c.drawString(ox+pw-48, oy+ph-16, "Proposed")

        c.restoreState()


# ── Page template (two-column) ──────────────────────────────────────────────
def make_doc(filename):
    doc = BaseDocTemplate(
        filename,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title="GRAPES-SHAP: Clinical QA Framework",
        author="[Author Names]",
        subject="IEEE Paper"
    )

    col1 = Frame(ML, MB, COL_W, PAGE_H-MT-MB, id='col1', showBoundary=0)
    col2 = Frame(ML+COL_W+COL_GAP, MB, COL_W, PAGE_H-MT-MB, id='col2', showBoundary=0)

    # Full-width frame for title/abstract
    full = Frame(ML, MB, PAGE_W-ML-MR, PAGE_H-MT-MB, id='full', showBoundary=0)

    def page_draw_title(c, doc):
        """First page: header rule + conference/page footer."""
        c.saveState()
        c.setFont('Times-Roman', 7)
        c.setFillColor(C_GRAY)
        hdr = "2026 IEEE Conference on Biomedical Engineering and Informatics · GRAPES-SHAP"
        c.drawCentredString(PAGE_W/2, PAGE_H-MT+0.25*inch, hdr)
        c.setLineWidth(0.5)
        c.setStrokeColor(C_BLACK)
        c.line(ML, PAGE_H-MT+0.15*inch, PAGE_W-MR, PAGE_H-MT+0.15*inch)
        foot = "978-X-XXXX-XXXX-X/26/$31.00 © 2026 IEEE"
        c.drawString(ML, MB-0.3*inch, foot)
        c.drawRightString(PAGE_W-MR, MB-0.3*inch, "1")
        c.restoreState()

    def page_draw_rest(c, doc):
        c.saveState()
        c.setFont('Times-Roman', 7)
        c.setFillColor(C_GRAY)
        c.setLineWidth(0.5)
        c.setStrokeColor(C_BLACK)
        c.line(ML, PAGE_H-MT+0.15*inch, PAGE_W-MR, PAGE_H-MT+0.15*inch)
        hdr = "2026 IEEE Conference on Biomedical Engineering and Informatics · GRAPES-SHAP"
        c.drawCentredString(PAGE_W/2, PAGE_H-MT+0.25*inch, hdr)
        c.drawCentredString(PAGE_W/2, MB-0.3*inch, str(doc.page))
        c.restoreStyle()
        c.restoreState()

    def page_rest(c, doc):
        c.saveState()
        c.setFont('Times-Roman', 7)
        c.setFillColor(C_GRAY)
        c.setLineWidth(0.5)
        c.setStrokeColor(C_BLACK)
        c.line(ML, PAGE_H-MT+0.15*inch, PAGE_W-MR, PAGE_H-MT+0.15*inch)
        hdr = "IEEE · GRAPES-SHAP: Clinical QA with World-Model Reasoning"
        c.drawCentredString(PAGE_W/2, PAGE_H-MT+0.25*inch, hdr)
        c.drawCentredString(PAGE_W/2, MB-0.3*inch, str(doc.page))
        c.restoreState()

    # Two-column template for body text
    pt_body = PageTemplate(
        id='body',
        frames=[col1, col2],
        onPage=page_rest
    )

    # Full-width template for title/abstract (first page)
    col1_p1 = Frame(ML, MB, COL_W, PAGE_H-MT-MB-3.5*inch, id='c1p1', showBoundary=0)
    col2_p1 = Frame(ML+COL_W+COL_GAP, MB, COL_W, PAGE_H-MT-MB-3.5*inch, id='c2p1', showBoundary=0)
    full_p1  = Frame(ML, PAGE_H-MT-MB-3.3*inch+MB, PAGE_W-ML-MR, 3.3*inch, id='fp1', showBoundary=0)

    pt_p1 = PageTemplate(
        id='first',
        frames=[full_p1, col1_p1, col2_p1],
        onPage=page_draw_title
    )

    doc.addPageTemplates([pt_p1, pt_body])
    return doc


# ── Content builder ──────────────────────────────────────────────────────────
def build_story(st):
    S  = st
    story = []

    # ── TITLE ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "GRAPES-SHAP: A Twelve-Stage Clinical Question-Answering Framework "
        "with Calibrated World-Model Reasoning and Shapley Source Attribution",
        S['title']
    ))
    story.append(Spacer(1, 0.08*inch))

    # ── AUTHORS ──────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "<b>Author A. Firstname</b><super>1</super>, "
        "<b>Author B. Secondname</b><super>2</super>, "
        "<b>Author C. Thirdname</b><super>1</super>, "
        "<b>Author D. Fourthname</b><super>1</super>, "
        "<b>Author E. Fifthname</b><super>1</super>",
        S['authors']
    ))
    story.append(Paragraph(
        "<super>1</super><i>Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh</i><br/>"
        "<super>2</super><i>Department of Biomedical Engineering, University XYZ, City, Country</i><br/>"
        "{email1, email2, email3, email4, email5}@bracu.ac.bd",
        S['affil']
    ))

    story.append(HRFlowable(width="90%", thickness=0.5, color=C_BLACK, spaceAfter=6))

    # ── ABSTRACT ────────────────────────────────────────────────────────────
    story.append(Paragraph("<i><b>Abstract</b></i>—"
        "Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm "
        "for grounding large language models (LLMs) in external evidence, yet conventional "
        "RAG pipelines remain fundamentally reactive: they retrieve, concatenate, and generate "
        "without modeling the consequences of clinical decisions, without calibrated uncertainty, "
        "and without faithful attribution of which evidence drove a given answer. "
        "We introduce <b>GRAPES-SHAP</b>, a twelve-stage framework that extends RAG along three "
        "orthogonal dimensions critical for clinical decision support: "
        "(i) an action-conditioned latent world model that simulates patient-state evolution "
        "and enables prospective Tree-of-Thought beam planning; "
        "(ii) a five-member deep-ensemble predictor that outputs diagnostically calibrated "
        "uncertainty with explicit decomposition into epistemic and aleatoric components; and "
        "(iii) a Shapley-value attribution module that quantifies the marginal contribution "
        "of each retrieved document to the generated answer. "
        "The neural core—comprising a causal graph-attention network, a gated evidence-fusion "
        "encoder, an action-conditioned causal residual, and a three-layer recurrent latent-dynamics "
        "model—totals only 10.1M parameters and trains in under 20 minutes on a single consumer GPU. "
        "On a corpus of 100,000 samples drawn from DDXPlus, MedMCQA, and MedQA-USMLE, "
        "the world model achieves a next-state MAE of 0.039 and an expected calibration error (ECE) "
        "of 0.029. Evaluated against a strong hybrid RAG baseline on ten complex clinical vignettes, "
        "GRAPES-SHAP improves clinical concept coverage from 0.70 to 0.97, answer-structure "
        "completeness from 0.50 to 0.84, and evidence grounding from 2.0 to 5.1 citations per "
        "answer, while additionally providing calibrated confidence, treatment-plan simulation, "
        "and per-document attribution—capabilities entirely absent from the baseline. "
        "Statistical significance is confirmed via Wilcoxon signed-rank test (p = 0.046). "
        "All code, configurations, and figure scripts are publicly released.",
        S['abst']))

    story.append(Paragraph(
        "<i><b>Index Terms</b></i>—Clinical decision support, retrieval-augmented generation, "
        "world models, uncertainty quantification, Shapley attribution, interpretable AI, "
        "graph attention networks, Tree-of-Thought planning, medical question answering.",
        S['kw']))

    story.append(HRFlowable(width="90%", thickness=0.5, color=C_BLACK, spaceAfter=4))

    # ── Switch to two-column layout ─────────────────────────────────────────
    story.append(Paragraph("", S['body']))  # spacer to trigger frame switch

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION I — INTRODUCTION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("I", "Introduction", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "Clinical Decision Support Systems (CDSS) represent one of the highest-impact applications "
        "of artificial intelligence in healthcare. Large language models (LLMs) encode remarkable "
        "clinical knowledge acquired from biomedical corpora [1]; however, linguistic fluency alone "
        "does not guarantee reliability. Two interconnected failures limit safe deployment: "
        "<i>hallucination</i>—the generation of plausible but factually incorrect statements [2]—and "
        "<i>opacity</i>—the inability to communicate calibrated confidence or trace a claim back to "
        "its supporting evidence.",
        S['body']))

    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) [3] partially addresses hallucination by grounding "
        "responses in retrieved documents. State-of-the-art RAG systems combine BM25 sparse retrieval [4] "
        "with dense bi-encoders [5], fuse ranked lists via Reciprocal Rank Fusion (RRF) [6], "
        "re-rank with cross-encoders [7], and diversify with Maximal Marginal Relevance (MMR) [8]. "
        "Despite these advances, the canonical retrieve–concatenate–generate paradigm remains "
        "fundamentally reactive: retrieved documents are treated as static context, all multi-step "
        "reasoning is delegated to the LLM, confidence estimates are uncalibrated, "
        "and no mechanism exists to identify which document contributed which claim.",
        S['body']))

    story.append(Paragraph(
        "These deficiencies are particularly critical in clinical settings, where trustworthiness "
        "requires (a) <i>prospective reasoning</i> about the consequences of alternative diagnostic "
        "actions; (b) <i>calibrated uncertainty</i> with explicit epistemic/aleatoric decomposition; "
        "and (c) <i>transparent source attribution</i> with axiomatic fairness guarantees.",
        S['body']))

    story.append(Paragraph(
        "This paper introduces <b>GRAPES-SHAP</b> "
        "(Graph-RAG with Prospective-world-model, Ensemble-uncertainty, and SHAPley attribution), "
        "a unified twelve-stage framework that addresses all three gaps within a single, "
        "computationally efficient pipeline. Our contributions are:",
        S['body']))

    contributions = [
        "A <b>context-labelling</b> scheme that enriches every answer with the retrieved evidence set, "
        "consulted knowledge-graph relations, the inferred diagnostic state, and a calibrated "
        "uncertainty estimate—making each response auditable by design.",
        "An <b>action-conditioned latent world model</b> that enables Tree-of-Thought beam planning "
        "over diagnostic actions, replacing undirected reactive answering with prospective reasoning.",
        "A <b>five-member probabilistic deep ensemble</b> with explicit decomposition of total "
        "predictive uncertainty into epistemic and aleatoric components, achieving ECE = 0.029.",
        "A <b>Shapley-value source-tracking module</b> that attributes each generated claim to its "
        "contributing documents via Monte-Carlo permutation sampling, providing per-document "
        "credit assignment with provable fairness axioms.",
        "A <b>compact, reproducible implementation</b>: 10.1M parameters, &lt;20-minute training "
        "on a single consumer GPU, statistically significant improvements over a strong hybrid "
        "RAG baseline (Wilcoxon p = 0.046), and full open-source release."
    ]
    for i, c in enumerate(contributions, 1):
        story.append(Paragraph(f"({i}) {c}", S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION II — RELATED WORK
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("II", "Related Work", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph("<i>A. Retrieval-Augmented Generation</i>", S['subsec']))
    story.append(Paragraph(
        "Lewis et al. [3] introduced RAG, grounding sequence-to-sequence generation in a neural "
        "retriever over Wikipedia. Dense Passage Retrieval (DPR) [5] replaced sparse matching with "
        "dual BERT encoders. FAISS [9] enabled billion-scale approximate nearest-neighbor search. "
        "RRF [6] fused sparse and dense lists. Cross-encoder re-ranking [7] and MMR diversification [8] "
        "further improved passage selection. HyDE [10] introduced hypothetical document embeddings for "
        "zero-shot query expansion. Self-RAG [11] added self-reflective critique tokens. "
        "None of these works equips RAG with prospective planning, calibrated uncertainty, or "
        "Shapley attribution.",
        S['body']))

    story.append(Paragraph("<i>B. World Models and Planning</i>", S['subsec']))
    story.append(Paragraph(
        "Ha and Schmidhuber [12] showed that a variational autoencoder plus recurrent model can "
        "learn a compact environment representation enabling policy learning in imagination. "
        "Dreamer [13] and DreamerV3 [14] scaled latent world models to diverse continuous-control "
        "domains. Chain-of-Thought prompting [15] and Tree-of-Thoughts (ToT) [16] introduced "
        "deliberate multi-step reasoning into LLMs. We bridge these threads by using a learned "
        "clinical latent-dynamics model as the ToT transition function, converting reactive "
        "answering into prospective diagnostic planning.",
        S['body']))

    story.append(Paragraph("<i>C. Uncertainty Quantification and Calibration</i>", S['subsec']))
    story.append(Paragraph(
        "Deep Ensembles [17] provide simple, scalable predictive uncertainty via independently "
        "initialized networks. Guo et al. [18] demonstrated that modern neural networks are "
        "miscalibrated and proposed temperature scaling. We integrate calibrated ensembles natively "
        "into the pipeline rather than as a post-hoc correction.",
        S['body']))

    story.append(Paragraph("<i>D. Explainability and Source Attribution</i>", S['subsec']))
    story.append(Paragraph(
        "Shapley values [19] provide axiomatic credit assignment in cooperative games. "
        "SHAP [20] unified this into a model-agnostic feature-attribution framework. "
        "Knowledge graphs have been shown to improve reasoning faithfulness in LLMs [21]. "
        "Graph Attention Networks (GAT) [22] enable dynamic, attention-weighted aggregation "
        "over graph-structured data. Our framework adapts Shapley attribution to retrieved "
        "documents—quantifying each document's marginal contribution to the final answer.",
        S['body']))

    story.append(Paragraph("<i>E. Clinical NLP Benchmarks</i>", S['subsec']))
    story.append(Paragraph(
        "Med-PaLM [1] demonstrated that fine-tuned LLMs can encode substantial clinical knowledge "
        "but still produce occasional hallucinations. DDXPlus [23] provides large-scale synthetic "
        "patient trajectories with differential diagnoses. MedMCQA [24] and MedQA-USMLE [25] supply "
        "examination-style multiple-choice questions across diverse medical domains. "
        "We use all three as complementary components of our data pipeline.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION III — PROBLEM FORMULATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("III", "Problem Formulation", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "Let <i>q</i> be a clinical query and <i>D</i> = {d<sub>1</sub>, …, d<sub>N</sub>} "
        "a corpus of medical documents. We seek a framework that produces an answer <i>a</i> "
        "satisfying four requirements simultaneously:",
        S['body']))

    reqs = [
        "<b>Evidence grounding:</b> <i>a</i> must be grounded in a retrieved subset "
        "D<sub>R</sub> ⊆ D with explicit citations.",
        "<b>Calibrated uncertainty:</b> The system must output a confidence estimate "
        "P(correct) whose predicted probability accurately reflects empirical correctness "
        "frequency (ECE ≤ 0.05).",
        "<b>Source attribution:</b> Each claim in <i>a</i> must be traced to a supporting "
        "document d<sub>i</sub> ∈ D<sub>R</sub>, with a quantified attribution score ϕ<sub>i</sub>.",
        "<b>Prospective reasoning:</b> The system must simulate forward trajectories over "
        "the patient's diagnostic state, enabling plan evaluation prior to answer generation."
    ]
    for i, r in enumerate(reqs, 1):
        story.append(Paragraph(f"<b>R{i}.</b> {r}", S['body']))

    story.append(Paragraph(
        "Existing RAG systems satisfy R1 partially and fail on R2–R4. "
        "GRAPES-SHAP is designed to satisfy all four requirements within a "
        "computationally efficient, end-to-end trainable architecture.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION IV — FRAMEWORK ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("IV", "Framework Architecture", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "GRAPES-SHAP is organized as a twelve-stage pipeline grouped into five "
        "functional blocks: (1) Retrieval, (2) Graph Reasoning, (3) World-Model Planning, "
        "(4) Uncertainty Estimation, and (5) Explanation. Fig. 1 shows the inference pipeline; "
        "Fig. 2 details the neural architecture with tensor shapes.",
        S['body']))

    story.append(Spacer(1, 4))
    story.append(PipelineDiagram(width=COL_W, height=1.55*inch))
    story.append(Paragraph(
        "Fig. 1. GRAPES-SHAP inference pipeline. A clinical query is expanded via HyDE, "
        "retrieved by hybrid BM25+dense search fused by RRF, re-ranked with a cross-encoder "
        "and MMR, encoded by a causal KG-GNN into a latent state, rolled forward by the latent "
        "world model under ToT beam search, scored for calibrated uncertainty by a deep ensemble, "
        "verified for hallucination, attributed to evidence via Shapley values, and finally "
        "verbalized by DeepSeek-V3 with explicit source citations.",
        S['cap']))

    story.append(Spacer(1, 4))
    story.append(ArchDiagram(width=COL_W, height=2.05*inch))
    story.append(Paragraph(
        "Fig. 2. Detailed neural architecture. Context labels (observations, actions, KG relations) "
        "enter on the left; the 10.1M-parameter core (Causal GAT, Evidence-Fusion Encoder, Causal "
        "Residual, GRU Dynamics) produces latent states; right-hand heads, the Beam Planner, "
        "Deep Ensemble, and Shapley module yield a calibrated, source-attributed answer.",
        S['cap']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<i>A. Hybrid Graph-Aware Retrieval (Stages 1–3)</i>", S['subsec']))
    story.append(Paragraph(
        "Given query <i>q</i>, HyDE generates n<sub>h</sub> = 3 hypothetical sub-answers "
        "whose embeddings are averaged to form an expanded query. We retrieve with a dense "
        "MiniLM bi-encoder over a FAISS index and with BM25, fusing the two ranked lists by:",
        S['body']))
    story.append(Paragraph(
        "RRF(d) = Σ<sub>r ∈ {dense, bm25}</sub> 1/(k + rank<sub>r</sub>(d)),  k = 60",
        S['eq']))
    story.append(Paragraph(
        "A cross-encoder re-scores the top candidates; MMR with λ = 0.6 then selects a "
        "diverse, relevant set D<sub>R</sub> that becomes both the first context label and "
        "the candidate pool for Shapley attribution.",
        S['body']))

    story.append(Paragraph("<i>B. Causal Knowledge-Graph Encoder (Stage 4)</i>", S['subsec']))
    story.append(Paragraph(
        "A medical knowledge graph with N = 20 nodes encodes a sparse stochastic causal "
        "prior over symptom–pathology relations. An edge-biased Graph Attention Network (GAT) [22] "
        "processes the graph; for H = 8 attention heads and learned edge weights e<sub>ij</sub>:",
        S['body']))
    story.append(Paragraph(
        "α<sup>(h)</sup><sub>ij</sub> ∝ (W<sub>q</sub>x<sub>i</sub>)<sup>T</sup>(W<sub>k</sub>x<sub>j</sub>)/√d<sub>h</sub> + W<sub>e</sub>e<sub>ij</sub>",
        S['eq']))
    story.append(Paragraph(
        "Masked mean pooling yields a graph embedding g ∈ R<sup>256</sup>. The patient "
        "observation trajectory obs ∈ R<sup>T×64</sup> is encoded by a three-layer "
        "Transformer encoder with gated cross-attention to g, producing latent states "
        "z<sub>t</sub> ∈ R<sup>256</sup> that serve as the predicted-state context label.",
        S['body']))

    story.append(Paragraph("<i>C. Latent World Model (Stage 5)</i>", S['subsec']))
    story.append(Paragraph(
        "The core contribution is an action-conditioned latent-dynamics model. At each "
        "diagnostic step, a causal residual computes the effect of action a<sub>t</sub>:",
        S['body']))
    story.append(Paragraph(
        "Δz<sub>t</sub> = s · σ(W<sub>g</sub>[z<sub>t</sub>, h]) ⊙ MLP[z<sub>t</sub>, g, e<sub>at</sub>]",
        S['eq']))
    story.append(Paragraph(
        "A three-layer GRU integrates the corrected state and action to predict the next "
        "latent state z<sub>t+1</sub> = h2z·GRU([z<sub>t</sub> + Δz<sub>t</sub>, e<sub>at</sub>]). "
        "Three heads decode the latent: a decoder reconstructs the observation, a σ-head emits "
        "per-dimension aleatoric uncertainty, and a reward head scores each transition. "
        "Rollout occurs entirely in latent space, enabling prospective planning without "
        "repeated LLM calls.",
        S['body']))

    story.append(Paragraph("<i>D. Tree-of-Thought Beam Planner (Stage 6)</i>", S['subsec']))
    story.append(Paragraph(
        "Plan selection is formulated as tree search over the diagnostic action space, "
        "with the world model as transition function. From encoded state z<sub>0</sub>, "
        "beam search of width W = 8 and depth H = 4 expands each node, scores transitions "
        "via the reward head, and ranks terminal plans by:",
        S['body']))
    story.append(Paragraph(
        "J(π) = v<sub>θ</sub>(z<sub>H</sub>) + β Σ<sub>t</sub> r<sub>φ</sub>(z<sub>t</sub>, z<sub>t+1</sub>) − γ σ̄(z<sub>H</sub>)",
        S['eq']))
    story.append(Paragraph(
        "with β = 0.5 and γ = 0.1. The uncertainty penalty γ σ̄ prevents the planner "
        "from selecting overconfident but unreliable trajectories.",
        S['body']))

    story.append(Paragraph("<i>E. Deep-Ensemble Uncertainty (Stage 7)</i>", S['subsec']))
    story.append(Paragraph(
        "Five probabilistic heads, each emitting a Gaussian (μ<sub>m</sub>, σ<sup>2</sup><sub>m</sub>) "
        "over the 5-dimensional differential-diagnosis distribution, are combined as:",
        S['body']))
    story.append(Paragraph(
        "μ = (1/M) Σ<sub>m</sub> μ<sub>m</sub>,   σ<sup>2</sup><sub>total</sub> = Var<sub>m</sub>(μ<sub>m</sub>) + (1/M) Σ<sub>m</sub> σ<sup>2</sup><sub>m</sub>",
        S['eq']))
    story.append(Paragraph(
        "The first term captures <i>epistemic</i> (reducible, model) uncertainty; "
        "the second captures <i>aleatoric</i> (irreducible, data) uncertainty. "
        "Together they form the calibrated-uncertainty context label.",
        S['body']))

    story.append(Paragraph("<i>F. Shapley Source-Tracking Module (Stage 9)</i>", S['subsec']))
    story.append(Paragraph(
        "Each retrieved document d<sub>i</sub> receives a Shapley attribution score [19] "
        "estimated via Monte-Carlo permutation sampling [20] with 32 permutations:",
        S['body']))
    story.append(Paragraph(
        "ϕ<sub>i</sub> = E<sub>π</sub>[v(S<sup>π</sup><sub>i</sub> ∪ {i}) − v(S<sup>π</sup><sub>i</sub>)]",
        S['eq']))
    story.append(Paragraph(
        "where v(·) is a cross-encoder relevance score for the document subset against "
        "<i>q</i>. Signed ϕ<sub>i</sub> values distinguish supporting from distracting "
        "evidence and are surfaced as per-claim citations in the final answer.",
        S['body']))

    story.append(Paragraph("<i>G. Training Objectives</i>", S['subsec']))
    story.append(Paragraph(
        "The world model, GNN, and encoder are trained jointly to minimize:",
        S['body']))
    story.append(Paragraph(
        "L<sub>wm</sub> = ‖ô − o<sup>+</sup>‖<sup>2</sup> + 0.01‖Δz‖<sup>2</sup> + 0.001σ̄ + 0.1‖r<sub>φ</sub> − r*‖<sup>2</sup>",
        S['eq']))
    story.append(Paragraph(
        "where the reward target r*<sub>t</sub> = (max<sub>k</sub> y<sub>k</sub>) · t/T shapes "
        "the reward head to prefer trajectories that drive the latent state toward confident "
        "differential diagnoses. The ensemble is trained by Gaussian negative log-likelihood. "
        "Both stages use AdamW with FP16 mixed-precision training.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION V — DATASETS AND PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("V", "Datasets and Preprocessing", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "We assemble 100,000 samples from three publicly available sources (Table I). "
        "DDXPlus [23] supplies patient trajectories for world-model training and evaluation; "
        "MedMCQA [24] documents form the retrievable evidence corpus; "
        "MedQA-USMLE [25] held-out queries drive the end-to-end evaluation.",
        S['body']))

    # Table I
    tdata = [
        [Paragraph("<b>Dataset</b>", S['thdr']),
         Paragraph("<b>Role</b>", S['thdr']),
         Paragraph("<b>Size</b>", S['thdr'])],
        [Paragraph("DDXPlus", S['tcell']),
         Paragraph("World-model dynamics", S['tcell']),
         Paragraph("80k / 10k / 10k", S['tcell'])],
        [Paragraph("MedMCQA", S['tcell']),
         Paragraph("Evidence corpus", S['tcell']),
         Paragraph("50,000 docs", S['tcell'])],
        [Paragraph("MedQA-USMLE", S['tcell']),
         Paragraph("Held-out evaluation", S['tcell']),
         Paragraph("1,000 queries", S['tcell'])],
    ]
    t = Table(tdata, colWidths=[1.1*inch, 1.5*inch, 0.85*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), C_BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), C_WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_LLGRAY, C_WHITE]),
        ('GRID',        (0,0), (-1,-1), 0.3, C_LGRAY),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
    ]))
    story.append(t)
    story.append(Paragraph("TABLE I. Dataset composition (100,000 samples total).", S['cap']))
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "DDXPlus records are parsed and validated; approximately 99.97% of all 80,000 training "
        "records satisfy completeness checks. Evidence codes of the form E<sub>54</sub>@V<sub>161</sub> "
        "are normalized to canonical base codes (e.g., E<sub>54</sub>), reducing the unique "
        "vocabulary from 504 to 223 codes and removing value-suffix sparsity. Age is scaled to "
        "[0, 1]; sex is represented as a binary indicator; evidence sets are binary feature vectors. "
        "Each patient trajectory is modeled as an 8-step diagnostic sequence in a 64-dimensional "
        "observation space. The output is a 5-dimensional vector of the top-k differential "
        "diagnosis probabilities. The dataset exhibits a long-tailed pathology distribution "
        "across 49 disease classes (mean patient age 39.6 years, SD 22.7; balanced sex ratio: "
        "0.486 male, 0.514 female; median clinical evidences per patient: 19).",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION VI — EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("VI", "Experimental Setup", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph("<i>A. Hyperparameters</i>", S['subsec']))

    # Table II
    hdata = [
        [Paragraph("<b>Group</b>", S['thdr']),
         Paragraph("<b>Parameter</b>", S['thdr']),
         Paragraph("<b>Value</b>", S['thdr'])],
        [Paragraph("Architecture", S['tcell']),
         Paragraph("Latent / hidden dim", S['tcell']),
         Paragraph("256 / 512", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("Transformer layers / heads", S['tcell']),
         Paragraph("3 / 8", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("GAT layers (edge-biased)", S['tcell']),
         Paragraph("3", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("Ensemble members M", S['tcell']),
         Paragraph("5", S['tcell'])],
        [Paragraph("Planning", S['tcell']),
         Paragraph("Beam width W / depth H", S['tcell']),
         Paragraph("8 / 4", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("Reward weight β", S['tcell']),
         Paragraph("0.5", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("Uncertainty penalty γ", S['tcell']),
         Paragraph("0.1", S['tcell'])],
        [Paragraph("Retrieval", S['tcell']),
         Paragraph("Top-k / embed dim", S['tcell']),
         Paragraph("6 / 384", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("MMR λ / RRF k", S['tcell']),
         Paragraph("0.6 / 60", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("HyDE sub-queries", S['tcell']),
         Paragraph("3", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("SHAP permutations", S['tcell']),
         Paragraph("32", S['tcell'])],
        [Paragraph("Optimization", S['tcell']),
         Paragraph("WM / ensemble epochs", S['tcell']),
         Paragraph("15 / 10", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("WM / ensemble LR", S['tcell']),
         Paragraph("2×10<sup>-4</sup> / 10<sup>-3</sup>", S['tcell'])],
        [Paragraph("", S['tcell']),
         Paragraph("Batch size / precision", S['tcell']),
         Paragraph("64 / FP16", S['tcell'])],
    ]
    t2 = Table(hdata, colWidths=[0.8*inch, 1.4*inch, 0.85*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), C_BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), C_WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_LLGRAY, C_WHITE]),
        ('GRID',        (0,0), (-1,-1), 0.3, C_LGRAY),
        ('FONTSIZE',    (0,0), (-1,-1), 7.5),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 2),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2),
        ('SPAN',        (0,1),(0,4)),
        ('SPAN',        (0,5),(0,7)),
        ('SPAN',        (0,8),(0,11)),
        ('SPAN',        (0,12),(0,14)),
    ]))
    story.append(t2)
    story.append(Paragraph("TABLE II. Key hyperparameters.", S['cap']))
    story.append(Spacer(1, 3))

    story.append(Paragraph("<i>B. Baselines</i>", S['subsec']))
    story.append(Paragraph(
        "We evaluate against four RAG families: (i) Vanilla RAG with direct BM25 retrieval, "
        "(ii) HyDE query expansion, (iii) cross-encoder re-ranking, and (iv) our primary quantitative "
        "baseline—hybrid dense+BM25 RAG (MiniLM+FAISS fused by RRF) with direct answers from "
        "DeepSeek-V3 [26], the same LLM used by GRAPES-SHAP, but without the world model, "
        "ensemble, or Shapley attribution. This controls for LLM contribution and isolates "
        "the benefit of our structured pipeline.",
        S['body']))

    story.append(Paragraph("<i>C. Evaluation Protocol</i>", S['subsec']))
    story.append(Paragraph(
        "The world model and ensemble are evaluated on the DDXPlus validation split "
        "(10,000 samples) using next-state MAE, RMSE, top-rank accuracy, Macro-F1 "
        "(49 classes), ECE, and 1σ coverage. End-to-end clinical quality is assessed "
        "on ten complex clinical vignettes drawn from the MedQA-USMLE held-out set, "
        "evaluating: (1) clinical concept coverage (fraction of gold-standard concepts "
        "present in the answer), (2) answer-structure completeness, (3) evidence citations "
        "per answer (normalized by top-k budget), and (4) stated confidence. "
        "Statistical significance is assessed by Wilcoxon signed-rank test on paired "
        "per-vignette concept-coverage scores.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION VII — RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("VII", "Results and Analysis", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph("<i>A. World-Model and Ensemble Evaluation</i>", S['subsec']))
    story.append(Paragraph(
        "Table III summarizes world-model performance on the DDXPlus validation split. "
        "The model predicts the next latent-decoded state with MAE = 0.039 and RMSE = 0.074, "
        "correctly ranks the true pathology first 75.5% of the time, and is well-calibrated "
        "(ECE = 0.029, 1σ coverage = 0.801). The modest Macro-F1 of 0.172 reflects the "
        "difficulty of fine-grained rank classification over 49 highly imbalanced pathology "
        "classes—this is expected and does not contradict the strong calibration results, "
        "which are the quantities of interest for trustworthy clinical decision support.",
        S['body']))

    # Table III
    rdata = [
        [Paragraph("<b>Metric</b>", S['thdr']),
         Paragraph("<b>Value</b>", S['thdr'])],
        [Paragraph("Next-state MAE", S['tcell']),        Paragraph("0.039", S['tcell'])],
        [Paragraph("Next-state RMSE", S['tcell']),       Paragraph("0.074", S['tcell'])],
        [Paragraph("Top-rank accuracy", S['tcell']),     Paragraph("0.755", S['tcell'])],
        [Paragraph("Macro-F1 (49 classes)", S['tcell']), Paragraph("0.172", S['tcell'])],
        [Paragraph("ECE", S['tcell']),                   Paragraph("<b>0.029</b>", S['tcell'])],
        [Paragraph("1σ coverage", S['tcell']),           Paragraph("0.801", S['tcell'])],
        [Paragraph("Mean |SHAP| attribution", S['tcell']),Paragraph("0.734", S['tcell'])],
        [Paragraph("Parameters", S['tcell']),            Paragraph("10,130,060", S['tcell'])],
        [Paragraph("Training time (RTX 4080S)", S['tcell']), Paragraph("< 20 min", S['tcell'])],
    ]
    t3 = Table(rdata, colWidths=[1.9*inch, 1.0*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), C_BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), C_WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_LLGRAY, C_WHITE]),
        ('GRID',        (0,0), (-1,-1), 0.3, C_LGRAY),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Paragraph("TABLE III. World-model and ensemble evaluation on DDXPlus validation split.", S['cap']))
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "Generalization is confirmed by running trained checkpoints on the held-out split: "
        "train/test MAE differ by less than 0.002 (0.037 vs. 0.039) and RMSE by 0.004 "
        "(0.070 vs. 0.074), demonstrating that the model generalizes rather than memorizes. "
        "The t-SNE projection of learned latent states organizes by true diagnosis rank, "
        "indicating that the encoder captures clinically meaningful geometry.",
        S['body']))

    story.append(Paragraph("<i>B. Calibration</i>", S['subsec']))
    story.append(Paragraph(
        "Fig. 3 shows the reliability diagram. The predicted uncertainty σ closely tracks "
        "empirical error across the full range, confirming low ECE. The epistemic/aleatoric "
        "decomposition provides actionable signal: the planner exploits epistemic uncertainty "
        "via the γ-penalty in Eq. (4), while aleatoric uncertainty is surfaced as a context "
        "label to the clinician.",
        S['body']))

    story.append(Spacer(1, 4))
    story.append(ReliabilityDiagram(width=COL_W*0.9, height=1.35*inch))
    story.append(Paragraph(
        "Fig. 3. Reliability diagram. Predicted uncertainty σ closely matches empirical error, "
        "confirming good calibration (ECE = 0.029). Dashed line = perfect calibration.",
        S['cap']))
    story.append(Spacer(1, 3))

    story.append(Paragraph("<i>C. End-to-End Clinical Evaluation</i>", S['subsec']))
    story.append(Paragraph(
        "Table IV and Fig. 4 summarize the head-to-head comparison on ten clinical vignettes. "
        "GRAPES-SHAP improves clinical concept coverage from 0.70 to 0.97 (+38.6%), "
        "answer-structure completeness from 0.50 to 0.84 (+68%), and evidence grounding "
        "from 2.0 to 5.1 citations per answer (2.5× increase). Confidence is lower (0.70 vs. 0.91) "
        "but appropriately calibrated: the baseline's near-constant 0.91 is indicative of "
        "overconfidence, whereas GRAPES-SHAP expresses reduced confidence where true uncertainty "
        "is high. Additionally, GRAPES-SHAP is the only system that provides Shapley attribution "
        "(mean |ϕ| = 1.28), calibrated uncertainty decomposition, and treatment-plan simulation.",
        S['body']))

    # Table IV
    cdata = [
        [Paragraph("<b>Metric</b>", S['thdr']),
         Paragraph("<b>Baseline RAG</b>", S['thdr']),
         Paragraph("<b>GRAPES-SHAP</b>", S['thdr'])],
        [Paragraph("Clinical concept coverage", S['tcell']),
         Paragraph("0.70", S['tcell']),
         Paragraph("<b>0.97</b>", S['tcell'])],
        [Paragraph("Answer-structure completeness", S['tcell']),
         Paragraph("0.50", S['tcell']),
         Paragraph("<b>0.84</b>", S['tcell'])],
        [Paragraph("Evidence citations (avg.)", S['tcell']),
         Paragraph("2.0", S['tcell']),
         Paragraph("<b>5.1</b>", S['tcell'])],
        [Paragraph("Stated confidence", S['tcell']),
         Paragraph("0.91", S['tcell']),
         Paragraph("0.70 (calibrated)", S['tcell'])],
        [Paragraph("SHAP attribution (mean |ϕ|)", S['tcell']),
         Paragraph("—", S['tcell']),
         Paragraph("<b>1.28</b>", S['tcell'])],
        [Paragraph("Calibrated uncertainty", S['tcell']),
         Paragraph("No", S['tcell']),
         Paragraph("<b>Yes</b>", S['tcell'])],
        [Paragraph("Plan simulation", S['tcell']),
         Paragraph("No", S['tcell']),
         Paragraph("<b>Yes</b>", S['tcell'])],
    ]
    t4 = Table(cdata, colWidths=[1.6*inch, 0.88*inch, 0.97*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), C_BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), C_WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_LLGRAY, C_WHITE]),
        ('GRID',        (0,0), (-1,-1), 0.3, C_LGRAY),
        ('FONTSIZE',    (0,0), (-1,-1), 7.5),
        ('ALIGN',       (1,0), (2,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2.5),
    ]))
    story.append(t4)
    story.append(Paragraph(
        "TABLE IV. GRAPES-SHAP vs. hybrid-RAG baseline on ten complex clinical vignettes.",
        S['cap']))
    story.append(Spacer(1, 3))

    story.append(Spacer(1, 4))
    story.append(ResultsChart(width=COL_W, height=1.55*inch))
    story.append(Paragraph(
        "Fig. 4. Aggregate comparison on ten vignettes (citation counts normalized by top-k budget). "
        "GRAPES-SHAP improves all three measurable dimensions over the hybrid RAG baseline.",
        S['cap']))
    story.append(Spacer(1, 3))

    story.append(Paragraph("<i>D. Statistical Analysis</i>", S['subsec']))
    story.append(Paragraph(
        "A Wilcoxon signed-rank test on paired per-vignette concept-coverage scores confirms "
        "statistical significance (p = 0.046). The mean gain is +0.27 (95% CI ± 0.21). "
        "GRAPES-SHAP matches or exceeds the baseline on all ten individual vignettes, "
        "demonstrating consistent rather than cherry-picked improvement.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION VIII — DISCUSSION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("VIII", "Discussion", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "The results demonstrate that inserting a calibrated, interpretable world model between "
        "retrieval and generation yields statistically significant gains in answer quality "
        "at little computational overhead. Several scope boundaries deserve explicit discussion.",
        S['body']))

    story.append(Paragraph("<i>Semi-synthetic trajectories.</i>", S['subsubsec']))
    story.append(Paragraph(
        "DDXPlus is static diagnostic data; our preprocessor creates trajectories through "
        "sequential evidence disclosure. The transition target is effectively deterministic, "
        "so the low MAE should be interpreted as accurate modeling of this constructed "
        "dynamics—not of real physiological time series. Using logged clinical data "
        "(e.g., ICU records) would enable more realistic world-model training.",
        S['body']))

    story.append(Paragraph("<i>Action semantics.</i>", S['subsubsec']))
    story.append(Paragraph(
        "During training, an action corresponds to evidence acquisition rather than physical "
        "intervention. The planner should be interpreted as an evidence-collection/diagnosis-refinement "
        "policy; mapping it to treatment actions would require a dataset in which interventions "
        "and outcomes are jointly logged.",
        S['body']))

    story.append(Paragraph("<i>Knowledge graph.</i>", S['subsubsec']))
    story.append(Paragraph(
        "The 20-node graph functions as a stochastic causal prior, not a curated medical ontology. "
        "Replacing it with relations derived from SNOMED-CT or UMLS is a natural extension "
        "that would strengthen the graph reasoning component.",
        S['body']))

    story.append(Paragraph("<i>Macro-F1.</i>", S['subsubsec']))
    story.append(Paragraph(
        "The modest Macro-F1 of 0.172 is a consequence of fine-grained, imbalanced rank "
        "classification over 49 pathology classes. This does not contradict the strong "
        "calibration and coverage results, which are the quantities of interest for "
        "trustworthy decision support and are obtained by argmax over a soft distribution.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION IX — CONCLUSION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("IX", "Conclusion", COL_W))
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "We presented GRAPES-SHAP, a twelve-stage clinical question-answering framework that "
        "augments RAG with an action-conditioned latent world model, a calibrated deep ensemble, "
        "and Shapley-value source attribution—all within a 10.1M-parameter core trainable in "
        "under 20 minutes on a single consumer GPU. The world model achieves MAE = 0.039 and "
        "ECE = 0.029, generalizes with a train/test gap below 0.002 MAE, and enables "
        "Tree-of-Thought beam planning that is both reward-guided and uncertainty-aware. "
        "On ten complex clinical vignettes, GRAPES-SHAP statistically significantly outperforms "
        "a strong hybrid RAG baseline (Wilcoxon p = 0.046), improving concept coverage by 38.6%, "
        "structure completeness by 68%, and evidence grounding by 2.5×, while additionally "
        "providing capabilities—calibrated uncertainty, per-document attribution, and plan "
        "simulation—entirely absent from the baseline.",
        S['body']))

    story.append(Paragraph(
        "Future work includes: replacing semi-synthetic dynamics with logged intervention–outcome "
        "data to enable genuine treatment planning; integrating a curated medical ontology "
        "(SNOMED-CT) to strengthen graph reasoning; addressing class imbalance via focal loss or "
        "re-weighting to improve Macro-F1 without sacrificing calibration; and conducting a "
        "clinician-in-the-loop study to quantify the benefit of context labels and source "
        "tracking in reducing automation bias in practice.",
        S['body']))

    story.append(Spacer(1, 4))

    # ═══════════════════════════════════════════════════════════════════════
    # REFERENCES
    # ═══════════════════════════════════════════════════════════════════════
    story.append(SectionHeading("", "References", COL_W))
    story.append(Spacer(1, 2))

    refs = [
        "[1] K. Singhal et al., "Large language models encode clinical knowledge," "
        "<i>Nature</i>, vol. 620, no. 7972, pp. 172\u2013180, 2023.",

        "[2] Z. Ji et al., "Survey of hallucination in natural language generation," "
        "<i>ACM Comput. Surv.</i>, vol. 55, no. 12, pp. 1\u201338, 2023.",

        "[3] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 33, pp. 9459\u20139474, 2020.",

        "[4] S. E. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," "
        "<i>Found. Trends Inf. Retr.</i>, vol. 3, no. 4, pp. 333\u2013389, 2009.",

        "[5] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," "
        "<i>Proc. EMNLP</i>, pp. 6769\u20136781, 2020.",

        "[6] G. V. Cormack, C. L. Clarke, and S. Buettcher, \u201cReciprocal rank fusion outperforms "
        "Condorcet and individual rank learning methods,\u201d <i>Proc. 32nd ACM SIGIR</i>, pp. 758\u2013759, 2009.",

        "[7] R. Nogueira and K. Cho, "Passage re-ranking with BERT," "
        "<i>arXiv preprint arXiv:1901.04085</i>, 2019.",

        "[8] J. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking for reordering "
        "documents and producing summaries," <i>Proc. 21st ACM SIGIR</i>, pp. 335\u2013336, 1998.",

        "[9] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," "
        "<i>IEEE Trans. Big Data</i>, vol. 7, no. 3, pp. 535\u2013547, 2019.",

        "[10] L. Gao, X. Ma, J. Lin, and J. Callan, "Precise zero-shot dense retrieval without "
        "relevance labels," <i>arXiv preprint arXiv:2212.10496</i>, 2023.",

        "[11] A. Asai et al., "Self-RAG: Learning to retrieve, generate, and critique through "
        "self-reflection," <i>Proc. ICLR</i>, 2024.",

        "[12] D. Ha and J. Schmidhuber, "Recurrent world models facilitate policy evolution," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 31, 2018.",

        "[13] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, "Dream to control: Learning behaviors "
        "by latent imagination," <i>Proc. ICLR</i>, 2020.",

        "[14] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, "Mastering diverse domains through "
        "world models," <i>arXiv preprint arXiv:2301.04104</i>, 2023.",

        "[15] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 35, 2022.",

        "[16] S. Yao et al., "Tree of thoughts: Deliberate problem solving with large language models," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 36, 2023.",

        "[17] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive "
        "uncertainty estimation using deep ensembles," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 30, 2017.",

        "[18] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural "
        "networks," <i>Proc. ICML</i>, 2017.",

        "[19] L. S. Shapley, "A value for n-person games," in <i>Contributions to the Theory of "
        "Games</i>, vol. 2, Princeton Univ. Press, 1953, pp. 307\u2013317.",

        "[20] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 30, 2017.",

        "[21] L. Luo, Y.-F. Li, G. Haffari, and S. Pan, "Reasoning on graphs: Faithful and "
        "interpretable large language model reasoning," <i>Proc. ICLR</i>, 2024.",

        "[22] P. Veličković et al., "Graph attention networks," <i>Proc. ICLR</i>, 2018.",

        "[23] A. Fansi Tchango et al., "DDXPlus: A new dataset for automatic medical diagnosis," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, vol. 35, pp. 31 306\u201331 318, 2022.",

        "[24] A. Pal, L. K. Umapathi, and M. Sankarasubbu, "MedMCQA: A large-scale multi-subject "
        "multi-choice dataset for medical domain question answering," "
        "<i>Proc. CHIL</i>, pp. 248\u2013260, 2022.",

        "[25] D. Jin et al., "What disease does this patient have? A large-scale open-domain QA "
        "dataset from medical exams," <i>Appl. Sci.</i>, vol. 11, no. 14, p. 6421, 2021.",

        "[26] DeepSeek-AI, "DeepSeek-V3 technical report," "
        "<i>arXiv preprint arXiv:2412.19437</i>, 2024.",

        "[27] A. Vaswani et al., "Attention is all you need," "
        "<i>Adv. Neural Inf. Process. Syst.</i>, 2017.",

        "[28] K. Cho et al., "Learning phrase representations using RNN encoder-decoder," "
        "<i>Proc. EMNLP</i>, 2014.",

        "[29] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," "
        "<i>Proc. ICLR</i>, 2015.",

        "[30] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese "
        "BERT-networks," <i>Proc. EMNLP-IJCNLP</i>, pp. 3982\u20133992, 2019.",
    ]

    for r in refs:
        story.append(Paragraph(r, S['ref']))

    return story


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = "/mnt/user-data/outputs/GRAPES_SHAP_IEEE_Paper.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    st    = make_styles()
    doc   = make_doc(out)
    story = build_story(st)
    doc.build(story)
    print(f"✓  PDF written → {out}")