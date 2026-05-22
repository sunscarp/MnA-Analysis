import io
import base64
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, HRFlowable, BalancedColumns
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO

# ─────────────────────────────────────────────
#  FONT REGISTRATION  (bundled Unicode font supports ₹)
# ─────────────────────────────────────────────
_FONTS_REGISTERED = False

FONT_REGULAR      = 'Helvetica'
FONT_BOLD         = 'Helvetica-Bold'
FONT_OBLIQUE      = 'Helvetica-Oblique'
FONT_BOLD_OBLIQUE = 'Helvetica-BoldOblique'


def _ensure_fonts():
    global _FONTS_REGISTERED, FONT_REGULAR, FONT_BOLD, FONT_OBLIQUE, FONT_BOLD_OBLIQUE
    if _FONTS_REGISTERED:
        return
    font_path = Path(__file__).resolve().parent.parent / 'fonts' / 'NotoSans-Regular.ttf'

    try:
        if not font_path.exists():
            raise FileNotFoundError(f'Missing font file: {font_path}')

        pdfmetrics.registerFont(TTFont('NotoSans', str(font_path)))
        addMapping('NotoSans', 0, 0, 'NotoSans')
        addMapping('NotoSans', 0, 1, 'NotoSans')
        addMapping('NotoSans', 1, 0, 'NotoSans')
        addMapping('NotoSans', 1, 1, 'NotoSans')

        FONT_REGULAR = 'NotoSans'
        FONT_BOLD = 'NotoSans'
        FONT_OBLIQUE = 'NotoSans'
        FONT_BOLD_OBLIQUE = 'NotoSans'
        _FONTS_REGISTERED = True
        print('Successfully registered bundled NotoSans font with ReportLab.')
    except Exception as e:
        FONT_REGULAR = 'Helvetica'
        FONT_BOLD = 'Helvetica-Bold'
        FONT_OBLIQUE = 'Helvetica-Oblique'
        FONT_BOLD_OBLIQUE = 'Helvetica-BoldOblique'
        _FONTS_REGISTERED = True
        print(f'Warning: Could not register bundled NotoSans font. Rupee symbol may not render. Error: {e}')

# Call once at import time
_ensure_fonts()


# ─────────────────────────────────────────────
#  BRAND PALETTE  (Goldman / JPMorgan inspired)
# ─────────────────────────────────────────────
NAVY        = colors.HexColor('#0A1628')
BLUE        = colors.HexColor('#1A3A6B')
ACCENT      = colors.HexColor('#C8A84B')
LIGHT_BLUE  = colors.HexColor('#2E5FA3')
MID_GREY    = colors.HexColor('#6B7280')
LIGHT_GREY  = colors.HexColor('#F4F5F7')
RULE_GREY   = colors.HexColor('#D1D5DB')
WHITE       = colors.white
BLACK       = colors.HexColor('#111827')

TABLE_HEADER_BG   = NAVY
TABLE_ALT_BG      = colors.HexColor('#F8F9FB')
TABLE_BORDER      = colors.HexColor('#DDE1E7')
POSITIVE_GREEN    = colors.HexColor('#15803D')
NEGATIVE_RED      = colors.HexColor('#B91C1C')
CAUTION_AMBER     = colors.HexColor('#B45309')


# ─────────────────────────────────────────────
#  PAGE TEMPLATE
#  FIX: Two separate PageTemplates — one for the
#  cover (canvas-drawn, no chrome), one for all
#  body pages (with header/footer chrome).
#  The cover template is ONLY applied to page 1.
# ─────────────────────────────────────────────

class IBDocTemplate(BaseDocTemplate):
    """
    Custom doc template with:
      - Page 1: full-bleed cover drawn on canvas, no frame content
      - Pages 2+: branded header/footer chrome + body frame
    """

    def __init__(self, filename, acquirer_name, target_name, cover_callback, **kwargs):
        self.acquirer_name   = acquirer_name
        self.target_name     = target_name
        self._cover_callback = cover_callback  # callable(canv, doc)
        BaseDocTemplate.__init__(self, filename, **kwargs)

        margin = 0.65 * inch

        # ── Cover frame: same margins as body but cover_callback
        #    draws over entire page; frame just needs to exist.
        cover_frame = Frame(
            margin, margin,
            self.pagesize[0] - 2 * margin,
            self.pagesize[1] - 2 * margin,
            id='cover'
        )
        cover_template = PageTemplate(
            id='Cover',
            frames=[cover_frame],
            onPage=self._draw_cover
        )

        # ── Body frame: top margin accounts for header chrome
        body_frame = Frame(
            margin,
            margin + 0.45 * inch,
            self.pagesize[0] - 2 * margin,
            self.pagesize[1] - 2 * margin - 0.65 * inch,
            id='body'
        )
        body_template = PageTemplate(
            id='Body',
            frames=[body_frame],
            onPage=self._draw_chrome
        )

        self.addPageTemplates([cover_template, body_template])

    # ── Cover: delegate to the provided callback ──
    def _draw_cover(self, canv, doc):
        self._cover_callback(canv, doc)

    # ── Body chrome: header rule + footer ─────────
    def _draw_chrome(self, canv, doc):
        w, h = doc.pagesize
        margin = 0.65 * inch

        # Top navy rule
        canv.setFillColor(NAVY)
        canv.rect(margin, h - margin, w - 2 * margin, 2, fill=1, stroke=0)

        # Header label
        canv.setFont(FONT_REGULAR, 7)
        canv.setFillColor(MID_GREY)
        canv.drawRightString(w - margin, h - margin + 5, 'M&A INTELLIGENCE PLATFORM')

        # Gold accent under header
        canv.setFillColor(ACCENT)
        canv.rect(margin, h - margin - 3, w - 2 * margin, 1.5, fill=1, stroke=0)

        # Footer rule
        canv.setFillColor(RULE_GREY)
        canv.rect(margin, margin - 4, w - 2 * margin, 0.5, fill=1, stroke=0)

        # Footer text
        canv.setFont(FONT_REGULAR, 6.5)
        canv.setFillColor(MID_GREY)
        disclaimer = ('For internal use only. Not for distribution. '
                      'This material does not constitute investment advice.')
        canv.drawString(margin, margin - 14, disclaimer)

        canv.setFont(FONT_BOLD, 7)
        canv.setFillColor(NAVY)
        canv.drawRightString(w - margin, margin - 14, f'Page {doc.page}')


# ─────────────────────────────────────────────
#  STYLE FACTORY  (all using FreeSans)
# ─────────────────────────────────────────────
def build_styles():
    _ensure_fonts()
    styles = {}

    styles['cover_firm'] = ParagraphStyle(
        'cover_firm', fontSize=10, fontName=FONT_REGULAR,
        textColor=ACCENT, alignment=TA_LEFT, spaceAfter=4
    )
    styles['cover_title'] = ParagraphStyle(
        'cover_title', fontSize=28, fontName=FONT_BOLD,
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=6, leading=34
    )
    styles['cover_sub'] = ParagraphStyle(
        'cover_sub', fontSize=13, fontName=FONT_REGULAR,
        textColor=colors.HexColor('#BFC9D9'), alignment=TA_LEFT, spaceAfter=4
    )
    styles['cover_meta'] = ParagraphStyle(
        'cover_meta', fontSize=8.5, fontName=FONT_REGULAR,
        textColor=colors.HexColor('#8DA0BC'), alignment=TA_LEFT, spaceAfter=2
    )
    styles['section_label'] = ParagraphStyle(
        'section_label', fontSize=7.5, fontName=FONT_BOLD,
        textColor=ACCENT, alignment=TA_LEFT, spaceAfter=2,
        spaceBefore=18, letterSpacing=1.5
    )
    styles['h1'] = ParagraphStyle(
        'h1', fontSize=16, fontName=FONT_BOLD,
        textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=6, spaceBefore=4, leading=20
    )
    styles['h2'] = ParagraphStyle(
        'h2', fontSize=11, fontName=FONT_BOLD,
        textColor=BLUE, alignment=TA_LEFT,
        spaceAfter=4, spaceBefore=10, leading=14
    )
    styles['h3'] = ParagraphStyle(
        'h3', fontSize=9.5, fontName=FONT_BOLD,
        textColor=LIGHT_BLUE, alignment=TA_LEFT,
        spaceAfter=3, spaceBefore=6
    )
    styles['body'] = ParagraphStyle(
        'body', fontSize=9, fontName=FONT_REGULAR,
        textColor=BLACK, alignment=TA_JUSTIFY,
        spaceAfter=5, leading=14
    )
    styles['body_small'] = ParagraphStyle(
        'body_small', fontSize=8, fontName=FONT_REGULAR,
        textColor=MID_GREY, alignment=TA_LEFT,
        spaceAfter=3, leading=12
    )
    styles['kpi_value'] = ParagraphStyle(
        'kpi_value', fontSize=18, fontName=FONT_BOLD,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=0
    )
    styles['kpi_label'] = ParagraphStyle(
        'kpi_label', fontSize=7, fontName=FONT_REGULAR,
        textColor=MID_GREY, alignment=TA_CENTER, spaceAfter=0
    )
    styles['disclaimer'] = ParagraphStyle(
        'disclaimer', fontSize=7.5, fontName=FONT_OBLIQUE,
        textColor=MID_GREY, alignment=TA_LEFT, leading=11
    )
    return styles


# ─────────────────────────────────────────────
#  REUSABLE FLOWABLE HELPERS
# ─────────────────────────────────────────────

def section_rule(styles, label: str):
    """Gold-ruled section divider with uppercase label."""
    return [
        Spacer(1, 0.15 * inch),
        Paragraph(label.upper(), styles['section_label']),
        HRFlowable(width='100%', thickness=0.75, color=ACCENT, spaceAfter=6),
    ]


def thin_rule(color=RULE_GREY):
    return HRFlowable(width='100%', thickness=0.5, color=color, spaceAfter=4, spaceBefore=4)


def kpi_card_table(kpis: list, col_count: int = 4):
    """
    kpis: list of (label, value, optional_delta_str)
    Renders a row of KPI boxes.
    """
    styles = build_styles()
    while len(kpis) % col_count != 0:
        kpis.append(('', '', ''))

    rows = []
    for i in range(0, len(kpis), col_count):
        chunk = kpis[i:i + col_count]
        val_row, lbl_row = [], []
        for lbl, val, *delta in chunk:
            v_text = val
            if delta and delta[0]:
                color_tag = 'green' if '+' in str(delta[0]) else 'red'
                v_text = f'{val}  <font color="{color_tag}" size="8">{delta[0]}</font>'
            val_row.append(Paragraph(v_text, styles['kpi_value']) if val else Paragraph('', styles['kpi_value']))
            lbl_row.append(Paragraph(lbl.upper(), styles['kpi_label']))
        rows.append(val_row)
        rows.append(lbl_row)

    col_w = 6.5 * inch / col_count
    t = Table(rows, colWidths=[col_w] * col_count)
    t.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ('BOX',           (0, 0), (-1, -1), 0.75, RULE_GREY),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, RULE_GREY),
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GREY),
    ]))
    return t


def ib_table(data: list, col_widths=None, header_bg=TABLE_HEADER_BG,
             font_size: int = 8, zebra: bool = True, first_col_bold: bool = True):
    """
    Render a clean IB-style table.
    data[0] = header row; data[1:] = body rows.
    """
    if not data:
        return Spacer(1, 0.1 * inch)

    page_width = 6.5 * inch
    if col_widths is None:
        col_widths = [page_width / len(data[0])] * len(data[0])

    def _wrap(cell, style):
        if isinstance(cell, str):
            return Paragraph(cell, style)
        return cell

    normal_style = ParagraphStyle('tc', fontSize=font_size, fontName=FONT_REGULAR,
                                  textColor=BLACK, leading=font_size + 3)
    bold_style   = ParagraphStyle('tb', fontSize=font_size, fontName=FONT_BOLD,
                                  textColor=BLACK, leading=font_size + 3)
    header_style = ParagraphStyle('th', fontSize=font_size, fontName=FONT_BOLD,
                                  textColor=WHITE, leading=font_size + 3)

    formatted = []
    for r_idx, row in enumerate(data):
        frow = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0:
                frow.append(_wrap(str(cell), header_style))
            elif first_col_bold and c_idx == 0:
                frow.append(_wrap(str(cell), bold_style))
            else:
                frow.append(_wrap(str(cell), normal_style))
        formatted.append(frow)

    t = Table(formatted, colWidths=col_widths, repeatRows=1)

    cmds = [
        ('BACKGROUND',    (0, 0),  (-1, 0),  header_bg),
        ('TEXTCOLOR',     (0, 0),  (-1, 0),  WHITE),
        ('FONTNAME',      (0, 0),  (-1, 0),  FONT_BOLD),
        ('FONTSIZE',      (0, 0),  (-1, -1), font_size),
        ('VALIGN',        (0, 0),  (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0),  (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0),  (-1, -1), 4),
        ('LEFTPADDING',   (0, 0),  (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0),  (-1, -1), 6),
        ('LINEBELOW',     (0, 0),  (-1, 0),  1.0, ACCENT),
        ('GRID',          (0, 0),  (-1, -1), 0.3, TABLE_BORDER),
        ('LINEBELOW',     (0, -1), (-1, -1), 0.75, NAVY),
    ]

    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_ALT_BG))

    t.setStyle(TableStyle(cmds))
    return t


def df_to_ib_table(df: pd.DataFrame, max_rows=25, font_size=8, col_widths=None):
    if df is None or df.empty:
        return Paragraph('No data available.', build_styles()['body_small'])
    limited = df.head(max_rows).copy()
    data = [list(limited.columns)] + limited.astype(str).values.tolist()
    return ib_table(data, col_widths=col_widths, font_size=font_size)


# ─────────────────────────────────────────────
#  TWO-COLUMN COMPANY METRICS LAYOUT
# ─────────────────────────────────────────────
def company_comparison_table(m1, m2, ticker1, ticker2, styles):
    rows = [
        ['Metric', m1['name'], m2['name']],
        ['Ticker',        ticker1,                              ticker2],
        ['Market Cap',    f"₹{m1.get('market_cap', 0)/1e9:,.1f}B",  f"₹{m2.get('market_cap', 0)/1e9:,.1f}B"],
        ['Revenue (TTM)', f"₹{m1.get('revenue', 0)/1e9:,.1f}B",     f"₹{m2.get('revenue', 0)/1e9:,.1f}B"],
        ['EBITDA',        f"₹{m1.get('ebitda', 0)/1e9:,.1f}B",      f"₹{m2.get('ebitda', 0)/1e9:,.1f}B"],
        ['Net Income',    f"₹{m1.get('net_income', 0)/1e9:,.1f}B",  f"₹{m2.get('net_income', 0)/1e9:,.1f}B"],
        ['P/E Ratio',     f"{m1.get('pe_ratio', 0):.1f}x",           f"{m2.get('pe_ratio', 0):.1f}x"],
        ['EV/EBITDA',     f"{m1.get('ev_ebitda', 0):.1f}x",          f"{m2.get('ev_ebitda', 0):.1f}x"],
    ]
    w = [2.1 * inch, 2.2 * inch, 2.2 * inch]
    return ib_table(rows, col_widths=w, font_size=9)


# ─────────────────────────────────────────────
#  MATPLOTLIB CHART HELPERS
# ─────────────────────────────────────────────
def _ib_chart_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#F8F9FB')
    ax.figure.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#D1D5DB')
    ax.tick_params(colors='#6B7280', labelsize=8)
    ax.yaxis.label.set_color('#6B7280')
    ax.xaxis.label.set_color('#6B7280')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color='#0A1628', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', color='#E5E7EB', linewidth=0.7, linestyle='--')


def _fig_to_image(fig, width=6.2 * inch, height=3.2 * inch):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image(BytesIO(buf.getvalue()), width=width, height=height)


# ─────────────────────────────────────────────
#  NEXT-PAGE TEMPLATE SWITCH FLOWABLE
#  FIX: Lets us explicitly switch from the Cover
#  template to the Body template after page 1.
# ─────────────────────────────────────────────
from reportlab.platypus.flowables import Flowable

class _SwitchToBody(Flowable):
    """Zero-height flowable that switches the active PageTemplate to 'Body'."""
    def __init__(self):
        Flowable.__init__(self)
        self.width  = 0
        self.height = 0

    def draw(self):
        pass

    def beforeDrawOn(self, canv, doc):
        pass

    def wrap(self, availW, availH):
        self._doc = self._doctemplate if hasattr(self, '_doctemplate') else None
        return 0, 0

    def getSpaceAfter(self):
        return 0

    def getSpaceBefore(self):
        return 0

    # ReportLab calls this before placing the flowable
    def identity(self, maxLen=None):
        return '_SwitchToBody'


class _NextTemplate(Flowable):
    """
    Reliably switches PageTemplate by name for the NEXT page.
    Equivalent to doctemplate.handle_nextPageTemplate.
    """
    def __init__(self, template_name):
        Flowable.__init__(self)
        self._name = template_name

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        self.canv._doctemplate.handle_nextPageTemplate(self._name)


# ─────────────────────────────────────────────
#  MAIN REPORT CLASS
# ─────────────────────────────────────────────

class MAReportGenerator:
    """Generate an investment-bank–quality PDF report for M&A analysis."""

    def __init__(self, company1, company2, dcf1, dcf2, comps1, comps2,
                 precedent1, precedent2, val_model, assumptions,
                 deal_terms, synergies, accretion, report_context=None):
        _ensure_fonts()
        self.company1       = company1
        self.company2       = company2
        self.dcf1           = dcf1
        self.dcf2           = dcf2
        self.comps1         = comps1
        self.comps2         = comps2
        self.precedent1     = precedent1
        self.precedent2     = precedent2
        self.val_model      = val_model
        self.assumptions    = assumptions
        self.deal_terms     = deal_terms
        self.synergies      = synergies
        self.accretion      = accretion
        self.report_context = report_context or {}
        self.timestamp      = datetime.now().strftime('%d %B %Y')

        self.m1 = company1.get_key_metrics()
        self.m2 = company2.get_key_metrics()

        self.acquirer_eps  = 0
        self.pro_forma_eps = 0

        self.S = build_styles()

    # ── Utility ──────────────────────────────

    def _money_b(self, value) -> str:
        try:
            return f'₹{float(value)/1e9:,.1f}B'
        except Exception:
            return 'N/A'

    def _section(self, label):
        return section_rule(self.S, label)

    # ── Chart generators ─────────────────────

    def _football_field_image(self):
        try:
            dcf_value = float(getattr(self.dcf2, 'per_share', 0) or 0)
            comps_value = float(getattr(self.comps2, 'per_share_weighted', 0) or 0)
            precedent_value = float(getattr(self.precedent2, 'per_share_with_premium', 0) or 0)
            current_price = float(getattr(self.dcf2, 'current_price', 0) or 0)
            premium = float(self.deal_terms.get('premium', 30) or 30)
            offer_price = current_price * (1 + premium / 100.0) if current_price > 0 else 0

            methods = ['DCF', 'Trading Comps', 'Precedent Transactions']
            values = [dcf_value, comps_value, precedent_value]
            ranges = [
                (dcf_value * 0.85, dcf_value * 1.15) if dcf_value > 0 else (0, 0),
                tuple(getattr(self.comps2, 'per_share_range', (comps_value * 0.8, comps_value * 1.2)))
                if comps_value > 0 else (0, 0),
                tuple(getattr(self.precedent2, 'per_share_range', (precedent_value * 0.8, precedent_value * 1.2)))
                if precedent_value > 0 else (0, 0),
            ]
            error_low = [max(value - low, 0) for value, (low, _) in zip(values, ranges)]
            error_high = [max(high - value, 0) for value, (_, high) in zip(values, ranges)]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                methods,
                values,
                xerr=[error_low, error_high],
                capsize=5,
                color=['#1A3A6B', '#2E5FA3', '#4A7EC7'],
                edgecolor='white',
                linewidth=1.5,
                zorder=3,
            )

            max_value = max([current_price, offer_price, *[value + error for value, error in zip(values, error_high)]], default=1)
            if current_price > 0:
                ax.axvline(x=current_price, color='green', linestyle='--', linewidth=2, label=f'Current: ₹{current_price:,.0f}')
            if offer_price > 0:
                ax.axvline(x=offer_price, color='red', linestyle='--', linewidth=2, label=f'Offer: ₹{offer_price:,.0f}')

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_width() + max_value * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'₹{value:,.0f}',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='#0A1628',
                )

            ax.set_xlabel('Value per Share (₹)', fontsize=10)
            ax.set_title('Valuation Football Field', fontsize=12, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_xlim(left=0, right=max_value * 1.18 if max_value > 0 else 1)
            ax.legend(loc='lower right')
            plt.tight_layout()
            return _fig_to_image(fig, width=6.2 * inch, height=2.9 * inch)
        except Exception as e:
            print(f'Football field unavailable: {e}')
            return Paragraph('Chart unavailable.', self.S['body_small'])

    def _sensitivity_image(self):
        try:
            sensitivity_matrix = getattr(self.dcf2, 'sensitivity_matrix', {}) or {}
            wacc_vars = list(sensitivity_matrix.get('wacc_variations', []))
            growth_vars = list(sensitivity_matrix.get('growth_variations', []))
            value_factors = sensitivity_matrix.get('values', {}) or {}

            if not wacc_vars:
                wacc_vars = [getattr(self.dcf2, 'wacc', 0) - 0.01, getattr(self.dcf2, 'wacc', 0), getattr(self.dcf2, 'wacc', 0) + 0.01]
            if not growth_vars:
                growth_vars = [getattr(self.dcf2, 'terminal_growth', 0) - 0.005, getattr(self.dcf2, 'terminal_growth', 0), getattr(self.dcf2, 'terminal_growth', 0) + 0.005]

            values = []
            for wacc in wacc_vars:
                row = []
                for growth in growth_vars:
                    if wacc in value_factors and growth in value_factors.get(wacc, {}):
                        value = value_factors[wacc][growth]
                    elif wacc > 0:
                        value = float(getattr(self.dcf2, 'per_share', 0) or 0) * (getattr(self.dcf2, 'wacc', 0) / wacc) * ((1 + growth) / (1 + getattr(self.dcf2, 'terminal_growth', 0)))
                    else:
                        value = float(getattr(self.dcf2, 'per_share', 0) or 0)
                    row.append(value)
                values.append(row)

            values_array = np.array(values, dtype=float)
            fig, ax = plt.subplots(figsize=(8, 4.6))
            cmap = LinearSegmentedColormap.from_list('sensitivity', ['#450A0A', '#1E2535', '#064E3B'])
            image = ax.imshow(values_array, cmap=cmap, aspect='auto')

            ax.set_xticks(np.arange(len(growth_vars)))
            ax.set_xticklabels([f'{growth:.1%}' for growth in growth_vars])
            ax.set_yticks(np.arange(len(wacc_vars)))
            ax.set_yticklabels([f'{wacc:.1%}' for wacc in wacc_vars])

            midpoint = float(np.nanmax(values_array) + np.nanmin(values_array)) / 2 if values_array.size else 0
            for row_index, row in enumerate(values_array):
                for column_index, value in enumerate(row):
                    text_color = 'white' if value < midpoint else '#0A1628'
                    ax.text(column_index, row_index, f'₹{value:,.0f}', ha='center', va='center', fontsize=8.5, fontweight='bold', color=text_color)

            ax.set_xlabel('Terminal Growth Rate', fontsize=10)
            ax.set_ylabel('WACC', fontsize=10)
            ax.set_title('DCF Sensitivity: WACC vs Terminal Growth', fontsize=12, fontweight='bold')
            ax.set_facecolor('#F8F9FB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#D1D5DB')
            ax.spines['bottom'].set_color('#D1D5DB')
            ax.tick_params(colors='#6B7280', labelsize=8)
            ax.grid(False)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label('Per Share Value (₹)')
            plt.tight_layout()
            return _fig_to_image(fig, width=6.2 * inch, height=3.3 * inch)
        except Exception as e:
            print(f'Sensitivity chart unavailable: {e}')
            return Paragraph('Chart unavailable.', self.S['body_small'])

    def _dcf_waterfall_image(self):
        try:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels  = ['Stage 1\nHigh Growth', 'Stage 2\nTransition', 'Terminal\nValue', 'Enterprise\nValue']
            s1, s2, tv = self.dcf2.stage1_pv, self.dcf2.stage2_pv, self.dcf2.terminal_pv
            ev = s1 + s2 + tv
            values = [s1, s2, tv, ev]
            bar_colors = ['#1A3A6B', '#2E5FA3', '#4A7EC7', '#C8A84B']

            bars = ax.bar(labels, [v / 1e9 for v in values], color=bar_colors,
                          width=0.55, zorder=3, edgecolor='white', linewidth=1.2)

            for bar, val in zip(bars, values):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h + 2,
                        f'₹{val/1e9:.0f}B', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color='#0A1628')

            for i in range(2):
                x0 = bars[i].get_x()   + bars[i].get_width()
                x1 = bars[i + 1].get_x()
                y  = bars[i].get_height()
                ax.plot([x0, x1], [y / 1e9, y / 1e9], color='#D1D5DB',
                        linewidth=0.8, linestyle='--', zorder=2)

            _ib_chart_style(ax, title='DCF Enterprise Value Waterfall', ylabel='₹ Billion')
            ax.set_ylim(0, max(v / 1e9 for v in values) * 1.2)
            plt.tight_layout()
            return _fig_to_image(fig, width=6.2 * inch, height=3.2 * inch)
        except Exception as e:
            print(f'DCF chart unavailable: {e}')
            return Paragraph('Chart unavailable.', self.S['body_small'])

    def _synergy_chart_image(self):
        try:
            rev  = self.synergies.get('annual_rev', 0)  / 1e9
            cost = self.synergies.get('annual_cost', 0) / 1e9
            categories = ['Revenue\nSynergies', 'Cost\nSynergies', 'Total\nSynergies']
            values_ann = [rev, cost, rev + cost]
            pv_vals = [
                self.synergies.get('pv_rev', 0)   / 1e9,
                self.synergies.get('pv_cost', 0)  / 1e9,
                self.synergies.get('pv_total', 0) / 1e9,
            ]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(categories))
            w = 0.35
            b1 = ax.bar(x - w / 2, values_ann, w, label='Annual Run-Rate',
                        color='#1A3A6B', zorder=3, edgecolor='white')
            b2 = ax.bar(x + w / 2, pv_vals,    w, label='Present Value',
                        color='#C8A84B', zorder=3, edgecolor='white')

            for bar in list(b1) + list(b2):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                            f'₹{h:.1f}B', ha='center', va='bottom', fontsize=7.5,
                            color='#0A1628', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend(fontsize=8, framealpha=0)
            _ib_chart_style(ax, title='Synergy Summary', ylabel='₹ Billion')
            plt.tight_layout()
            return _fig_to_image(fig, width=5.0 * inch, height=2.8 * inch)
        except Exception as e:
            return Paragraph('Chart unavailable.', self.S['body_small'])

    def _add_ai_assumptions_section(self, story):
        """Add a section showing AI-generated assumptions if they're active."""
        report_context = self.report_context or {}
        ai_active = report_context.get("ai_active", False)
        
        if not ai_active:
            return
        
        story += self._section('07  |  AI-Generated Assumptions')
        story.append(Paragraph('Machine Learning Enhanced Valuation Inputs', self.S['h1']))
        story.append(thin_rule())
        
        # Add note about AI usage
        ai_mode = report_context.get("ai_mode", "base").upper()
        note = (
            f'<i>This report uses AI-generated assumptions generated by Groq LLM in <b>{ai_mode}</b> mode. '
            f'The AI analyzed sector context, company financials, and market conditions to produce these inputs.</i>'
        )
        story.append(Paragraph(note, self.S['body_small']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Get AI rationales if available
        ai_rationales = report_context.get("ai_rationales", {})
        ai_confidence = report_context.get("ai_confidence", "Medium")
        
        # Create comparison table
        ai_comparison_rows = [
            ['Parameter', 'Value Used', 'AI Rationale'],
        ]
        
        # DCF assumptions
        dcf_fields = [
            ('Stage 1 Growth Rate', f"{getattr(self.assumptions, 'stage1_growth', 0):.1%}", 
             ai_rationales.get('stage1_growth', 'AI-generated based on historical growth and sector outlook')),
            ('Terminal Growth Rate', f"{getattr(self.assumptions, 'terminal_growth', 0):.1%}", 
             ai_rationales.get('terminal_growth', 'Based on long-term GDP growth and market maturity')),
            ('Stage 1 EBITDA Margin', f"{getattr(self.assumptions, 'stage1_ebitda_margin', 0):.1%}", 
             ai_rationales.get('stage1_ebitda_margin', 'AI estimated margin improvement potential')),
            ('Terminal EBITDA Margin', f"{getattr(self.assumptions, 'terminal_margin', 0):.1%}", 
             ai_rationales.get('terminal_margin', 'Convergence to sector steady-state margins')),
            ('Reinvestment Rate', f"{getattr(self.assumptions, 'reinvestment_rate', 0):.1%}", 
             ai_rationales.get('reinvestment_rate', 'Based on historical capex and working capital needs')),
        ]
        
        # Market multiple assumptions
        multiples_fields = [
            ('EV/EBITDA Multiple', f"{getattr(self.assumptions, 'ev_ebitda_multiple', 0):.1f}x", 
             ai_rationales.get('ev_ebitda_multiple', 'Sector-average comparable company multiple')),
            ('P/E Ratio', f"{getattr(self.assumptions, 'pe_multiple', 0):.1f}x", 
             ai_rationales.get('pe_multiple', 'Peer group valuation benchmark')),
            ('Control Premium', f"{getattr(self.assumptions, 'control_premium', 0):.1%}", 
             ai_rationales.get('control_premium', 'Based on historical M&A transactions in sector')),
        ]
        
        # Macro assumptions
        macro_fields = [
            ('Risk-Free Rate', f"{getattr(self.assumptions, 'risk_free_rate', 0):.1%}", 
             ai_rationales.get('risk_free_rate', 'Current 10-year government bond yield')),
            ('Equity Risk Premium', f"{getattr(self.assumptions, 'equity_risk_premium', 0):.1%}", 
             ai_rationales.get('equity_risk_premium', 'Market-derived implied premium')),
            ('Tax Rate', f"{getattr(self.assumptions, 'tax_rate', 0):.1%}", 
             ai_rationales.get('tax_rate', 'Statutory corporate tax rate')),
        ]
        
        # Combine all fields
        all_fields = dcf_fields + multiples_fields + macro_fields
        
        for field_name, value, rationale in all_fields:
            # Truncate rationale if too long
            if len(rationale) > 80:
                rationale = rationale[:77] + "..."
            ai_comparison_rows.append([field_name, value, rationale])
        
        # Create the table
        col_widths = [2.0 * inch, 1.2 * inch, 3.3 * inch]
        story.append(ib_table(ai_comparison_rows, col_widths=col_widths, font_size=8, zebra=True))
        
        # Add AI confidence
        story.append(Spacer(1, 0.1 * inch))
        
        confidence_color = POSITIVE_GREEN if ai_confidence == "High" else CAUTION_AMBER if ai_confidence == "Medium" else NEGATIVE_RED
        story.append(Paragraph(
            f'<b>AI Confidence Level:</b> <font color="{confidence_color.hexval() if hasattr(confidence_color, "hexval") else "#B45309"}">{ai_confidence}</font>',
            self.S['body']
        ))
        
        # Add key risks identified by AI
        ai_risks = report_context.get("ai_key_risks", [])
        if ai_risks:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph('Key Risks Identified by AI', self.S['h2']))
            for risk in ai_risks:
                story.append(Paragraph(f'• {risk}', self.S['body_small']))
        
        story.append(PageBreak())

    # ── Cover page (raw canvas) ───────────────

    def _build_cover(self, canv, doc):
        """
        Draws the full-bleed cover on the canvas.
        Called as the onPage callback for the 'Cover' PageTemplate.
        """
        w, h = A4
        margin = 0.65 * inch

        # Navy background slab (top 58%)
        canv.setFillColor(NAVY)
        canv.rect(0, h * 0.42, w, h * 0.58, fill=1, stroke=0)

        # Gold accent band
        canv.setFillColor(ACCENT)
        canv.rect(0, h * 0.42 - 6, w, 6, fill=1, stroke=0)

        # Firm name
        canv.setFont(FONT_BOLD, 9)
        canv.setFillColor(ACCENT)
        canv.drawString(margin, h * 0.42 + h * 0.58 - margin, 'M&A INTELLIGENCE PLATFORM')

        # Main title
        canv.setFont(FONT_BOLD, 30)
        canv.setFillColor(WHITE)
        canv.drawString(margin, h * 0.68, 'Comprehensive')
        canv.drawString(margin, h * 0.68 - 36, 'Valuation Report')

        # Subtitle
        canv.setFont(FONT_REGULAR, 14)
        canv.setFillColor(colors.HexColor('#BFC9D9'))
        canv.drawString(margin, h * 0.68 - 72, 'Merger & Acquisition Analysis')

        # Divider
        canv.setFillColor(ACCENT)
        canv.rect(margin, h * 0.68 - 84, 3 * inch, 1.5, fill=1, stroke=0)

        # Party labels
        canv.setFont(FONT_REGULAR, 8.5)
        canv.setFillColor(colors.HexColor('#8DA0BC'))
        canv.drawString(margin, h * 0.68 - 106, 'ACQUIRER')
        canv.drawString(margin + 2.8 * inch, h * 0.68 - 106, 'TARGET')

        canv.setFont(FONT_BOLD, 13)
        canv.setFillColor(WHITE)
        canv.drawString(margin, h * 0.68 - 124, self.m1['name'])
        canv.setFont(FONT_REGULAR, 11)
        canv.setFillColor(colors.HexColor('#BFC9D9'))
        canv.drawString(margin, h * 0.68 - 140, self.company1.ticker)

        canv.setFont(FONT_BOLD, 13)
        canv.setFillColor(WHITE)
        canv.drawString(margin + 2.8 * inch, h * 0.68 - 124, self.m2['name'])
        canv.setFont(FONT_REGULAR, 11)
        canv.setFillColor(colors.HexColor('#BFC9D9'))
        canv.drawString(margin + 2.8 * inch, h * 0.68 - 140, self.company2.ticker)

        # Arrow
        canv.setFont(FONT_BOLD, 18)
        canv.setFillColor(ACCENT)
        canv.drawString(margin + 1.9 * inch, h * 0.68 - 128, '→')

        # Bottom metadata
        meta_y = h * 0.42 - 60
        canv.setFont(FONT_BOLD, 8)
        canv.setFillColor(NAVY)
        canv.drawString(margin, meta_y, 'DATE OF PREPARATION')

        canv.setFont(FONT_REGULAR, 10)
        canv.setFillColor(BLACK)
        canv.drawString(margin, meta_y - 16, self.timestamp)

        # Bottom disclaimer
        canv.setFont(FONT_OBLIQUE, 6.5)
        canv.setFillColor(MID_GREY)
        text = ('This document is prepared solely for informational purposes and does not '
                'constitute investment advice, an offer to sell, or a solicitation to buy securities.')
        canv.drawString(margin, margin * 0.8, text)

    # ── Main generate ─────────────────────────

    def generate(self, filename='MA_Report.pdf'):
        """
        Build the PDF.

        Page layout:
          Page 1  → 'Cover' template  (full-bleed canvas art, no body frame content)
          Page 2+ → 'Body'  template  (header/footer chrome + text frame)
        """
        doc = IBDocTemplate(
            filename,
            acquirer_name=self.m1['name'],
            target_name=self.m2['name'],
            cover_callback=self._build_cover,
            pagesize=A4,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.80 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []

        # ── FIX: The cover page is drawn entirely by _build_cover via the
        #    'Cover' PageTemplate onPage callback. We push a NextPageTemplate
        #    flowable so that the VERY FIRST page break switches to 'Body'.
        #    Then we immediately break to start body content on page 2.
        story.append(_NextTemplate('Body'))
        story.append(PageBreak())   # end page 1 (cover) → start page 2 (body)

        # ════════════════════════════════════════
        #  1. EXECUTIVE SUMMARY
        # ════════════════════════════════════════
        story += self._section('01  |  Executive Summary')
        story.append(Paragraph('Transaction Summary', self.S['h1']))
        story.append(thin_rule())

        valid_values = [v for v in [
            getattr(self.dcf2,       'per_share',              0),
            getattr(self.comps2,     'per_share_weighted',     0),
            getattr(self.precedent2, 'per_share_with_premium', 0),
        ] if v and v > 0]
        weighted_avg = sum(valid_values) / len(valid_values) if valid_values else 0
        offer_price  = self.dcf2.current_price * (1 + self.deal_terms.get('premium', 30) / 100)

        kpis = [
            ('DCF Value / Share',    f'₹{self.dcf2.per_share:,.0f}',   f'{self.dcf2.implied_premium:+.0f}%'),
            ('Trading Comps',        f'₹{self.comps2.per_share_weighted:,.0f}', ''),
            ('Precedent Txns',       f'₹{self.precedent2.per_share_with_premium:,.0f}', ''),
            ('Blended Avg',          f'₹{weighted_avg:,.0f}',          ''),
            ('Offer Price',          f'₹{offer_price:,.0f}',           f'+{self.deal_terms.get("premium",30)}%'),
            ('Offer Premium',        f'{self.deal_terms.get("premium",30)}%', ''),
            ('EPS Accretion / (Dil)',f'{self.accretion:+.1f}%',        ''),
            ('Current Price',        f'₹{self.dcf2.current_price:,.2f}', ''),
        ]
        story.append(kpi_card_table(kpis, col_count=4))
        story.append(Spacer(1, 0.15 * inch))

        narrative = (
            f'This report presents a comprehensive valuation analysis of <b>{self.m2["name"]}</b> '
            f'({self.company2.ticker}) as a potential acquisition target for <b>{self.m1["name"]}</b> '
            f'({self.company1.ticker}). Three independent valuation methodologies — discounted cash flow, '
            f'trading comparables, and precedent transactions — converge on a blended implied value of '
            f'<b>₹{weighted_avg:,.0f} per share</b>, representing a '
            f'<b>{self.dcf2.implied_premium:+.0f}%</b> premium to the current trading price. '
            f'The proposed offer price of <b>₹{offer_price:,.0f}</b> implies a '
            f'<b>{self.deal_terms.get("premium", 30)}%</b> control premium. '
            f'On an accretion/dilution basis, the transaction is projected to be '
            f'<b>{"accretive" if self.accretion >= 0 else "dilutive"}</b> to acquirer EPS '
            f'by <b>{abs(self.accretion):.1f}%</b>.'
        )
        story.append(Paragraph(narrative, self.S['body']))
        story.append(PageBreak())

        # ════════════════════════════════════════
        #  2. TRANSACTION OVERVIEW
        # ════════════════════════════════════════
        story += self._section('02  |  Transaction Overview')
        story.append(Paragraph('Deal Structure & Key Terms', self.S['h1']))
        story.append(thin_rule())

        deal_rows = [
            ['Parameter', 'Value', 'Notes'],
            ['Target Current Share Price',  f'₹{self.dcf2.current_price:,.2f}',          'As at date of analysis'],
            ['Offer Premium',               f'{self.deal_terms.get("premium", 30):.1f}%', 'Over last close'],
            ['Implied Offer Price',         f'₹{offer_price:,.2f}',                      'Per share'],
            ['Cash Consideration',          f'{self.deal_terms.get("cash_pct", 60):.0f}%',''],
            ['Stock Consideration',         f'{100 - self.deal_terms.get("cash_pct", 60):.0f}%', ''],
            ['Acquirer Standalone EPS',     f'₹{self.acquirer_eps:,.2f}' if self.acquirer_eps else 'N/A', ''],
            ['Pro-Forma EPS',               f'₹{self.pro_forma_eps:,.2f}' if self.pro_forma_eps else 'N/A', ''],
            ['EPS Accretion / (Dilution)',  f'{self.accretion:+.1f}%',                   'vs Acquirer standalone'],
        ]
        story.append(ib_table(deal_rows,
                              col_widths=[2.8 * inch, 1.8 * inch, 1.9 * inch],
                              font_size=9))
        story.append(PageBreak())

        # ════════════════════════════════════════
        #  3. COMPANY OVERVIEW
        # ════════════════════════════════════════
        story += self._section('03  |  Company Overview')
        story.append(Paragraph('Comparative Financial Profile', self.S['h1']))
        story.append(thin_rule())

        story.append(company_comparison_table(
            self.m1, self.m2,
            self.company1.ticker, self.company2.ticker,
            self.S
        ))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph('Target — Additional Detail', self.S['h2']))
        target_detail = [
            ['Metric', 'Value'],
            ['52-Week High', f'₹{self.m2.get("week_52_high", 0):,.0f}'],
            ['52-Week Low',  f'₹{self.m2.get("week_52_low",  0):,.0f}'],
            ['Cash & Equivalents', f'₹{self.m2.get("cash", 0)/1e9:,.1f}B'],
            ['Total Debt',         f'₹{self.m2.get("total_debt", 0)/1e9:,.1f}B'],
            ['Net Debt',           f'₹{(self.m2.get("total_debt", 0) - self.m2.get("cash", 0))/1e9:,.1f}B'],
        ]
        story.append(ib_table(target_detail, col_widths=[3.0 * inch, 3.0 * inch], font_size=9))
        story.append(PageBreak())

        # ════════════════════════════════════════
        #  4. VALUATION ANALYSIS
        # ════════════════════════════════════════
        story += self._section('04  |  Valuation Analysis')

        story.append(Paragraph('Valuation Football Field', self.S['h1']))
        story.append(thin_rule())
        story.append(Paragraph(
            'The football field chart below illustrates the implied equity value per share '
            'across three independent methodologies, benchmarked against the current share '
            'price and proposed offer price.',
            self.S['body']
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(self._football_field_image())
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph('DCF — Enterprise Value Waterfall', self.S['h2']))
        story.append(self._dcf_waterfall_image())
        story.append(Spacer(1, 0.1 * inch))

        ev = max(self.dcf2.enterprise_value, 1)
        dcf_rows = [
            ['Component', '₹ Billion', '% of EV'],
            ['Stage 1 — High Growth FCF',
             f'{self.dcf2.stage1_pv/1e9:.1f}',
             f'{self.dcf2.stage1_pv/ev*100:.1f}%'],
            ['Stage 2 — Transition FCF',
             f'{self.dcf2.stage2_pv/1e9:.1f}',
             f'{self.dcf2.stage2_pv/ev*100:.1f}%'],
            ['Terminal Value (Gordon Growth)',
             f'{self.dcf2.terminal_pv/1e9:.1f}',
             f'{self.dcf2.terminal_pv/ev*100:.1f}%'],
            ['Total Enterprise Value',
             f'{self.dcf2.enterprise_value/1e9:.1f}', '100.0%'],
            ['Less: Net Debt',
             f'({(self.m2.get("total_debt", 0) - self.m2.get("cash", 0))/1e9:.1f})', ''],
            ['Equity Value',
             f'{self.dcf2.equity_value/1e9:.1f}', ''],
            ['DCF Per Share', f'₹{self.dcf2.per_share:,.0f}', ''],
        ]
        story.append(ib_table(dcf_rows,
                              col_widths=[3.4 * inch, 1.5 * inch, 1.6 * inch],
                              font_size=9))
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph('Detailed Free Cash Flow Projections', self.S['h2']))
        proj_years = getattr(self.dcf2, 'projection_years', []) or []
        rev_proj   = getattr(self.dcf2, 'revenue_projections', []) or []
        fcf_proj   = getattr(self.dcf2, 'fcf_projections', []) or []
        fcf_rows   = [['Year', 'Revenue (₹B)', 'FCF (₹B)', 'PV Factor', 'PV of FCF (₹B)']]
        wacc = getattr(self.dcf2, 'wacc', 0) or 0
        for i, yr in enumerate(proj_years):
            rev = rev_proj[i] if i < len(rev_proj) else 0
            fcf = fcf_proj[i] if i < len(fcf_proj) else 0
            pv  = 1 / (1 + wacc) ** (i + 0.5) if wacc else 0
            fcf_rows.append([
                str(yr),
                f'{rev/1e9:.2f}',
                f'{fcf/1e9:.2f}',
                f'{pv:.4f}',
                f'{fcf*pv/1e9:.2f}',
            ])
        story.append(ib_table(fcf_rows,
                              col_widths=[0.9 * inch, 1.5 * inch, 1.3 * inch, 1.3 * inch, 1.5 * inch],
                              font_size=8))
        story.append(PageBreak())

        story += self._section('04  |  Valuation Analysis (cont.)')
        story.append(Paragraph('WACC / Terminal Growth Rate Sensitivity', self.S['h1']))
        story.append(thin_rule())
        story.append(Paragraph(
            'The sensitivity table shows the implied DCF per share across a range of WACC '
            'and terminal growth rate assumptions. Shaded cells indicate values above the '
            'proposed offer price.', self.S['body']
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(self._sensitivity_image())
        story.append(PageBreak())

        # ════════════════════════════════════════
        #  5. DEAL MECHANICS
        # ════════════════════════════════════════
        story += self._section('05  |  Deal Mechanics & Accretion Analysis')
        story.append(Paragraph('Sources & Uses / Accretion–Dilution', self.S['h1']))
        story.append(thin_rule())

        ppa = self.report_context.get('ppa', {})
        if ppa:
            sources_df = ppa.get('sources_df')
            uses_df    = ppa.get('uses_df')

            if isinstance(sources_df, pd.DataFrame) and not sources_df.empty:
                story.append(Paragraph('Sources of Funds', self.S['h2']))
                story.append(df_to_ib_table(sources_df, font_size=8))
                story.append(Spacer(1, 0.1 * inch))

            if isinstance(uses_df, pd.DataFrame) and not uses_df.empty:
                story.append(Paragraph('Uses of Funds', self.S['h2']))
                story.append(df_to_ib_table(uses_df, font_size=8))
                story.append(Spacer(1, 0.15 * inch))

            ppa_result = ppa.get('ppa_result', {})
            if ppa_result:
                story.append(Paragraph('Purchase Price Allocation (PPA)', self.S['h2']))
                ppa_rows = [
                    ['Component', '₹ Billion'],
                    ['Purchase Price',           f'{ppa_result.get("purchase_price", 0)/1e9:.2f}'],
                    ['Target Book Value',         f'{ppa_result.get("target_book_value", 0)/1e9:.2f}'],
                    ['Tangible Book Value',       f'{ppa_result.get("target_tangible_book_value", 0)/1e9:.2f}'],
                    ['PP&E Write-up',             f'{ppa_result.get("ppe_write_up", 0)/1e9:.2f}'],
                    ['Identifiable Intangibles',  f'{ppa_result.get("intangibles_write_up", 0)/1e9:.2f}'],
                    ['Goodwill',                  f'{ppa_result.get("goodwill", 0)/1e9:.2f}'],
                ]
                story.append(ib_table(ppa_rows, col_widths=[3.5 * inch, 3.0 * inch], font_size=9))
                story.append(Spacer(1, 0.12 * inch))

            pro_forma_df = ppa.get('pro_forma_df')
            if isinstance(pro_forma_df, pd.DataFrame) and not pro_forma_df.empty:
                story.append(Paragraph('Pro Forma Balance Sheet', self.S['h2']))
                story.append(df_to_ib_table(pro_forma_df, max_rows=18, font_size=7))
                story.append(Spacer(1, 0.12 * inch))

            ppa_metrics_rows = [
                ['Metric', 'Value'],
                ['Annual Amortization Expense',    self._money_b(ppa.get('annual_amortization', 0))],
                ['After-Tax Amortization',         self._money_b(ppa.get('annual_amortization_after_tax', 0))],
                ['Annual Interest Expense',        self._money_b(ppa.get('annual_interest', 0))],
                ['After-Tax Interest Expense',     self._money_b(ppa.get('annual_interest_after_tax', 0))],
                ['Total Annual PPA Impact',        self._money_b(ppa.get('total_annual_impact', 0))],
                ['Impact on Pro Forma EPS',        f'₹{ppa.get("impact_per_share", 0):.2f}'],
            ]
            story.append(Paragraph('PPA Impact Summary', self.S['h2']))
            story.append(ib_table(ppa_metrics_rows, col_widths=[3.5 * inch, 3.0 * inch], font_size=9))
        else:
            deal_mech_rows = [
                ['Parameter', 'Value'],
                ['Target Price',               f'₹{self.dcf2.current_price:,.2f}'],
                ['Offer Price',                f'₹{offer_price:,.2f}'],
                ['Premium to Current',         f'{self.deal_terms.get("premium", 30)}%'],
                ['Cash Component',             f'{self.deal_terms.get("cash_pct", 60)}%'],
                ['Stock Component',            f'{100 - self.deal_terms.get("cash_pct", 60)}%'],
                ['Acquirer Standalone EPS',    f'₹{self.acquirer_eps:,.2f}' if self.acquirer_eps else 'N/A'],
                ['Pro-Forma EPS',              f'₹{self.pro_forma_eps:,.2f}' if self.pro_forma_eps else 'N/A'],
                ['EPS Accretion / (Dilution)', f'{self.accretion:+.1f}%'],
            ]
            story.append(ib_table(deal_mech_rows, col_widths=[3.4 * inch, 3.1 * inch], font_size=9))

        story.append(PageBreak())

        # ════════════════════════════════════════
        #  6. SYNERGY ANALYSIS
        # ════════════════════════════════════════
        story += self._section('06  |  Synergy Analysis')
        story.append(Paragraph('Identified Synergies', self.S['h1']))
        story.append(thin_rule())

        syn_rows = [
            ['Synergy Type', 'Annual Run-Rate', 'Present Value', '% of Deal Value'],
            ['Revenue Synergies',
             self._money_b(self.synergies.get('annual_rev', 0)),
             self._money_b(self.synergies.get('pv_rev', 0)),
             f'{self.synergies.get("pv_rev", 0)/(self.m2.get("market_cap", 1)+1)*100:.1f}%'],
            ['Cost Synergies',
             self._money_b(self.synergies.get('annual_cost', 0)),
             self._money_b(self.synergies.get('pv_cost', 0)),
             f'{self.synergies.get("pv_cost", 0)/(self.m2.get("market_cap", 1)+1)*100:.1f}%'],
            ['Total Synergies',
             self._money_b(self.synergies.get('annual_total', 0)),
             self._money_b(self.synergies.get('pv_total', 0)),
             f'{self.synergies.get("pv_total", 0)/(self.m2.get("market_cap", 1)+1)*100:.1f}%'],
        ]
        story.append(ib_table(syn_rows,
                              col_widths=[2.0 * inch, 1.6 * inch, 1.6 * inch, 1.3 * inch],
                              font_size=9))
        story.append(Spacer(1, 0.15 * inch))
        story.append(self._synergy_chart_image())
        story.append(PageBreak())

        # ════════════════════════════════════════
        #  7. VALUATION ASSUMPTIONS (Manual)
        # ════════════════════════════════════════
        story += self._section('08  |  Valuation Assumptions')
        story.append(Paragraph('Key Modelling Assumptions', self.S['h1']))
        story.append(thin_rule())
        try:
            assumptions_df = self.assumptions.to_dataframe()
            story.append(df_to_ib_table(assumptions_df.head(25), font_size=8))
        except Exception as e:
            story.append(Paragraph(f'Assumptions data unavailable: {e}', self.S['body_small']))
        story.append(PageBreak())
        
        # ════════════════════════════════════════
        #  8. AI ASSUMPTIONS SECTION (if active)
        # ════════════════════════════════════════
        self._add_ai_assumptions_section(story)
        
        # ════════════════════════════════════════
        #  9. INVESTMENT RECOMMENDATION
        # ════════════════════════════════════════
        story += self._section('09  |  Investment Recommendation')
        story.append(Paragraph('Summary Recommendation', self.S['h1']))
        story.append(thin_rule())

        memo = self.report_context.get('memo', {})
        recommendation = memo.get('recommendation')
        rec_color_key  = memo.get('rec_color')

        color_map = {'success': POSITIVE_GREEN, 'warning': CAUTION_AMBER, 'error': NEGATIVE_RED}
        if recommendation:
            rec_color_hex = color_map.get(rec_color_key, NAVY)
        else:
            if self.accretion > 5 and self.dcf2.implied_premium < 30:
                recommendation, rec_color_hex = 'PROCEED', POSITIVE_GREEN
            elif self.accretion > 0 or self.synergies.get('pv_total', 0) > self.m2.get('market_cap', 0) * 0.2:
                recommendation, rec_color_hex = 'CONSIDER WITH CAUTION', CAUTION_AMBER
            else:
                recommendation, rec_color_hex = 'DO NOT PROCEED', NEGATIVE_RED

        rec_text_map = {
            'PROCEED':               'Strong strategic rationale with accretive deal economics.',
            'CONSIDER WITH CAUTION': 'Moderate benefits identified — recommend price negotiation and further diligence.',
            'DO NOT PROCEED':        'Unfavorable economics at current pricing — explore alternative structures.',
        }
        rec_sub = rec_text_map.get(recommendation, '')

        hex_str = rec_color_hex.hexval() if hasattr(rec_color_hex, 'hexval') else str(rec_color_hex)
        rec_badge_data = [[
            Paragraph(f'<font color="{hex_str}"><b>{recommendation}</b></font>',
                      ParagraphStyle('rb', fontSize=16, fontName=FONT_BOLD,
                                     textColor=rec_color_hex, alignment=TA_CENTER)),
        ]]
        rec_badge = Table(rec_badge_data, colWidths=[6.5 * inch])
        rec_badge.setStyle(TableStyle([
            ('BOX',           (0, 0), (-1, -1), 1.5, rec_color_hex),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
            ('TOPPADDING',    (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GREY),
        ]))
        story.append(rec_badge)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(rec_sub, self.S['body']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph('Key Rationale', self.S['h2']))
        default_drivers = [
            f'Valuation: Target assessed as {getattr(self.dcf2, "fairness_rating", "fairly valued").lower()} '
            f'({self.dcf2.implied_premium:+.0f}% premium to DCF intrinsic value)',
            f'Accretion/Dilution: {self.accretion:+.1f}% impact on acquirer standalone EPS',
            f'Synergy Value: ₹{self.synergies.get("pv_total", 0)/1e9:.1f}B present value of identified synergies',
            f'Control Premium: {self.deal_terms.get("premium", 30)}% proposed vs '
            f'{getattr(self.precedent2, "control_premium", 0.25):.0%} sector average',
        ]
        drivers = memo.get('key_drivers', default_drivers)
        for d in drivers:
            story.append(Paragraph(f'◆  {d}', ParagraphStyle(
                'bullet', fontSize=9, fontName=FONT_REGULAR,
                textColor=BLACK, leftIndent=12, spaceAfter=4, leading=13
            )))

        if memo.get('risks') is not None:
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph('Key Risk Factors', self.S['h2']))
            story.append(df_to_ib_table(memo['risks'], max_rows=10, font_size=8))

        # ── Disclaimer ───────────────────────────
        story.append(Spacer(1, 0.4 * inch))
        story.append(HRFlowable(width='100%', thickness=0.5, color=RULE_GREY))
        story.append(Spacer(1, 0.1 * inch))
        disclaimer = (
            '<b>Important Disclaimer</b><br/>'
            'This report has been prepared solely for informational purposes and does not constitute '
            'investment advice, a recommendation, or an offer or solicitation to buy or sell any security. '
            'The analysis is based on publicly available information and modelling assumptions that may '
            'not reflect actual market conditions or outcomes. All projections are inherently uncertain. '
            'Recipients should conduct their own independent analysis and consult qualified advisers '
            'before making any investment decision. Past performance is not indicative of future results.<br/><br/>'
            '<i>Generated by M&amp;A Intelligence Platform — Professional Edition</i>'
        )
        story.append(Paragraph(disclaimer, self.S['disclaimer']))

        # ── Build ────────────────────────────────
        doc.build(story)
        return filename