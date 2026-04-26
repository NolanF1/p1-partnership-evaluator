"""
P1 Motor Club — Partnership Evaluator
======================================
CLV:CAC scoring model, calibrated on 41 verified sponsor deals.
Sectors: Luxury Auto · Aviation · Watches · Apparel · Hospitality · Wealth

Usage
-----
    python evaluator.py                          # run all sector reports
    python evaluator.py --sector aviation        # single sector
    python evaluator.py --deal 1500000 --overlap 0.75 --sector aviation

Model
-----
    P1 CAC      = Deal Value / (Members × Overlap × Conv Rate)
    CLV:CAC     = Sponsor CLV / P1 CAC
    Sector Score = log-linear percentile in sector bell curve (0–100)
    Decision     = STRONG PURSUE (≥10x) | PURSUE (≥3x) | CONDITIONAL (≥1x) | DO NOT PURSUE
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA — 41 verified sponsors (v5 model)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sponsor:
    name: str
    category: str
    sector: str                   # internal sector id
    deal: float                   # deal value ($)
    overlap: float                # audience overlap (0–1)
    conv: float                   # conversion rate (0–1)
    custom_conv: bool             # whether conv rate is custom
    p1_cac: float                 # pre-computed P1 CAC ($)
    event_benchmark_cac: float    # sponsor's standard CAC ($)
    clv: float                    # customer lifetime value ($)
    decision: str                 # canonical decision label

    @property
    def clv_to_cac(self) -> float:
        return self.clv / self.p1_cac if self.p1_cac else 0.0


TRAINING: list[Sponsor] = [
    # ── Luxury Auto ──────────────────────────────────────────────────────────
    Sponsor("BMW",              "Luxury Auto", "lux-auto", 2450000, 0.65, 0.10, False, 62821,  80000,  360000, "PURSUE"),
    Sponsor("Lamborghini",      "Luxury Auto", "lux-auto", 3000000, 0.60, 0.06, False, 138889, 180000, 500000, "PURSUE"),
    Sponsor("Mercedes-AMG",     "Luxury Auto", "lux-auto", 2000000, 0.65, 0.10, False, 51282,  80000,  190000, "PURSUE"),
    Sponsor("Porsche",          "Luxury Auto", "lux-auto", 2000000, 0.65, 0.10, False, 51282,  80000,  180000, "PURSUE"),
    Sponsor("McLaren",          "Luxury Auto", "lux-auto", 2800000, 0.50, 0.06, False, 155556, 90000,  420000, "CONDITIONAL"),
    Sponsor("Cadillac",         "Luxury Auto", "lux-auto", 1200000, 0.50, 0.06, False, 66667,  68000,  130000, "CONDITIONAL"),
    Sponsor("Genesis",          "Luxury Auto", "lux-auto", 400000,  0.40, 0.03, True,  55556,  55000,  85000,  "CONDITIONAL"),
    Sponsor("Radical Racecars", "Luxury Auto", "lux-auto", 50000,   0.65, 0.10, False, 1282,   6000,   90000,  "STRONG PURSUE"),
    Sponsor("Rivian",           "Luxury Auto", "lux-auto", 600000,  0.30, 0.03, False, 111111, 85000,  95000,  "DO NOT PURSUE"),
    # ── Aviation ─────────────────────────────────────────────────────────────
    Sponsor("NetJets (BRK)",    "Aviation / Private Travel", "aviation", 1000000, 0.80, 0.10, False, 20833,  32500,   5000000, "STRONG PURSUE"),
    Sponsor("VistaJet",         "Aviation / Private Travel", "aviation", 1500000, 0.85, 0.10, False, 29412,  500000,  2250000, "STRONG PURSUE"),
    Sponsor("Sentient Jet",     "Aviation / Private Travel", "aviation", 750000,  0.65, 0.06, True,  32051,  35000,   1075000, "STRONG PURSUE"),
    Sponsor("NetJets Partner",  "Aviation / Private Travel", "aviation", 1200000, 0.50, 0.06, False, 66667,  150000,  2000000, "STRONG PURSUE"),
    Sponsor("ExoJet",           "Aviation / Private Travel", "aviation", 1200000, 0.50, 0.06, False, 66667,  150000,  2000000, "STRONG PURSUE"),
    Sponsor("Wheels Up",        "Aviation / Private Travel", "aviation", 625000,  0.60, 0.06, False, 28935,  35000,   425000,  "STRONG PURSUE"),
    Sponsor("Performance Flight","Aviation / Private Travel","aviation", 20000,   0.35, 0.03, False, 3175,   3000,    18000,   "PURSUE"),
    # ── Watches / Jewelry ────────────────────────────────────────────────────
    Sponsor("Chopard",          "Watches / Jewelry", "watches", 150000,  0.60, 0.06, False, 6944,   35000, 425000, "STRONG PURSUE"),
    Sponsor("Vacheron",         "Watches / Jewelry", "watches", 472500,  0.60, 0.06, False, 21875,  35000, 425000, "STRONG PURSUE"),
    Sponsor("Hublot",           "Watches / Jewelry", "watches", 1500000, 0.60, 0.10, True,  41667,  68000, 210000, "PURSUE"),
    Sponsor("Richard Mille",    "Watches / Jewelry", "watches", 800000,  0.45, 0.06, False, 49383,  65000, 200000, "PURSUE"),
    Sponsor("Zenith",           "Watches / Jewelry", "watches", 800000,  0.55, 0.10, True,  24242,  25000, 55000,  "CONDITIONAL"),
    Sponsor("Tudor",            "Watches / Jewelry", "watches", 660000,  0.65, 0.20, True,  8462,   1100,  10000,  "CONDITIONAL"),
    Sponsor("Omega",            "Watches / Jewelry", "watches", 1440000, 0.82, 0.15, True,  19512,  2400,  22800,  "CONDITIONAL"),
    Sponsor("Rolex",            "Watches / Jewelry", "watches", 3600000, 0.50, 0.20, True,  60000,  6000,  60000,  "CONDITIONAL"),
    Sponsor("TAG Heuer",        "Watches / Jewelry", "watches", 2520000, 0.72, 0.10, False, 58333,  4200,  30000,  "DO NOT PURSUE"),
    Sponsor("Breitling",        "Watches / Jewelry", "watches", 1860000, 0.62, 0.10, False, 50000,  3100,  18600,  "DO NOT PURSUE"),
    # ── Hospitality ──────────────────────────────────────────────────────────
    Sponsor("Exclusive Resorts","Hospitality / Hotels", "hospitality", 500000,  0.40, 0.10, True,  20833,  20000, 600000, "STRONG PURSUE"),
    Sponsor("Resort World Catskills","Hospitality / Hotels","hospitality",5375000,0.50,0.06,False,298611,25000, 350000, "CONDITIONAL"),
    Sponsor("Aman Resorts",     "Hospitality / Hotels", "hospitality", 1000000, 0.55, 0.06, False, 50505,  20000, 600000, "STRONG PURSUE"),
    Sponsor("Belmond (LVMH)",   "Hospitality / Hotels", "hospitality", 500000,  0.50, 0.06, False, 27778,  15000, 300000, "STRONG PURSUE"),
    Sponsor("Rosewood Hotels",  "Hospitality / Hotels", "hospitality", 500000,  0.40, 0.10, True,  20833,  10000, 225000, "STRONG PURSUE"),
    Sponsor("Four Seasons",     "Hospitality / Hotels", "hospitality", 750000,  0.45, 0.06, False, 46296,  15000, 300000, "PURSUE"),
    Sponsor("Auberge Resorts",  "Hospitality / Hotels", "hospitality", 750000,  0.55, 0.06, False, 37879,  12000, 225000, "PURSUE"),
    # ── Apparel / Fashion ────────────────────────────────────────────────────
    Sponsor("Brunello Cucinelli","Apparel / Fashion","apparel", 500000,  0.60, 0.10, True,  13889,  28000, 120000, "PURSUE"),
    Sponsor("Loro Piana",       "Apparel / Fashion","apparel", 450000,  0.55, 0.06, False, 22727,  25000, 95000,  "PURSUE"),
    Sponsor("Louis Vuitton",    "Apparel / Fashion","apparel", 350000,  0.40, 0.06, False, 24306,  22000, 75000,  "PURSUE"),
    Sponsor("Hugo Boss",        "Apparel / Fashion","apparel", 700000,  0.50, 0.06, False, 38889,  42000, 85000,  "CONDITIONAL"),
    Sponsor("Adidas",           "Apparel / Fashion","apparel", 300000,  0.15, 0.03, False, 111111, 58,    850,    "DO NOT PURSUE"),
    # ── Wealth & Other ───────────────────────────────────────────────────────
    Sponsor("Wealth Mgmt. (illustrative)","Wealth & Other","wealth",750000,0.55,0.06,False,37879,125000,1500000,"STRONG PURSUE"),
    Sponsor("DuPont",           "Wealth & Other","wealth", 900000,  0.80, 0.045,True, 41667,  35000, 425000, "STRONG PURSUE"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR CURVES — log-linear bell curve anchors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorCurve:
    id: str
    name: str
    anchors: list[tuple[float, float]]   # (ratio, score)
    gold: tuple[str, float]              # (name, ratio)
    median: tuple[str, float]
    worst: tuple[str, float]
    note: str
    overlap_range: str
    default_overlap: float


SECTOR_CURVES: dict[str, SectorCurve] = {
    "aviation": SectorCurve(
        id="aviation", name="Private Aviation",
        anchors=[(0,0),(3,8),(10,20),(14.7,35),(30,55),(76.5,76),(240,100)],
        gold=("NetJets (BRK)", 240), median=("ExoJet", 30), worst=("Wheels Up", 14.7),
        note="Aviation runs hot — worst deal is still 14.7x. Gold: NetJets BRK at 240x.",
        overlap_range="60–85%", default_overlap=0.65,
    ),
    "lux-auto": SectorCurve(
        id="lux-auto", name="Luxury Auto",
        anchors=[(0,0),(0.9,8),(2.7,48),(3.6,62),(5.7,78),(10,87),(84,100)],
        gold=("Ferrari:VistaJet model", 84), median=("McLaren", 2.7), worst=("Rivian", 0.9),
        note="Luxury auto runs 0.9–5.7x. Aspirational ceiling: Ferrari:VistaJet model at 84x.",
        overlap_range="30–65%", default_overlap=0.55,
    ),
    "watches": SectorCurve(
        id="watches", name="Watches / Jewelry",
        anchors=[(0,0),(0.4,5),(1.0,15),(1.2,22),(2.3,36),(5.0,55),(10,65),(19.4,82),(61.2,100)],
        gold=("Chopard", 61.2), median=("Tudor", 1.2), worst=("Breitling", 0.4),
        note="Watches span 0.4–61.2x. Digital CPM brands (TAG, Breitling, Rolex) score low — expected.",
        overlap_range="45–82%", default_overlap=0.60,
    ),
    "hospitality": SectorCurve(
        id="hospitality", name="Hospitality",
        anchors=[(0,0),(3,14),(5.9,28),(10,46),(10.8,52),(18,72),(28.8,100)],
        gold=("Exclusive Resorts", 28.8), median=("Belmond (LVMH)", 10.8), worst=("Auberge Resorts", 5.9),
        note="Hospitality runs 5.9–28.8x. Every example clears the 3x Pursue floor.",
        overlap_range="40–55%", default_overlap=0.50,
    ),
    "apparel": SectorCurve(
        id="apparel", name="Apparel / Fashion",
        anchors=[(0,0),(1,10),(2.2,28),(3.1,48),(4.2,62),(8.6,82),(20,100)],
        gold=("Brunello Cucinelli", 8.6), median=("Louis Vuitton", 3.1), worst=("Adidas", 0.0),
        note="Apparel 0–8.6x. Adidas is a documented outlier — mass-market digital CAC.",
        overlap_range="15–60%", default_overlap=0.50,
    ),
    "wealth": SectorCurve(
        id="wealth", name="Wealth Management",
        anchors=[(0,0),(3,15),(10,38),(10.2,40),(25,68),(39.6,85),(100,100)],
        gold=("Wealth Mgmt.", 39.6), median=("DuPont / Wealth", 25), worst=("DuPont", 10.2),
        note="Two data points. Score uses absolute thresholds as primary guide.",
        overlap_range="50–80%", default_overlap=0.55,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ASSUMPTIONS (defaults mirror the HTML tool)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Assumptions:
    members: int = 600
    high_conv: float = 0.10   # > 60% overlap
    mid_conv:  float = 0.06   # 40–60% overlap
    low_conv:  float = 0.03   # < 40% overlap

    def conv_for_overlap(self, overlap: float) -> float:
        if overlap > 0.60: return self.high_conv
        if overlap >= 0.40: return self.mid_conv
        return self.low_conv


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def bell_score(ratio: float, sector_id: str) -> float:
    """Log-linear interpolation across sector bell curve anchors → score 0–100."""
    curve = SECTOR_CURVES.get(sector_id)
    if not curve or ratio <= 0:
        return 0.0
    anchors = curve.anchors
    if ratio >= anchors[-1][0]:
        return 100.0
    for i in range(len(anchors) - 1):
        lo_r, lo_s = anchors[i]
        hi_r, hi_s = anchors[i + 1]
        if lo_r <= ratio <= hi_r:
            log_r  = math.log(max(ratio, 0.01))
            log_lo = math.log(max(lo_r, 0.01))
            log_hi = math.log(max(hi_r, 0.01))
            t = (log_r - log_lo) / (log_hi - log_lo) if log_hi > log_lo else 0
            return lo_s + t * (hi_s - lo_s)
    return 0.0


def decision_tier(ratio: float) -> str:
    """Absolute CLV:CAC ratio → decision tier label."""
    if ratio >= 10: return "STRONG PURSUE"
    if ratio >= 3:  return "PURSUE"
    if ratio >= 1:  return "CONDITIONAL"
    return "DO NOT PURSUE"


@dataclass
class EvalResult:
    sector: str
    deal_value: float
    overlap: float
    conv_rate: float
    members: int
    est_customers: float
    p1_cac: float
    event_benchmark_cac: float
    clv: float
    clv_to_cac: float
    sector_score: float
    decision: str
    cac_advantage: float


def evaluate(
    deal_value: float,
    overlap: float,
    sector_id: str,
    event_benchmark_cac: float,
    clv: float,
    assumptions: Optional[Assumptions] = None,
) -> EvalResult:
    """Run the full CLV:CAC model for a proposed deal."""
    a = assumptions or Assumptions()
    curve = SECTOR_CURVES[sector_id]
    conv = a.conv_for_overlap(overlap)
    est_customers = a.members * overlap * conv
    p1_cac = deal_value / est_customers if est_customers > 0 else float("inf")
    ratio  = clv / p1_cac if p1_cac > 0 else 0.0
    score  = bell_score(ratio, sector_id)
    return EvalResult(
        sector=sector_id,
        deal_value=deal_value,
        overlap=overlap,
        conv_rate=conv,
        members=a.members,
        est_customers=est_customers,
        p1_cac=p1_cac,
        event_benchmark_cac=event_benchmark_cac,
        clv=clv,
        clv_to_cac=ratio,
        sector_score=score,
        decision=decision_tier(ratio),
        cac_advantage=event_benchmark_cac - p1_cac,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

TIER_SYMBOLS = {
    "STRONG PURSUE": "★★",
    "PURSUE":        "★",
    "CONDITIONAL":   "◑",
    "DO NOT PURSUE": "✗",
}

def _fmt(n: float) -> str:
    if abs(n) >= 1_000_000: return f"${n/1e6:.2f}M"
    if abs(n) >= 1_000:     return f"${n/1e3:.0f}K"
    return f"${n:,.0f}"


def print_sector_report(sector_id: str) -> None:
    curve = SECTOR_CURVES[sector_id]
    sponsors = [s for s in TRAINING if s.sector == sector_id]
    sponsors_sorted = sorted(sponsors, key=lambda s: s.clv_to_cac, reverse=True)

    print(f"\n{'─'*70}")
    print(f"  {curve.name.upper()}")
    print(f"  Overlap range: {curve.overlap_range}  |  Gold standard: {curve.gold[0]} {curve.gold[1]:.0f}x")
    print(f"{'─'*70}")
    print(f"  {'Sponsor':<28} {'Deal':>9}  {'Overlap':>7}  {'CLV:CAC':>7}  {'Score':>5}  {'Decision'}")
    print(f"  {'─'*27}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*20}")
    for s in sponsors_sorted:
        ratio = s.clv_to_cac
        score = bell_score(ratio, sector_id)
        sym   = TIER_SYMBOLS.get(s.decision, " ")
        print(f"  {s.name:<28} {_fmt(s.deal):>9}  {s.overlap*100:>6.0f}%  {ratio:>6.1f}x  {score:>4.0f}  {sym} {s.decision}")
    print(f"  {curve.note}")


def print_all_sectors() -> None:
    print("\nP1 MOTOR CLUB — PARTNERSHIP EVALUATOR")
    print("Model v5 · 41 verified sponsors · Decision tiers: ≥10x SP | ≥3x P | ≥1x C | <1x DNP")
    for sid in SECTOR_CURVES:
        print_sector_report(sid)
    print()


def print_eval(result: EvalResult, sponsor_name: str = "Proposed Deal") -> None:
    sym = TIER_SYMBOLS.get(result.decision, " ")
    curve = SECTOR_CURVES[result.sector]
    print(f"\n{'═'*50}")
    print(f"  {sponsor_name.upper()} — {curve.name.upper()}")
    print(f"{'═'*50}")
    print(f"  Deal Value         {_fmt(result.deal_value):>14}")
    print(f"  Audience Overlap   {result.overlap*100:>13.0f}%")
    print(f"  Conv. Rate         {result.conv_rate*100:>13.1f}%")
    print(f"  Est. Customers     {result.est_customers:>14.1f}")
    print(f"  P1 CAC             {_fmt(result.p1_cac):>14}")
    print(f"  Benchmark CAC      {_fmt(result.event_benchmark_cac):>14}")
    print(f"  CAC Advantage      {_fmt(result.cac_advantage):>14}  {'✓' if result.cac_advantage >= 0 else '✗'}")
    print(f"  CLV                {_fmt(result.clv):>14}")
    print(f"  CLV:CAC Ratio      {result.clv_to_cac:>13.1f}x")
    print(f"  Sector Score       {result.sector_score:>13.0f}/100")
    print(f"  Decision           {sym} {result.decision:>12}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="P1 Motor Club — Partnership Evaluator")
    parser.add_argument("--sector",   help="Sector id: lux-auto | aviation | watches | apparel | hospitality | wealth")
    parser.add_argument("--deal",     type=float, help="Deal value ($)")
    parser.add_argument("--overlap",  type=float, help="Audience overlap (0–1, e.g. 0.65)")
    parser.add_argument("--clv",      type=float, help="Sponsor CLV ($)")
    parser.add_argument("--cac",      type=float, help="Sponsor event benchmark CAC ($)")
    parser.add_argument("--members",  type=int,   default=600, help="P1 active members (default 600)")
    parser.add_argument("--name",     default="Proposed Deal", help="Deal name label")
    args = parser.parse_args()

    if args.deal and args.sector:
        # Evaluate a specific deal
        curve = SECTOR_CURVES.get(args.sector)
        if not curve:
            print(f"Unknown sector '{args.sector}'. Choose from: {', '.join(SECTOR_CURVES)}")
            return
        overlap = args.overlap or curve.default_overlap
        sponsors_in_sector = [s for s in TRAINING if s.sector == args.sector]
        default_cac = sum(s.event_benchmark_cac for s in sponsors_in_sector) / max(len(sponsors_in_sector), 1)
        default_clv = sum(s.clv for s in sponsors_in_sector) / max(len(sponsors_in_sector), 1)
        result = evaluate(
            deal_value=args.deal,
            overlap=overlap,
            sector_id=args.sector,
            event_benchmark_cac=args.cac or default_cac,
            clv=args.clv or default_clv,
            assumptions=Assumptions(members=args.members),
        )
        print_eval(result, args.name)
        if args.sector:
            print_sector_report(args.sector)
    elif args.sector:
        print_sector_report(args.sector)
    else:
        print_all_sectors()


if __name__ == "__main__":
    main()
