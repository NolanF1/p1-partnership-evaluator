# P1 Motor Club — Partnership Evaluator

A CLV:CAC scoring model for evaluating sponsorship deals against P1's 600-member core, calibrated on 41 verified deals across six sectors.

## Sectors

| ID | Name | Overlap Range | Gold Standard |
|---|---|---|---|
| `lux-auto` | Luxury Auto | 30–65% | Ferrari:VistaJet model (84x) |
| `aviation` | Private Aviation | 60–85% | NetJets BRK (240x) |
| `watches` | Watches / Jewelry | 45–82% | Chopard (61.2x) |
| `apparel` | Apparel / Fashion | 15–60% | Brunello Cucinelli (8.6x) |
| `hospitality` | Hospitality | 40–55% | Exclusive Resorts (28.8x) |
| `wealth` | Wealth Management | 50–80% | Wealth Mgmt. (39.6x) |

## Decision Tiers

| CLV:CAC | Decision |
|---|---|
| ≥ 10x | **STRONG PURSUE** |
| ≥ 3x | **PURSUE** |
| ≥ 1x | **CONDITIONAL** |
| < 1x | **DO NOT PURSUE** |

## Model

```
P1 CAC       = Deal Value / (Members × Overlap × Conv Rate)
CLV:CAC      = Sponsor CLV / P1 CAC
Sector Score = log-linear percentile in sector bell curve (0–100)
```

Conv rate auto-assigned by overlap tier:
- `> 60%` overlap → 10% (Porsche PEC benchmark)
- `40–60%` overlap → 6% (Bain Luxury Study 2024)
- `< 40%` overlap → 3% (ANA/DMA benchmark)

## Usage

```bash
# Full report — all sectors
python evaluator.py

# Single sector
python evaluator.py --sector aviation

# Evaluate a specific deal
python evaluator.py \
  --sector aviation \
  --deal 1500000 \
  --overlap 0.75 \
  --clv 2250000 \
  --cac 500000 \
  --name "VistaJet 2025"
```

## Example Output

```
══════════════════════════════════════════════════
  VISTAJET 2025 — PRIVATE AVIATION
══════════════════════════════════════════════════
  Deal Value            $1.50M
  Audience Overlap         75%
  Conv. Rate              10.0%
  Est. Customers          45.0
  P1 CAC             $33,333
  Benchmark CAC     $500,000
  CAC Advantage     $466,667  ✓
  CLV                $2.25M
  CLV:CAC Ratio         67.5x
  Sector Score          97/100
  Decision          ★★ STRONG PURSUE
──────────────────────────────────────────────────
```

## File Structure

```
p1_partnership_evaluator/
├── evaluator.py      # full model: training data, scoring engine, CLI
└── README.md
```

## Dependencies

None. Standard library only (`math`, `argparse`, `dataclasses`). Python 3.8+.
