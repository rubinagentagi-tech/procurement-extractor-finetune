#!/usr/bin/env python3
"""
Synthetic procurement-extraction dataset generator for the fine-tune trial.

Task taught to the model: read a supplier quotation (plain text) and emit
the structured JSON: supplier, quote reference, date, line items and any
anomalies (math errors, missing unit prices). Every entity is fictional.

Output: data/train.jsonl + data/val.jsonl (alpaca: instruction/input/output).
Deterministic: seed fixed at 42.
"""
import json, random, datetime

random.seed(42)
rng = random.Random(42)

SUPPLIERS = ["Harborlight Marine Supply Inc.", "Nordmarin Industrial Inc.",
             "PrairieWorks Industrial", "Cascade Freight Parts", "Summitline Trading Co.",
             "Bluewater Industrial Supply", "Keystone Machinery Ltd.", "Ridgeline Components"]
CITIES = ["St. John's, NL", "Montreal, QC", "Calgary, AB", "Vancouver, BC", "Winnipeg, MB", "Halifax, NS"]
PARTS = [  # (part_no, desc, unit_price_cad)
    ("BFP-1400", "Bilge pump strainer, bronze", 148.50),
    ("HHA-3/8-2500", "Hydraulic hose assembly, 3/8 in x 2500 psi", 38.40),
    ("FLT-185", "Oil filter, spin-on, 185 mm", 12.75),
    ("VLB-44", "V-belt, 44 in, wrapped", 9.90),
    ("SKF-3B", "Shaft seal kit, 3 in bore", 210.00),
    ("GNZ-25", "Grease fitting, zinc, 1/4 NPT", 0.85),
    ("SBC-110", "Starter battery cable, 110 cm", 27.30),
    ("PRV-300", "Pressure relief valve, 300 psi", 186.20),
    ("GKT-022", "Gasket sheet, 1.5 mm, 1 m x 1 m", 64.00),
    ("LMP-24V", "LED lamp, 24 V, 10 W, marine", 33.60),
    ("IML-300", "Impeller, neoprene, 300 series pump", 54.25),
    ("THR-010", "Thermostat, 82 deg C", 22.10),
    ("CBL-5C", "Control cable, 5 m, push-pull", 96.40),
    ("FSC-22", "Fuel sender cap, 22 in", 44.90),
    ("BRG-6205", "Ball bearing, 6205-2RS", 11.35),
    ("HSS-8", "Hose clamp, stainless, 8 in", 3.20),
]
ADJ = ["DELIVERY: 2-3 weeks.", "SHIPPING: FOB origin.", "WARRANTY: 12 months.",
       "PAYMENT: net 30.", "PRICES IN CAD. Taxes extra.", "MINIMUM ORDER: none.",
       "STOCK ITEMS SHIP WITHIN 5 BUSINESS DAYS.", "QUOTE VALID 30 DAYS."]

def gen_quote(idx):
    sup = rng.choice(SUPPLIERS)
    city = rng.choice(CITIES)
    ref = f"Q-{rng.randint(10000, 99999)}"
    date = (datetime.date(2026, 9, 1) - datetime.timedelta(days=rng.randint(1, 90))).isoformat()
    n = rng.randint(3, 7)
    chosen = rng.sample(PARTS, n)
    rows = []          # (part, desc, qty, unit, line_total, note)
    anomaly = None
    for i, (pn, desc, up) in enumerate(chosen):
        qty = rng.choice([1, 2, 3, 4, 6, 8, 10, 12])
        tot = round(qty * up, 2)
        note = None
        # plant deliberate anomalies (flag them in the label, not silently wrong math)
        if anomaly is None and i == 2 and rng.random() < 0.5:
            tot = round(tot - rng.choice([0.10, 0.25, 1.00]), 2)   # math mismatch
            anomaly = "line_total_mismatch"
        elif anomaly is None and i == 3 and rng.random() < 0.4:
            up = None                                                # missing unit price
            anomaly = "missing_unit_price"
        rows.append((pn, desc, qty, up, tot, note))
    lines = []
    lines.append(f"{sup}")
    lines.append(f"{rng.choice(['ATTN: QUOTING DESK', 'ATTN: SALES DEPT', 'ATTN: PARTS DEPT'])}")
    lines.append(f"QUOTATION {ref}          DATE: {date}")
    lines.append(f"To: {'Meridian Freight Group'}")
    lines.append("-" * 58)
    lines.append("ITEM  PART NO      QTY  UOM  DESCRIPTION          UNIT PRICE  LINE TOTAL")
    for k, (pn, desc, qty, up, tot, _) in enumerate(rows, 1):
        up_s = f"{up:,.2f}" if up is not None else "  ---  "
        lines.append(f"{k:<5}{pn:<14}{qty:<5}EA   {desc[:22]:<22}${up_s:>8}  ${tot:>9,.2f}")
    lines.append("-" * 58)
    sub = round(sum(r[4] for r in rows), 2)
    lines.append(f"{'SUBTOTAL':<50}${sub:>9,.2f}")
    ship = round(rng.uniform(25, 150), 2)
    lines.append(f"{'SHIPPING & HANDLING':<50}${ship:>9,.2f}")
    lines.append(f"{'TOTAL':<50}${round(sub + ship, 2):>9,.2f}")
    lines.append(rng.choice(ADJ))
    lines.append(f"Questions: {rng.choice(['quotes@', 'sales@', 'parts@'])}{rng.choice(['nordmarin', 'harborlight', 'prairieworks', 'summitline']) + '.example.com'}")
    text = "\n".join(lines)

    items = [{"part_no": pn, "description": desc, "qty": qty,
              "unit_price": up, "line_total": round(qty * up, 2) if up else None}
             for pn, desc, qty, up, _t, _n in rows]
    out = {"supplier": sup, "quote_no": ref, "date": date, "currency": "CAD",
           "items": items, "subtotal": sub,
           "flags": [anomaly] if anomaly else []}
    return text, out

def make(split):
    rows = []
    for i in range(split):
        text, out = gen_quote(i)
        rows.append({"instruction": "Extract this quotation into JSON: supplier, quote_no, date, "
                                    "items (part_no, description, qty, unit_price, line_total), subtotal, "
                                    "and flags for anomalies (line_total_mismatch, missing_unit_price). "
                                    "Use only what is in the document. Unit prices may be absent. "
                                    "Output ONLY valid JSON, nothing else.",
                     "input": text,
                     "output": json.dumps(out)})
    return rows

train = make(2000)
val = make(300)
# deterministic shuffle once so val_split is not needed
rng.shuffle(train); rng.shuffle(val)
import pathlib
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/train.jsonl", "w") as f:
    for r in train: f.write(json.dumps(r) + "\n")
with open("data/val.jsonl", "w") as f:
    for r in val: f.write(json.dumps(r) + "\n")
print(f"train: {len(train)}  val: {len(val)}")
print("sample input head:")
print(train[0]["input"].splitlines()[0:4])
print("sample output:", train[0]["output"][:200])
