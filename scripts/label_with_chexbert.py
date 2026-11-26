"""Adapt f1chexbert 0.0.2 report labels for CXR Insight's existing CSV importer.

Run from Radiology_Report_AI:
    python -m scripts.label_with_chexbert --limit 5 --output data/processed/labeled_reports_sample.csv
    python -m scripts.label_with_chexbert

INPUT:  headerless single-column CSV from python -m scripts.export_chexpert
OUTPUT: canonical 14 finding columns in the same report order.

Unlike F1CheXbert.get_label()'s default binary evaluation mode, this reads the
14 classification heads to distinguish absent, negative, positive and uncertain.
Do NOT equate CheXbert labels with independently adjudicated image findings.
"""
import argparse
import csv
from pathlib import Path

from radreport.config import FINDINGS

# CheXbert's 14 classification heads use this order (NOT project FINDINGS order).
HEAD_ORDER = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding",
]
# Stanford class convention for first 13 heads:
# 0 = unmentioned (empty); 1 = positive; 2 = negative; 3 = uncertain.
CLASS_TO_CSV = {0: "", 1: "1.0", 2: "0.0", 3: "-1.0"}


def decode_head_classes(predicted):
    if len(predicted) != len(HEAD_ORDER):
        raise ValueError(f"Expected 14 CheXbert heads, got {len(predicted)}")
    if any(p not in (0, 1, 2, 3) for p in predicted[:-1]):
        raise ValueError("Unexpected four-class head values from CheXbert")
    if predicted[-1] not in (0, 1):
        raise ValueError("Unexpected No Finding head value from CheXbert")
    values = {name: CLASS_TO_CSV[p] for name, p in zip(HEAD_ORDER[:-1], predicted[:-1])}
    values["No Finding"] = "1.0" if predicted[-1] == 1 else "0.0"
    return {name: values[name] for name in FINDINGS}


def predicted_head_classes(labeler, report, torch, device):
    """Use raw classification heads; do not collapse uncertainty/blank into binary."""
    encoded = labeler.tokenizer(
        report, truncation=True, max_length=512, return_tensors="pt"
    )
    ids = encoded["input_ids"].to(device)
    with torch.inference_mode():
        if hasattr(labeler.model, "cls_logits"):
            logits = labeler.model.cls_logits(ids)
        else:
            # Older f1chexbert releases expose the heads through model.forward.
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(ids)
            logits = labeler.model(ids, attention_mask.to(device))
    if not isinstance(logits, (list, tuple)) or len(logits) != 14:
        raise RuntimeError("Incompatible f1chexbert head API; no labels were invented. Send the traceback.")
    classes = []
    for i, head in enumerate(logits):
        expected = 2 if i == 13 else 4
        if head.ndim != 2 or head.shape[0] != 1 or head.shape[1] != expected:
            raise RuntimeError(f"Unexpected shape for CheXbert head {i}: {tuple(head.shape)}")
        classes.append(int(head.argmax(dim=1).item()))
    return classes


def label_file(input_path, output_path, limit, device):
    import torch
    from f1chexbert import F1CheXbert

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output CSV paths must differ.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}; run python -m scripts.export_chexpert first")
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows or any(len(row) != 1 for row in rows):
        raise ValueError("Expected nonempty headerless single-column CSV created by export_chexpert.")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No reports to label")

    print(f"Loading CheXbert on {device}; {len(rows)} report(s) to label.", flush=True)
    labeler = F1CheXbert(device=device)
    labeler.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to .part first, so failed labeling never appears as a complete output.
    tmp = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FINDINGS)
            writer.writeheader()
            for i, row in enumerate(rows, 1):
                classes = predicted_head_classes(labeler, row[0] or " ", torch, device)
                writer.writerow(decode_head_classes(classes))
                if i == 1 or i % 100 == 0 or i == len(rows):
                    print(f"Labeled {i}/{len(rows)}", flush=True)
        tmp.replace(output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Saved {len(rows)} aligned CheXbert label rows to {output_path}")
    print("Research warning: text-derived weak labels, not radiologist-verified image ground truth.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/chexpert_input.csv")
    p.add_argument("--output", default="data/processed/labeled_reports.csv")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    label_file(Path(args.input), Path(args.output), args.limit, args.device)


if __name__ == "__main__":
    main()
