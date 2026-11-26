# CXR Insight v9 — Finding-Aware Fine-Tuning & Paired Test Evaluation

**Educational research prototype only. NOT for clinical use.** No generated report is a diagnosis; all outputs require qualified review. The existing IU dataset, fixed train/validation/test splits, and trained v8 checkpoint are retained. This update **does not contain newly trained weights or verified accuracy gains**.

## What changes

- Continue the trained **RAD-DINO → GPT-2** model from `models_v8/generator.pt` instead of restarting from random cross-attention weights.
- Add a multi-label weak finding-prediction head that shares the visual-to-text feature bridge with the decoder. Its loss is combined with the ordinary **Findings + Impression** token loss.
- Use your existing 13 CheXbert pathology/device categories (excluding "No Finding") and **two additional strictly text-derived observations**: explicit pulmonary-artery enlargement and prior sternotomy. Extra concepts are trained only when the *IU training split* contains enough positive and explicitly negative samples. Otherwise the corresponding head is not trained; they cannot be assumed detectable.
- Mask uncertain and missing labels. Do not infer normality from "no acute pulmonary findings." Use validation loss and never tune on the test set.
- Compare v8 and v9 predictions on **the identical frozen 366-study test split**: lexical BLEU-4, ROUGE-L, mention omissions, explicit false-negative statements, and paired bootstrap confidence intervals. All mention outcomes use report text; no independent radiologist verification is claimed.
- Support `CXR_V9_MODE=1` in the same app with the existing v8 report layout and mandatory-review routing. v9 auxiliary scores are shown only as *uncalibrated research signals*.

## 1. Install on Windows

In VS Code, **stop the Uvicorn server** with Ctrl+C. In your already working project `Radiology_Report_AI`, activate your existing `.venv` (shown as `(.venv)`). Download `CXR_Insight_v9_Finding_Aware_Patch.zip` and **move it into that exact folder**.

```powershell
# Verify that the downloaded file really exists before extracting:
Get-Item .\CXR_Insight_v9_Finding_Aware_Patch.zip

# Back up your code and current v8 checkpoint. Do not remove them.
New-Item -ItemType Directory -Force .\backup_before_v9 | Out-Null
Copy-Item .\app\service.py .\backup_before_v9\service.py
Copy-Item .\models_v8\metadata.json .\backup_before_v9\v8_metadata.json

Expand-Archive -LiteralPath .\CXR_Insight_v9_Finding_Aware_Patch.zip -DestinationPath . -Force
python -m pytest -q
```

The update uses the same `requirements-v8.txt` dependencies. If modules are missing, install those (not CPU PyTorch) with `python -m pip install -r requirements-v8.txt`. Check `python -c "import torch; print(torch.cuda.is_available())"` prints `True` on your RTX 4060.

## 2. Verify the problem case and split

```powershell
python -m scripts.v9_check_study --uid 2
```

This locates study 2 in the frozen `train`, `val`, or `test` split and prints its reference findings. **A training-set example cannot demonstrate generalization.** The report mentions borderline cardiomegaly, enlarged pulmonary arteries, and sternotomy; the generator omitted those features in your previous screenshot. Compare text, don't infer the image is definitively abnormal without expert review.

## 3. Run finding-aware continuation training

**The existing checkpoint is not overwritten.** Use:

```powershell
python -m scripts.v9_train_generator --init-v8 models_v8 --output models_v9 --epochs 3 --batch-size 1 --accumulation 8 --lr 0.00001 --aux-weight 0.25
```

The script checks split SHA-256 hashes, caches, and model provenance. It prints which finding categories have enough observed training support, and saves only the best **validation** loss checkpoint to:

```
models_v9\generator.pt
models_v9\metadata.json
```

If your GPU runs out of VRAM, close other GPU applications, keep batch size at 1, and avoid running the web server concurrently. More training is *not* guaranteed to help and may increase false-positive findings. Do **not** delete `models_v8`.

## 4. Paired test on the same studies

First, optional two-study smoke test:

```powershell
python -m scripts.v9_compare --limit 2
```

For the actual **complete 366-study test** do **not** use `--limit`:

```powershell
python -m scripts.v9_compare
Get-Content .\results_v9\comparison.json
Start-Process .\results_v9\paired_casebook.csv
```

Outputs:

- `results_v9/comparison.json`: v8 and v9 BLEU-4 / ROUGE-L, mention omissions, and paired bootstrap confidence interval for change in reference-positive mention recall.
- `results_v9/paired_casebook.csv`: matched reference text, v8 report and v9 report per study, missed and explicitly negated reference mentions.

**Only replace v8 with v9 if held-out results support a meaningful gain without unsafe new hallucinations.** CheXbert/lexical concept measures and ROUGE/BLEU are proxies, not clinical accuracy, and the IU split is study-level rather than verified patient-level.

## 5. Use trained v9 in your existing website (optional)

```powershell
$env:CXR_V9_MODE = '1'
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Your `model_version` should be `rad-dino-gpt2-v9-research`. Report output, entity checks, self-consistency, Grad-CAM abstentions and mandatory review remain in the UI. The auxiliary weak-label scores are included in API `safety.v9_auxiliary_weak_label_scores`; **do not interpret them as disease probabilities**.

To return to v8 instantly, stop Uvicorn and run:

```powershell
Remove-Item Env:CXR_V9_MODE -ErrorAction SilentlyContinue
$env:CXR_V8_MODE = '1'
python -m uvicorn app.main:app --reload
```

If v9 fails to load, it must show a failed-model status rather than fabricating a report. Keep v8 as your presentation fallback.

## Scientific limitations

- IU X-Ray is small; a finding-aware objective is a hypothesis, not a guarantee of improved clinical results.
- The 14-class CheXbert labels are derived from historical text, not independently confirmed X-ray diagnoses. Novel classes such as pulmonary artery enlargement and sternotomy are weak textual patterns and may lack adequate training support.
- "No acute findings" does not necessarily mean no chronic or postoperative abnormalities.
- Five-generation agreement and Grad-CAM are not accuracy certificates. A good safety system must abstain or demand review when results conflict.
- Genuine clinical sensitivity, false-negative rates for real disease, subgroup performance and generalization require independent expert annotation and appropriate external datasets. This package does not provide that evidence.
