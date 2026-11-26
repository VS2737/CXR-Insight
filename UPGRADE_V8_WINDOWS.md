# CXR Insight v8 — RTX 4060 Windows Training & Evaluation

**Purpose:** Research-only improvement experiment using the **SAME IU X-Ray dataset** as v7. Nothing here guarantees detecting all disease or zero-error clinical reporting. Trained weights are *not in this ZIP*; they must be downloaded/learned on your machine.

## What's new

- **Chest-X-ray-pretrained vision:** Microsoft RAD-DINO (ViT-B/14) replaces ImageNet EfficientNet for the NEW experimental generator. Its released weights were pretrained on other public chest X-ray datasets. Your own training/evaluation dataset is unchanged: IU X-Ray. Source: https://huggingface.co/microsoft/rad-dino
- **GPU-efficient pipeline:** Precompute frozen RAD-DINO tokens once, then fine-tune GPT-2 cross-attention and last two language blocks on *actual* IU **Findings + Impression** text. Default RTX 4060 settings use batch size 1 and gradient accumulation 8.
- **Separate abnormality head:** Fit a 14-observation logistic probe from frozen RAD-DINO CLS features using your existing CheXbert-derived labels. It abstains for classes without enough train/validation evidence and preserves uncertain/unmentioned states. Scores are not diagnostic probabilities.
- **Training-only visual nearest-neighbor reference cases:** Retrieval never includes val/test studies; reference analogues are labeled as analogues, NOT diagnoses for the uploaded X-ray.
- **Conservative existing protections:** Your v4-v7 contradiction checks, five-generation finding-level agreement, DenseNet+Grad-CAM, mandatory review, and v6-style report remain. Nothing auto-approves a draft.
- **Matched validation/test experiments** for BLEU-4, ROUGE-L, contradiction proxy, possible omission proxy, and per-class weak-label F1. Train and validation are never merged into the test corpus.

## 1. BEFORE editing anything

Open VS Code at your EXISTING `Radiology_Report_AI` project folder (the one with `app`, `radreport`, `scripts`, `models`). Run commands in PowerShell there. Stop the Uvicorn server with Ctrl+C.

Back up your models and files:

```powershell
New-Item -ItemType Directory -Force backup_before_v8 | Out-Null
Copy-Item -Recurse app,radreport,scripts backup_before_v8
Copy-Item models\generator.pt backup_before_v8\ -ErrorAction SilentlyContinue
Copy-Item models\classifier.pt backup_before_v8\ -ErrorAction SilentlyContinue
```

Download `CXR_Insight_v8_Patch.zip` and move the ZIP into your existing `Radiology_Report_AI` folder. Run:

```powershell
Expand-Archive -LiteralPath .\CXR_Insight_v8_Patch.zip -DestinationPath . -Force
```

Activate the SAME `.venv` you used for v7. From your project root, locate it before activating:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__,torch.cuda.is_available())"
```

You should have Python 3.11, PyTorch with CUDA=True and your RTX 4060. Install other requirements if needed:

```powershell
python -m pip install -r requirements-v8.txt
python -m pytest -q
python -m scripts.v8_preflight
```

If you see `CUDA=False`, fix GPU PyTorch first, as in your earlier v7 setup. Do not proceed with full GPU training on CPU accidentally.

## 2. Cache RAD-DINO features (first download requires internet)

```powershell
python -m scripts.v8_cache_rad_dino --split all
```

This extracts 7×7 spatial tokens and a CLS embedding per **frontal** IU image into `data/v8_rad_features/`. It is resumable: successfully cached studies are skipped on rerun. Does not download or add new IU studies. Memory requirements are reduced by freezing RAD-DINO.

## 3. Train your new full Findings + Impression report generator

```powershell
python -m scripts.v8_train_generator --epochs 8 --batch-size 1 --accumulation 8 --lr 0.00003 --patience 3
```

Best checkpoint: `models_v8/generator.pt` (your `models/generator.pt` baseline is never overwritten). Training selects the checkpoint on validation loss only; it has NOT been evaluated on the held-out test set yet. If GPU memory is insufficient, do not increase batch size; close other GPU programs and check the exact CUDA error.

## 4. Train the additional anomaly probe and study retriever

Assumes the previously generated `data/processed/labels.csv` exists and contains CheXbert-derived weak labels:

```powershell
python -m scripts.v8_train_probe
python -m scripts.v8_build_retrieval
```

Outputs: `models_v8/finding_probe.json` and `models_v8/retrieval/`.

Class-specific unknown/unmentioned results are retained. The probe only supports the existing 14 CheXpert categories, not every imaginable anomaly.

## 5. Compare OLD and NEW on VALIDATION first

```powershell
python -m scripts.evaluate --input data/processed/val.jsonl --output results_v8/baseline_val --seed 42
python -m scripts.v8_evaluate --input data/processed/val.jsonl --output results_v8/candidate_val --seed 42
python -m scripts.compare_generator_runs --baseline results_v8/baseline_val/predictions.csv --candidate results_v8/candidate_val/predictions.csv --output results_v8/paired_val.json
```

Compare BLEU-4, ROUGE-L, omissions and contradictions. **Do not promote v8 just because some examples look nicer.** If v8 is worse, stay with v7.

## 6. Run a FINAL full-test comparison only after selecting the model

```powershell
python -m scripts.evaluate --input data/processed/test.jsonl --output results_v8/baseline_test --seed 42
python -m scripts.v8_evaluate --input data/processed/test.jsonl --output results_v8/candidate_test --seed 42
python -m scripts.compare_generator_runs --baseline results_v8/baseline_test/predictions.csv --candidate results_v8/candidate_test/predictions.csv --output results_v8/paired_test.json
python -m scripts.v8_evaluate_probe
```

Read JSON/CSV files in `results_v8/`. Report true scores without rounding them into a clinical "accuracy" percentage. Remember the IU CSV supports only **study-level** grouping; not a proven patient-level split. The weak-label probe's F1 is NOT clinical detection accuracy.

## 7. Enable new generator on your SAME website

Stop the server, then in the project terminal:

```powershell
$env:CXR_V8_MODE = '1'
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and upload a *frontal IU X-ray*; choose frontal projection. The same existing report layout displays model-produced **Findings** and **Impression** when the new checkpoint does produce both. The API's `v8_retrieved_training_analogues` are separately labeled and must not be confused with a diagnosis.

To return to your original, proven-to-launch v7 application:

```powershell
# Stop server first, then:
Remove-Item Env:CXR_V8_MODE -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --reload
```

You can alternatively use the bundled PowerShell helper:

```powershell
.\RUN_V8_WINDOWS.ps1 -Mode Preflight
.\RUN_V8_WINDOWS.ps1 -Mode Train
.\RUN_V8_WINDOWS.ps1 -Mode Evaluate
.\RUN_V8_WINDOWS.ps1 -Mode Serve
```

If PowerShell blocks scripts, use the direct Python commands above.

## Known limits / honest presentation notes

1. **Cannot detect every abnormality.** Even large models miss findings. The head covers 14 known categories, and rare conditions may not have enough positive examples to learn at all.
2. **No trained weights supplied.** Source code and offline tests cannot establish an improvement over your original BLEU-4=0.0549. You must train it and compare held-out results.
3. **No validated clinical safety claims.** MeSH tags and observations are derived from generated text; Grad-CAM does not prove finding localization. Review is always required.
4. **Pretraining used external image data.** Your project's *fine-tuning dataset* is still IU X-Ray. For strict data-origin policies, disclose RAD-DINO's external pretraining and ask your instructor whether it is allowed (it is analogous to ImageNet pretraining in v7).
5. **8 GB VRAM limit.** Precompute RAD-DINO features and train the decoder separately. We do not claim larger 7B medical multimodal models can be fully trained locally on this laptop.
6. **Reproducibility.** The script verifies original train/val/test file SHA256 values and refuses inconsistent caches/checkpoints. GPU algorithm nondeterminism can remain.

## Evidence you can present

- Baseline EfficientNet-B0+GPT-2 report and its prior held-out scores.
- New RAD-DINO encoder and frozen-feature pipeline architecture.
- New *Findings* + *Impression* output and its honest safety checks.
- Validation and test comparison table **only after scripts finish successfully**.
- Limitation statement: "research-only AI drafts require qualified radiologist review".
