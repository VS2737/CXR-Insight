# CXR Insight

CXR Insight is a B.Tech Project Life Cycle Management project at ATLAS SkillTech University. It generates a draft radiology impression from a chest X-ray and shows a reviewer where the draft may conflict with image evidence. The prototype is for educational research. Every generated draft requires qualified human review before anyone can draw clinical conclusions.

## How the prototype works

1. Upload a de-identified chest X-ray and select its projection.
2. An image encoder extracts visual features. GPT-2 uses those features to generate report text.
3. The interface displays the draft, the model version, three safety checks, and the reasons for review.

The screenshot demonstration uses the **EfficientNet-B0 + GPT-2 baseline**, which generates only the Impression section. Later experiments replaced the generator’s EfficientNet-B0 image encoder with **RAD-DINO**, pretrained on chest X-rays. Version 9 added a training objective for individual findings.

## Three safety checks

| Check | What the reviewer learns |
|---|---|
| Image–text agreement | A separate X-ray classifier flags possible disagreements or omissions in the generated text. Its scores are research signals. |
| Self-consistency | The system compares findings across five generated drafts. High agreement measures repeatability. All five drafts can repeat the same error. |
| Grad-CAM grounding | For supported positive findings, the system checks whether the classifier’s heatmap falls in an expected image region. When the draft names no positive finding to locate, grounding can return **unknown**. |

In the baseline demonstration, the generated impression read “No acute cardiopulmonary findings.” The interface showed **99% self-consistency**, **failed image–text agreement**, and **unknown Grad-CAM grounding**. The disagreement triggered a review flag. The example illustrates why consistency alone cannot establish accuracy.

## Dataset and evaluation

The project used 3,660 IU Chest X-Ray studies, split by study into 2,928 for training, 366 for validation, and 366 for testing. Patient-level independence could not be verified. The team inspected some test examples during development, which limits conclusions about performance on new data.

| Experiment | BLEU-4 | ROUGE-L F1 | Mean positive-finding mention recall |
|---|---:|---:|---:|
| v8 · RAD-DINO + GPT-2 | 0.1174 | 0.3094 | 1.48% |
| v9 · RAD-DINO + GPT-2 with finding supervision | 0.1349 | 0.3220 | 2.21% |

BLEU-4 and ROUGE-L measure text overlap with reference reports during offline evaluation. The live safety checks operate separately. Mention recall measures how many positive findings in a reference report also appear in the draft, averaged over 128 studies with positive reference mentions. Both versions missed at least one reference finding in 127 of those studies.

## Repository layout

| Directory | Contents |
|---|---|
| `app/` | FastAPI application and web interface |
| `radreport/` | Python package for model and reporting logic |
| `scripts/` | Data preparation, training, and evaluation scripts |
| `tests/` | Automated project tests |

## Running the local prototype

Use Python 3.11 and PowerShell from the project root. For a fresh checkout, the matching dependency files and trained checkpoints are prerequisites. The baseline demo uses the local generator and classifier checkpoints at `models/generator.pt` and `models/classifier.pt`. A RAD-DINO run requires its matching version-specific dependencies, assets, and checkpoint.

With the project's requirements file available, create an environment and install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

With the model assets in place, start the application:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Check the displayed model version to see which checkpoint generated the draft.

To run the automated tests in an environment with the project’s test dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Team

Gunn Mulchandani · Siddhant Amin · Vaishnavi Singh · Durva Chhabria
