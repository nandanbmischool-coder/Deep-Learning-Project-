# PneumoVision AI

## Explainable Pneumonia Detection from Chest X-Rays

PneumoVision AI is a full-stack deep-learning project for research and education. It combines a PyTorch DenseNet121 classifier, Grad-CAM explanations, a FastAPI backend, and a React/TypeScript dashboard for uploading X-rays, viewing predictions, inspecting model performance, and managing training runs.

> **Medical safety:** This project is not clinically validated and must not be used for diagnosis or medical decisions.

## Highlights

- Transfer learning with ImageNet-pretrained DenseNet121
- Binary classification: `NORMAL` vs `PNEUMONIA`
- Explainability with Grad-CAM heatmaps
- Validation-only threshold selection using Youden's J
- Leakage-aware, stratified group validation split
- Class-weighted loss for imbalanced data
- Two-phase training: classifier head, then final convolutional block
- FastAPI prediction and training APIs
- React dashboard with analysis, history, dataset, performance, and training views
- Model registry and evaluation reports

## Trained model

The included checkpoint was trained on the Kaggle **Chest X-Ray Images (Pneumonia)** dataset.

| Item | Value |
|---|---|
| Architecture | DenseNet121 |
| Pretrained weights | Torchvision ImageNet `IMAGENET1K_V1` |
| Input | RGB, `224 x 224` |
| Classes | `NORMAL`, `PNEUMONIA` |
| Total epochs | 20 |
| Head-only epochs | 5 |
| Fine-tuning epochs | 15 |
| Batch size | 32 |
| Optimizer | AdamW |
| Head learning rate | `0.001` |
| Fine-tuning learning rate | `0.0001` |
| Best checkpoint | Epoch 18 |
| Device | CPU |
| Parameters | 6,955,906 |

### Test results

Results are from 624 untouched test images:

| Metric | Result |
|---|---:|
| Accuracy | 85.74% |
| Precision | 81.55% |
| Recall / sensitivity | 99.74% |
| Specificity | 62.39% |
| F1 score | 89.73% |
| ROC-AUC | 97.01% |
| PR-AUC | 97.76% |

The model prioritizes pneumonia sensitivity. Its lower specificity means some normal images are classified as pneumonia.

## Preprocessing and deep-learning method

Images are resized to `224 x 224`, converted to RGB tensors, and normalized with ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training augmentation uses small affine transformations. Horizontal flipping is disabled by default because chest X-ray laterality can be meaningful.

Training uses supervised binary classification, weighted cross-entropy, cosine learning-rate scheduling, early stopping, and fine-tuning of DenseNet121's final feature block. Grad-CAM uses `features.denseblock4` to visualize image regions contributing to the selected prediction.

## Run locally on Windows

### 1. Backend

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -r requirements-ml.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open:

- Dashboard: <http://127.0.0.1:5173>
- API documentation: <http://127.0.0.1:8000/api/docs>

The backend automatically discovers completed runs under `data/artifacts/runs/` and registers the included model.

## Dataset

Download the dataset from Kaggle:

<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

Place the extracted files at:

```text
data/raw/chest_xray/
  train/NORMAL
  train/PNEUMONIA
  val/NORMAL
  val/PNEUMONIA
  test/NORMAL
  test/PNEUMONIA
```

Then validate:

```powershell
cd backend
python -m app.ml.dataset_validation
```

The raw dataset is intentionally not committed to GitHub because it is approximately 2.3 GB and exceeds practical repository limits. Use the Kaggle download instructions above to reproduce it.

## Train a new model

```powershell
cd backend
python -m app.ml.train `
  --arch densenet121 `
  --epochs 20 `
  --head-epochs 5 `
  --batch-size 32 `
  --patience 5 `
  --run-dir ..\data\artifacts\runs\densenet121-new `
  --version densenet121-new
```

## Important files

- `backend/app/ml/data.py` - dataset discovery, splitting, preprocessing, and augmentation
- `backend/app/ml/model.py` - model factory and transfer-learning configuration
- `backend/app/ml/train.py` - training and evaluation pipeline
- `backend/app/ml/gradcam.py` - Grad-CAM generation
- `frontend/src/pages/` - dashboard pages
- `data/artifacts/runs/densenet121-full/evaluation.json` - measured evaluation report
- `data/artifacts/runs/densenet121-full/evaluation.html` - formatted evaluation report
- `data/artifacts/runs/densenet121-full/model.pt` - trained checkpoint

## License and responsible use

Check the source dataset's Kaggle license and terms before redistribution. This application is intended for academic demonstration and research only.
