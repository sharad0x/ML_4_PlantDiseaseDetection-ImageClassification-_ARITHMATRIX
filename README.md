# Plant Disease Classification (PyTorch + Streamlit)

A deep learning project to classify plant leaf diseases using **EfficientNet-B3**, fine-tuned on the [PlantDoc dataset](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset).  
The project provides scripts for training, evaluation, batch inference, and an interactive web app for single image prediction.  

---

## Dataset
We use the **PlantDoc dataset**:  
👉 [Download here](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)  

After download, extract it and arrange like this: 
``` 
data/
├── train/
├── test/
└── inference/ # (for your own test images)
```

---

## Repo Structure
```
repo/
 ├── app.py
 ├── src/
 |    ├── train.py
 |    ├── evaluate.py
 |    └── inference.py
 ├── saved_models/
 |    └── plant_disease_efficientnet_b3.pth
 ├── requirements.txt
 ├── README.md
 ├── results/
 │    ├── inference/
 |    |     ├── inference_grid.png
 |    ├── plots/
 |    |     ├── accuracy_curve.png
 │    |     └── loss_curve.png
 |    ├── classification_report.txt
 |    └── confusion_matrix.png
 ├── demo/
 │    ├── demo1.png
 │    └── demo2.png
 └── data/  
      ├── train/    (excluded from repo)
      ├── test/     (excluded from repo)
      └── inference/
```

---

## Features

- **Fine-tuned EfficientNet-B3 (ImageNet)** — model is initialized from ImageNet weights and trained in two stages (classifier head, then full fine-tune) for robust leaf-disease features.
- **train.py** — GPU-enabled training pipeline (uses `data/train`, automatic 80/20 split for validation, two-stage training). Saves checkpoint to `saved_models/plant_disease_efficientnet_b3.pth` and training plots to `results/plots/loss_curve.png` and `results/plots/accuracy_curve.png`.
- **evaluate.py** — loads the saved model, evaluates on `data/test`, writes `results/classification_report.txt`, and saves `results/confusion_matrix.png`.
- **inference.py** — batch inference script that reads images from `data/inference/`, runs predictions and produces a single visualization grid saved at `results/inference/inference_grid.png`.
- **Streamlit app (`app.py`)** — web UI to upload a single image and see the uploaded image and prediction side-by-side (GPU enabled when available).
- **Reproducibility & utilities** — pinned `requirements.txt`, recommended `.gitignore`, and dataset download instructions (Kaggle CLI) included in the README.

---

## Setup

Clone repo and install dependencies:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

---

## Training

To train the model from scratch:
```
python src/train.py
```

This will:
- Train the CNN on data/train/
- Save the best model as model.pth
- Generate accuracy curves

---

## Evaluation

To evaluate the trained model:
```
python src/evaluate.py
```

This will generate:
- classification_report.txt (precision, recall, f1-score)
- accuracy_curve.png (accuracy over epochs)

---

## Inference (Batch Prediction)

Place your test images inside data/inference/ (e.g., cat.23.png, dog.82.png).

Run:
```
python src/inference.py
```

This will:
- Pick random 12 images from data/inference/
- Predict their labels
- Display and save results in a grid format as results/inference_results.png

---

## Demo

<p align="center">
  <img src="demo/demo1.png" alt="Demo Screenshot 1" width="45%"/>
  <img src="demo/demo2.png" alt="Demo Screenshot 2" width="45%"/>
</p>

---

## Streamlit App

Run interactive app:
```
streamlit run app.py
```

Features:
- Upload a single image
- Get real-time prediction with probability
- See image + predicted class side-by-side