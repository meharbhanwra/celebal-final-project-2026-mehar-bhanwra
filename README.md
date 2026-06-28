# Satellite Image Land-Use Classifier & Temporal Change Detector

This project implements an end-to-end remote sensing pipeline for **satellite land-use classification** and **temporal change detection** using deep learning. The system is trained on the **EuroSAT** dataset, evaluated on both spatially separated and external datasets, and deployed through an interactive **Streamlit** application.

---

# Setup

## Clone the Repository

```bash
git clone https://github.com/meharbhanwra/celebal-final-project-2026-mehar-bhanwra.git
cd satellite-project
```

## Create a Virtual Environment

Using Conda:

```bash
conda create -n satellite python=3.12
conda activate satellite
```

## Install Dependencies

```bash
pip install -r requirements.txt
```
## Create Required Directory Structure

 ```text
│
├── app/
├── checkpoints/
├── data/
│ ├── eurosat/
│ ├── ucmerced/
│ └── processed/
├── notebooks/
├── outputs/
│ ├── change_detection/
│ ├── confusion matrices/
│ ├── error_analysis/
│ └── figures/
├── src/
│ ├── datasets/
│ ├── models/
│ ├── utils/
├── tests/
├── README.md
└── requirements.txt
```
Add EuroSAT and UC Merced Land Use Datasets to their respective folders. Run the notebooks in this exact order:
  ```text
  00_setup_check.ipynb
  01_dataset_eda.ipynb
  02_geospatial_check.ipynb
  02_spatial_metadata.ipynb
  03_spatial_split.ipynb
  04_create_splits.ipynb
  05_dataloaders.ipynb
  06_baseline_cnn.ipynb
  07_baseline_evaluation.ipynb
  08_resnet18_frozen.ipynb
  09_resnet18_finetuning_unfreezing.ipynb
  10_ucmerced_holdout_evaluation.ipynb
  11_random_split_experiment.ipynb
  12_spatial_leakage_analysis.ipynb
  13_temporal_change_detection.ipynb
  14_error_analysis.ipynb
```

## Launch the Streamlit Dashboard

```bash
streamlit run app/app.py
```

---

# Dataset

## EuroSAT

The primary dataset used for training and evaluation is the **EuroSAT RGB** dataset, consisting of **27,000 Sentinel-2 satellite images** belonging to ten land-use classes.

### Classes

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

### Data Preparation

The project includes:

* Dataset verification
* RGB image preprocessing
* Spatial metadata extraction
* Geographic clustering
* Spatially-aware train/validation/test split generation
* Random split generation for comparison
* Dataset statistics and visualization

---

## UC Merced Land Use Dataset

To evaluate cross-dataset generalization, the trained model is tested on a mapped subset of the **UC Merced Land Use Dataset**.

The original 21 UC Merced classes are mapped to compatible EuroSAT classes before evaluation.

---

# Training

The project implements three progressively stronger models.

## Baseline CNN

A custom convolutional neural network trained from scratch on the spatially separated EuroSAT dataset.

## ResNet18 (Frozen)

A pretrained ResNet18 backbone with only the final classification layer trained.

## ResNet18 (Fine-Tuned)

The best-performing model.

Features include:

* Transfer Learning
* Fine-Tuning
* Early Stopping
* Model Checkpointing
* Spatially-aware evaluation

---

## Temporal Change Detection

The fine-tuned ResNet18 backbone is reused as a feature extractor.

For two satellite images:

1. Extract 512-dimensional embeddings.
2. Compute cosine similarity.
3. Compare against an operating threshold.
4. Predict:

   * Changed
   * Unchanged

An absolute pixel-difference heatmap is generated to visually highlight changed regions.

---

# Evaluation

The project evaluates the models using several complementary experiments.

## Spatial Evaluation

Performance is measured on a geographically separated EuroSAT test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Confusion Matrix

---

## External Validation

The fine-tuned model is evaluated on the mapped UC Merced dataset to measure generalization to unseen data.

---

## Random vs Spatial Split Comparison

A controlled experiment compares:

* Random dataset split
* Spatially-aware dataset split

to demonstrate the impact of spatial data leakage on reported performance.

---

## Error Analysis

Comprehensive error analysis includes:

* Confidence distribution
* Class-wise performance
* Confusion analysis
* Highest-confidence errors
* Lowest-confidence predictions
* Visual inspection of correct and incorrect predictions

---

# Dashboard

The project includes an interactive Streamlit dashboard.

## Land-Use Classification

Features:

* Upload satellite image
* Predict land-use class
* Display prediction confidence
* Show top predictions
* Visualize class probabilities

---

## Temporal Change Detection

Features:

* Upload T1 and T2 satellite images
* Compute cosine similarity
* Predict Changed / Unchanged
* Display similarity score
* Generate visual change heatmap

---

# Project Structure

```text
│
├── app/
├── checkpoints/
├── data/
│ ├── eurosat/
│ ├── ucmerced/
│ └── processed/
├── notebooks/
├── outputs/
│ ├── change_detection/
│ ├── confusion matrices/
│ ├── error_analysis/
│ └── figures/
├── src/
│ ├── datasets/
│ ├── models/
│ ├── utils/
├── tests/
├── README.md
└── requirements.txt

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Pillow

---

# Future Improvements

* Semantic segmentation for pixel-level change detection
* Support for multispectral Sentinel-2 imagery
* Vision Transformer (ViT) backbone
* Self-supervised representation learning
* Interactive geographic visualization
* Deployment using Docker or cloud infrastructure
