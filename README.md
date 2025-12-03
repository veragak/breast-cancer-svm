# Breast Cancer Classification using Support Vector Machines (SVM)
**Author:** Vera Gak Anagrova  
**Master in Data Science & Marketing Analytics – Erasmus School of Economics (ESE)**

This repository contains a full end-to-end machine learning workflow for classifying breast tumors as **malignant (M)** or **benign (B)** using the *Breast Cancer Wisconsin Diagnostic* dataset.  
The project includes:

- Data preprocessing and feature engineering  
- Multiple SVM models (linear, radial, tuned radial)  
- Benchmark with Logistic Regression  
- Full comparison of performance metrics  
- Variable importance analysis  
- 2D decision boundary visualization  
- A **Shiny web application** for real-time tumor classification  
- Project presentation slides  

This work demonstrates practical expertise in machine learning, R programming, reproducible workflows, and model deployment.

---

## Project Structure

`breast-cancer-svm`
- `data/`
    - breast_cancer.csv
- `R-Code/`
    - breast_cancer_svm.R # Full analysis pipeline
- `app/`
    - app.R # Shiny web application
    - svm_shiny_model.rds # Tuned SVM model (auto-generated)
-  `figs/` # Automatically saved figures
     - histograms.pdf
     - svm_linear_cv.png
     - svm_radial_cv.png
     - confusion_matrices
     - decision_boundary_2d.png
- `slides/`
    - breast_cancer_svm_presentation.pdf
- `README.md`


---

## Dataset Description

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, one of the most widely used medical ML datasets.

- **Observations:** 569  
- **Predictors:** 30 continuous features describing cell nuclei (radius, texture, concavity, smoothness, perimeter…)  
- **Target:** diagnosis (`M` = malignant, `B` = benign)

After preprocessing:
- ID-like variables are removed  
- Outcome is converted to factor  
- Numeric features are standardized **only using training data parameters**  
- Optional histogram PDF is generated for exploratory analysis  

---

## Research Question

> **Is Support Vector Machines an accurate method to predict whether a cell is malignant or benign?**

This is answered by comparing SVM models with logistic regression and evaluating multiple classification metrics.

---

## Methodology

### 1. **Data Preparation**
- Removal of ID variables  
- Diagnosis label encoding (`M`/`B`)  
- Missing value checks  
- Duplicate and outlier checks (z-score > 3)  
- Scaling based on training distribution  
- Train-test split (70% / 30%)

---

### 2. **Models Implemented**

#### ✔ Linear SVM
- Tuning of cost parameter `C`
- Visualization of CV accuracy vs. C

#### ✔ Radial SVM (default)
- Starting gamma = `1 / number_of_features`

#### ✔ Tuned Radial SVM (main model)
Hyperparameter grid:
- **Cost (C):** 0.1, 1, 2, 5, 10, 20, 30, 40  
- **Sigma (γ):** 0.001, 0.01, 0.05, 0.1, 0.2  

Cross-validation:
- **10-fold**
- **Repeated 3 times**
- Evaluation metric: **ROC**

This model is later reduced to 6 interpretable features for deployment in the Shiny app.

#### Logistic Regression (benchmark)

---

## Performance Metrics

All models are evaluated using:

- Accuracy  
- Sensitivity (Recall for malignant tumors)  
- Specificity  
- Balanced Accuracy  
- Kappa Statistic  
- Confusion Matrix  
- ROC (SVM tuned)

The tuned radial SVM achieves the best performance in sensitivity, which is essential in medical diagnostics where false negatives are critical.

---

## Figures & Visualizations

The analysis script automatically generates figures saved inside the `figs/` directory:

### 🔹 1. Histograms of all numeric features  
File: `figs/histograms.pdf`

### 🔹 2. Hyperparameter tuning plots  
Files:
- `figs/svm_linear_cv.png`
- `figs/svm_radial_cv.png`

### 🔹 3. Heatmaps of confusion matrices  
Files:
- `figs/confusion_matrices/linear.png`
- `figs/confusion_matrices/radial_default.png`
- `figs/confusion_matrices/radial_tuned.png`

### 🔹 4. Variable Importance (Radial SVM)  
File:
- `figs/variable_importance.png`

### 🔹 5. 2D Decision Boundary Visualization  
File:
- `figs/decision_boundary_2d.png`

---

## Shiny Web Application

The repository includes a full deployable Shiny app (`app/app.R`) that allows users to:

- Move sliders for 6 tumor-related features  
- Obtain real-time predictions using the tuned SVM model  
- View the classification as **“MALIGNANT”** or **“BENIGN”** with UI feedback colors  

To run the app:

```r
shiny::runApp("app")
```


## How to Reproduce This Project

### **1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-svm.git
cd breast-cancer-svm
```
### **2. Run the analysis script**
source("R-Code/breast_cancer_svm.R")

This will:
- run the full analysis
- generate plots into the `figs/` directory
- train the reduced SVM model
- save the model into `app/svm_shiny_model.rds`

### **3. Launch the Shiny App**
```r
shiny::runApp("app")
```

---

## Results Summary

- The **tuned radial SVM** outperforms both the linear SVM and logistic regression  
- Achieves highest:
  - **Balanced Accuracy**
  - **Sensitivity** for malignant tumors
  - **ROC AUC**
- The model is stable across **repeated cross-validation** folds  
- The decision boundary is **non-linear**, confirming that a radial kernel is appropriate  
- Variable importance analysis identifies **shape-based features** (radius, area, concavity) as strongest predictors  

---

## What This Project Demonstrates

- End-to-end **machine learning workflow in R**
- Strong understanding of **Support Vector Machines (SVM)**
- **Hyperparameter tuning** using the `caret` package
- Clear **visualization & reporting**
- Extensive **model evaluation and comparison**
- Deployment of ML models through a **Shiny web application**
- Clean project structure and fully **reproducible code**

---

## Keywords for Recruiters (SEO)

**Machine Learning • SVM • R • Shiny App • Logistic Regression • Classification • Healthcare Analytics • Predictive Modeling • Data Science • Feature Engineering • Cross-Validation • ROC • caret • Model Deployment • Reproducible ML**




