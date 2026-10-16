# WineLens 🍷
### Wine Quality Prediction Using Machine Learning

WineLens is a data science project that explores the relationship between physicochemical properties and wine quality. It uses exploratory data analysis, regression, classification, model comparison, and feature-importance analysis to predict wine quality scores.

The project also includes a Streamlit application that allows users to enter chemical measurements and compare predictions from two Random Forest models.

## Objectives

- Explore the red and white wine quality datasets.
- Investigate distributions, correlations, duplicate records, outliers, and skewness.
- Train and compare machine learning models for red wine quality prediction.
- Analyze feature importance and model errors.
- Build an interactive application using the trained models.

## Dataset

The project uses the Wine Quality dataset containing red and white vinho verde wine samples and their physicochemical measurements.

The features include:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target is the recorded wine quality score.

The raw CSV files are included in the `Datasets/` folder.

**Dataset source:** [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Models

WineLens compares two prediction approaches:

1. **Random Forest Regressor** — predicts a numerical quality score.
2. **Random Forest Classifier** — predicts an integer quality category.

The modeling experiments focus on the red wine dataset.

### Reported experimental results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5035 | 0.6245 | 0.4032 |
| Default Random Forest Regressor | 0.4238 | 0.5536 | 0.5311 |
| Tuned Random Forest Regressor | 0.4140 | 0.5406 | 0.5528 |

The tuned Random Forest Regressor was the strongest regression model in the experiments conducted.

The Balanced Random Forest Classifier achieved:

| Metric | Score |
|---|---:|
| Accuracy | 68.12% |
| Balanced accuracy | 33.68% |
| Macro F1 | 0.3373 |

These results were obtained using the same test split across multiple experiments. They should be treated as exploratory results rather than an independent final estimate of generalization.

## Feature Importance

Alcohol, sulphates, and volatile acidity ranked among the most influential features in the Random Forest feature-importance analyses.

Feature importance describes predictive relationships learned by a model. It does not establish that a feature causes a change in wine quality.

## Application

The Streamlit application allows users to:

- Enter the 11 physicochemical measurements.
- Obtain a classification prediction.
- Obtain a regression prediction.
- Compare the two model outputs.

The predictions are estimates of recorded quality scores, not professional wine assessments.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/thekarak/WineLens.git
cd WineLens
```

### 2. Create and activate a virtual environment

Using Python 3.11 or another compatible Python version:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

If necessary, install `uv` first using its official installation instructions: https://docs.astral.sh/uv/

### 4. Launch the application

```bash
uv run --active streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Repository Structure

```text
WineLens/
├── README.md
├── WineLens_Final.ipynb
├── app.py
├── requirements.txt
├── .gitignore
├── Datasets/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── models/
│   ├── winelens_classifier.joblib
│   └── winelens_regressor.joblib
└── reports/
    └── figures/
        ├── quality_distribution.png
        ├── correlation_heatmap.png
        ├── alcohol_distribution_by_type.png
        ├── alcohol_vs_quality.png
        ├── volatile_acidity_vs_quality.png
        ├── measurements_by_wine_type.png
        ├── feature_correlation_red.png
        ├── feature_correlation_white.png
        ├── regplot_alcohol.png
        ├── regplot_volatile_acidity.png
        ├── regplot_sulphates.png
        ├── regplot_density.png
        ├── feature_histograms_red.png
        ├── feature_histograms_white.png
        ├── residual_analysis.png
        ├── feature_importance.png
        └── permutation_importance.png
```

## Limitations

- The quality target is imbalanced, with relatively few examples of extreme quality scores.
- The classifier performs substantially worse on balanced metrics than its overall accuracy might suggest.
- The test split was consulted repeatedly during model development.
- Feature importance and correlation do not demonstrate causation.
- Performance may differ on wines from other producers, regions, vintages, or measurement processes.

## Future Improvements

- Establish a final evaluation using an untouched holdout set.
- Improve minority-class evaluation and investigate alternative modeling approaches.
- Add clearer prediction explanations and model evaluation visualizations.
- Improve application design and deploy a public demo.
- Automate reproducible training and evaluation.

## Author

Karak 
