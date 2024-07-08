# Abalone Age Prediction

This project involves predicting the age of Abalones using easily available physical measurements, avoiding the traditional and tedious method of counting rings through a microscope.

## Dataset Overview

### Datasets Used

Two datasets were utilized for this project:

1. **Dataset 1**
   - Additional features derived from original measurements were added, e.g., `(Whole weight + Shell weight)`.
   - Outliers were managed using the IQR method.

2. **Dataset 2**
   - No additional features were added due to a smaller sample size (around 4K samples), aiming to prevent overfitting.
   - Outliers were handled implicitly through model training.

### Dataset Details

- **Features:**
  - **Sex:** Nominal (M, F, I)
  - **Length:** Continuous (mm)
  - **Diameter:** Continuous (mm)
  - **Height:** Continuous (mm)
  - **Whole weight, Whole weight.1, Whole weight.2:** Continuous (grams)
  - **Shell weight:** Continuous (grams)

- **Target Variable:**
  - **Rings:** Integer (+1.5 gives the age in years)

## Data Exploration

- **Heatmap Analysis:** Shows correlations between features and the target variable `Rings`.
- **Literature Review:** Previous studies highlighted classification and regression approaches using machine learning models like ANN, KNN, Naive Bayes, SVM, Linear Regression, Random Forest, and others.

## Prediction Techniques

### Models Explored

1. **Linear Regression Model**
   - Fitted a linear regression model (`y = ax + b`).
   - Evaluation metrics: RMSLE, MAE.

2. **XGBoost**
   - Used XGBoost regressor, hyperparameter tuned with GridSearchCV.
   - Better performance in handling outliers compared to linear regression.

3. **Random Forest**
   - Employed a Random Forest model, hyperparameter tuned with GridSearchCV.
   - Higher errors on smaller target variable values in Dataset 1 due to larger sample size.

4. **Neural Network**
   - Utilized a two-layered neural network.
   - Hyperparameters tuned using Automated Neural Network methods.
   - Achieved lowest RMSLE on Dataset 1, indicating capability to capture complex non-linear relationships.

5. **CatBoost**
   - Applied CatBoostRegressor with L2 regularization.
   - Effective in handling outliers, achieved low RMSLE on Dataset 2 despite smaller sample size.

## Results and Comparisons

- **Model Performance:**
  - Linear Regression: Poor performance on Dataset 1 due to removed outliers during training.
  - Neural Network: Lowest RMSLE and MAE on Dataset 1, robust to complex features and outlier removal.
  - CatBoost: Lowest RMSLE on Dataset 2, robust to smaller sample size and outlier handling.

### Figures and Visualizations

Include key figures from your analysis, such as:

- Heatmap of feature correlations
- Comparison plots of model predictions vs. actual values
- Metric values (RMSLE, MAE) for each model and dataset

## References

- [Kaggle Playground Series: Abalone Competition](https://www.kaggle.com/competitions/playground-series-s4e4/overview)
- [UCI Machine Learning Repository: Abalone Dataset](https://archive.ics.uci.edu/dataset/1/abalone)
- [Mehta, 2019 - Classification of Abalones using Machine Learning](https://www.ijcaonline.org/archives/volume178/number50/mehta-2019-ijca-919425.pdf)
- [Second Research Paper - Abalone Age Prediction Using Machine Learning](https://www.researchgate.net/publication/359927109_Abalone_Age_Prediction_Using_Machine_Learning)
