# Predicting Lung Disease Using Data Mining and Visualisation techniques

## Dataset

* **Source**: [Kaggle Lung Cancer Dataset](https://doi.org/10.34740/kaggle/dsv/10827884)
* **Records**: 5,000
* **Features**: 18 behavioural, environmental, and medical variables
* **Note**: Dataset is synthetic but based on realistic distributions and domain knowledge

## 🧠 Project Workflow

1. **Data Understanding & Cleaning**

   * Checked for missing values and duplicates
   * Verified data types and structure
   * Ensured all records were valid

2. **Exploratory Data Analysis (EDA)**

   * Plotted distributions (age, gender, smoking)
   * Used bar plots, pie charts, and correlation heatmaps
   * Interactive visualizations built with Plotly for deeper subgroup insights

3. **Feature Engineering**

   * Created age groupings
   * Addressed outliers in key features like immune stress and breathing issues

4. **Model Building**

   * Trained 7 classifiers:

     * Logistic Regression
     * Random Forest ✅ (Best Overall)
     * Gradient Boosting
     * Naïve Bayes
     * Decision Tree
     * K-Nearest Neighbours (KNN)
     * Support Vector Machine (SVM)
   * Data split: 80% training / 20% test
   * Basic hyperparameter tuning (e.g., `n_estimators=100` for Random Forest)

5. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1, AUC
   * Visualization: Confusion matrix, ROC curves, heatmaps
   * Random Forest had best overall performance and balance

##  Key Findings

* **Smoking** was the most important predictor of pulmonary disease.
* **Other strong predictors**: Breathing issues, low oxygen saturation, energy levels, throat discomfort.
* **Environmental exposure** to pollution increased disease risk but was not a standalone cause.
* **Age and smoking** together showed a clear pattern: seniors who smoke had the highest disease rates.
* **Logistic Regression** had the best AUC score (0.936), making it interpretable and powerful.
* **KNN and SVM underperformed**, especially in classifying the disease group.

## Tools & Libraries

* Python (Jupyter Notebook)
* pandas, matplotlib, seaborn
* scikit-learn
* plotly (interactive visualization)

##  Recommendations

* Apply **k-fold cross-validation** (e.g., k=5) for more robust model evaluation.
* Use **dimensionality reduction** (like PCA) to improve performance and reduce complexity.
* Consider incorporating **explainable AI** tools like SHAP or LIME in future iterations to improve clinical transparency.
* Exclude features with low predictive power (e.g., gender, alcohol, mental stress) in future models.

##  Reflection

This project strengthened my end-to-end skills in:

* Data cleaning and feature engineering
* Building, evaluating, and comparing classification models
* Extracting actionable health insights from structured data
* Applying visualisation for exploratory and explanatory analysis

It also deepened my understanding of model selection strategies, especially in health contexts where **interpretability, precision, and recall** are critical.
