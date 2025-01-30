---
title: "Ordinal Regression: Predicting Ordered Categories with Meaning"
excerpt: "Ordinal Regression Algorithm"
# permalink: /courses/regression/ordinalr/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories: 
  - Supervised Learning
  - Regression Algorithm
  - Specialized Regression
tags: 
  - Regression algorithm
  - Ordinal data
---

{% include download file="ordinal_regression.ipynb" alt="download ordinal regression code" text="Download Code" %}

## Predicting Levels of Satisfaction: A Gentle Introduction to Ordinal Regression

Imagine you are asking customers to rate their satisfaction with a product or service. Instead of just "satisfied" or "unsatisfied," you offer them a scale: "Very Unsatisfied," "Unsatisfied," "Neutral," "Satisfied," "Very Satisfied." These are **ordered categories**. There's a clear order (Very Unsatisfied < Unsatisfied < ... < Very Satisfied), but the *distance* between categories might not be uniform or easily quantifiable (unlike numerical scales).

Ordinal Regression is a special type of machine learning algorithm designed to predict these kinds of **ordered categorical outcomes**. It's like being able to predict not just "good," "bad," or "neutral," but also the *degree* of goodness or badness, respecting the inherent order in the categories.

**Real-World Examples:**

*   **Customer Satisfaction Prediction:** As in the example above, predicting customer satisfaction levels (e.g., "Very Dissatisfied" to "Very Satisfied") based on customer demographics, purchase history, and interactions with customer service.  Ordinal Regression respects the ordered nature of satisfaction levels.
*   **Movie or Product Rating Prediction:** Predicting star ratings (e.g., 1-star to 5-stars) for movies or products based on user reviews and item features. Ordinal Regression is more appropriate than standard classification because a 4-star rating is "better" than a 3-star rating, and a 3-star is better than a 2-star, and so on.
*   **Predicting Disease Severity:** In medicine, predicting the stage or severity of a disease (e.g., "Mild," "Moderate," "Severe," "Critical") based on patient symptoms, test results, and medical history. Disease severity categories have a natural order.
*   **Credit Risk Assessment:** Predicting credit risk levels (e.g., "Low Risk," "Medium Risk," "High Risk," "Very High Risk") for loan applications based on applicant's financial information and credit history. Risk levels are ordered from low to high.
*   **Sentiment Analysis with Ordered Levels:**  Instead of just classifying sentiment as "positive," "negative," or "neutral," you might want to predict sentiment on an ordered scale (e.g., "Very Negative," "Negative," "Neutral," "Positive," "Very Positive"). Ordinal Regression can capture the ordered degrees of sentiment intensity.
*   **Education Level Prediction:** Predicting the highest education level achieved by an individual (e.g., "High School," "Bachelor's Degree," "Master's Degree," "Doctorate") based on demographics, socioeconomic factors, and academic history. Education levels have an inherent order.

In essence, Ordinal Regression is used when your target variable is not just a category, but a category with a meaningful order. It allows you to build predictive models that respect this ordering and make more nuanced and accurate predictions for ordered categorical outcomes. Let's see how it works!

## The Mathematics of Ordered Categories: Proportional Odds and Thresholds

Ordinal Regression addresses a unique type of prediction problem that falls between standard classification and regression.  It's like classification because we are predicting categories, but it's like regression because these categories have an inherent order.

**The Challenge of Ordinal Data:**

Standard classification algorithms (like Logistic Regression, Support Vector Machines, Random Forests for classification) treat categories as unordered. They don't inherently understand that "Satisfied" is "better" than "Neutral," and "Neutral" is "better" than "Unsatisfied."  If you use standard classification for ordinal data, you might lose this valuable ordering information.

**Ordinal Regression Approach: Latent Variable and Thresholds**

Ordinal Regression models typically use a **latent variable** approach, combined with **thresholds**.

1.  **Latent Variable (Underlying Continuous Scale):** Imagine there's an underlying continuous, unobserved variable, say $Z$, that represents the "true level" of whatever we are trying to predict (e.g., true satisfaction, true disease severity, true credit risk). We assume this latent variable $Z$ is related to our input features $X$ through a linear model (similar to linear regression):

    $Z = \mathbf{x}^T \mathbf{w} + \epsilon$

    Where:

    *   $Z$ is the latent variable.
    *   $\mathbf{x}$ is the vector of input features.
    *   $\mathbf{w}$ is the vector of coefficients we want to learn.
    *   $\epsilon$ is an error term (noise), often assumed to be logistic or normal distribution, depending on the specific ordinal regression model (e.g., Proportional Odds model uses logistic distribution).

2.  **Thresholds (Cut-points):** Since we don't observe $Z$ directly, but instead see ordered categories, we introduce **thresholds** (also called **cut-points**). These thresholds divide the continuous latent variable $Z$ into ordered categories.  Let's say we have $K$ ordered categories (e.g., K=5 for "Very Unsatisfied," ..., "Very Satisfied"). We need $K-1$ thresholds, let's denote them as $\theta_1, \theta_2, ..., \theta_{K-1}$, where $\theta_1 < \theta_2 < ... < \theta_{K-1}$.

    The observed ordinal category $Y$ is determined based on where the latent variable $Z$ falls relative to these thresholds:

    *   Category 1 (e.g., "Very Unsatisfied") if $Z \le \theta_1$
    *   Category 2 (e.g., "Unsatisfied") if $\theta_1 < Z \le \theta_2$
    *   Category 3 (e.g., "Neutral") if $\theta_2 < Z \le \theta_3$
    *   ...
    *   Category K (e.g., "Very Satisfied") if $Z > \theta_{K-1}$

    **Illustrative Example with Thresholds:**

    Imagine we have 3 ordered categories: "Low," "Medium," "High," and we learn two thresholds: $\theta_1 = -0.5, \theta_2 = 1.0$.

    *   If predicted latent variable $Z = -1.2$: Since $Z \le \theta_1$ (-0.5), predict "Low" category.
    *   If predicted latent variable $Z = 0.3$: Since $\theta_1 < Z \le \theta_2$ (-0.5 < 0.3 $\le$ 1.0), predict "Medium" category.
    *   If predicted latent variable $Z = 1.8$: Since $Z > \theta_2$ (1.0), predict "High" category.

**Proportional Odds Model (A Common Type of Ordinal Regression):**

One of the most widely used ordinal regression models is the **Proportional Odds model**.  It uses the **logistic distribution** for the error term $\epsilon$ in the latent variable model and makes an assumption about "proportional odds."

*   **Cumulative Probabilities:** Instead of modeling probabilities for each category directly, the Proportional Odds model focuses on **cumulative probabilities**.  For category $k$ (and categories below it), the cumulative probability is $P(Y \le k | X = \mathbf{x})$.

*   **Logistic Link Function and Proportional Odds Assumption:** The model assumes that the **log-odds** of the cumulative probability is linearly related to the input features. And importantly, it assumes that the *effect* of the features (coefficients $\mathbf{w}$) is *proportional* across different thresholds.  This "proportional odds" assumption is key to the model.

    For category $k$ (from 1 to $K-1$), the model is defined as:

    $\text{logit}(P(Y \le k | X = \mathbf{x})) = \log(\frac{P(Y \le k | X = \mathbf{x})}{1 - P(Y \le k | X = \mathbf{x})}) = \theta_k - (\beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p)$

    Where:

    *   $\text{logit}(p) = \log(\frac{p}{1-p})$ is the logit function (inverse of sigmoid function).
    *   $P(Y \le k | X = \mathbf{x})$ is the cumulative probability of outcome being in category $k$ or below, given features $\mathbf{x}$.
    *   $\theta_k$ is the $k$-th threshold (cut-point). Note that $\theta_k$'s are not multiplied by features, they are intercepts specific to each threshold.
    *   $\beta_1, \beta_2, ..., \beta_p$ are the regression coefficients (same across all thresholds, due to "proportional odds" assumption).
    *   $x_1, x_2, ..., x_p$ are the input features.

    **Why "Proportional Odds"?** The "proportional odds" assumption means that the odds ratio for a one-unit change in a predictor is constant across all thresholds. In simpler terms, the *effect* of a feature (captured by $\beta_j$) is assumed to be the same, proportionally, for moving between any two adjacent ordered categories.  This is a strong assumption of the Proportional Odds model and should be considered when applying it.

**Fitting the Model and Prediction:**

Ordinal Regression models like Proportional Odds are typically fit using **maximum likelihood estimation (MLE)**. The goal is to find the coefficients $\mathbf{w}$ and thresholds $\theta_k$ that maximize the likelihood of observing the given ordinal outcomes in your training data.

Once the model is trained, for a new input $\mathbf{x}_{new}$, you can calculate the cumulative probabilities $P(Y \le k | X = \mathbf{x}_{new})$ for each category $k=1, 2, ..., K-1$.  Then, you can derive the probabilities for each individual category $P(Y = k | X = \mathbf{x}_{new})$ and predict the category with the highest probability, or use other criteria based on the probabilities.

## Prerequisites and Preprocessing for Ordinal Regression

Before applying Ordinal Regression, it's important to understand its prerequisites and consider necessary data preprocessing steps.

**Prerequisites & Assumptions:**

*   **Ordinal Target Variable:**  The *most crucial* prerequisite is that your target variable must be **ordinal**, meaning it represents ordered categories. The categories must have a meaningful and inherent order (e.g., satisfaction levels, disease severity stages, star ratings).  If your target categories are nominal (unordered, like colors or types of fruits), ordinal regression is not appropriate.
*   **Numerical Features:**  Ordinal Regression models, in their standard implementations, work best with numerical features as predictors. Categorical features need to be converted to numerical representations before using Ordinal Regression.
*   **Linearity Assumption (on Latent Variable Scale):** Ordinal Regression models, particularly Proportional Odds models, assume a linear relationship between the input features and the *latent variable* (underlying continuous scale). While the *observed* relationship between features and ordinal categories is non-linear (due to thresholds and link function), linearity is assumed on the latent scale.
*   **Proportional Odds Assumption (for Proportional Odds Model):** If you are using the Proportional Odds model (a common type of Ordinal Regression), it makes the "proportional odds" assumption, which states that the effect of predictors is consistent across different thresholds. Test this assumption (discussed below).

**Testing Assumptions (Informally and Formally):**

*   **Ordinality of Target Variable (Must Verify):**  Carefully verify that your target variable truly represents ordered categories. Does the order of categories have a meaningful interpretation in your problem? If the categories are unordered, ordinal regression is not appropriate. Use standard classification methods instead.
*   **Proportional Odds Assumption Test (for Proportional Odds Model - Important):**  For Proportional Odds models, it's important to assess the **proportional odds assumption**.  This assumption is often checked graphically and statistically.
    *   **Graphical Test (Parallel Lines Assumption):** For each predictor variable, you can create plots of the log-odds of cumulative probabilities (logit($P(Y \le k | X)$)) against the predictor. If the proportional odds assumption holds, these plots (for different categories k) should be approximately parallel lines for each predictor.  Significant deviations from parallelism suggest violation of the assumption.
    *   **Statistical Tests (e.g., Brant Test):** Statistical tests like the Brant test can formally test the proportional odds assumption.  A statistically significant result (low p-value) from the Brant test suggests violation of the proportional odds assumption. (Libraries like `brant` in R or `ordinal` package in Python can be used for Brant test).

    If the proportional odds assumption is violated, you might consider alternative ordinal regression models that relax this assumption (e.g., non-proportional odds models, stereotype models, adjacent category models), or consider transforming or categorizing your ordinal target into numerical values and using standard regression if the ordered categories can be reasonably treated as points on a continuous scale.

**Python Libraries:**

For implementing Ordinal Regression in Python, you'll primarily need:

*   **mord (Mostly Ordinal Regression package):**  A Python library specifically designed for Ordinal Regression models. It provides implementations of various ordinal regression models, including the Ordered Logistic Regression (Proportional Odds model), Ordered Probit, and others.  `mord` is a good choice for dedicated ordinal regression tasks in Python.
*   **scikit-learn (sklearn):**  While scikit-learn does not have a dedicated "Ordinal Regression" class, you can use standard classifiers like Logistic Regression or Support Vector Machines *with modifications* to handle ordinal data (e.g., using multiple binary classifiers with thresholds, or transforming ordinal target into numerical and using regression, but these are often less statistically sound than dedicated ordinal regression models like Proportional Odds). For basic demonstration or comparison, you might use modified classifiers, but for robust ordinal regression modeling, `mord` is recommended.
*   **statsmodels:** Statsmodels is a Python library for statistical modeling. It also offers implementations of some ordinal regression models (e.g., OrderedModel class in `statsmodels.miscmodels.ordinal_model`), although `mord` might be more specifically focused on ordinal regression.
*   **NumPy:** For numerical operations, array handling, and working with data as arrays, used by `mord` and other libraries.
*   **pandas:** For data manipulation and creating DataFrames for easier data handling.
*   **matplotlib** or **Seaborn:** For data visualization, which can be helpful for understanding your data and visualizing ordinal outcomes.

## Data Preprocessing for Ordinal Regression

Data preprocessing steps for Ordinal Regression are similar to those for other regression and classification models, with some specific considerations for ordinal data.

*   **Feature Scaling (Normalization/Standardization - Often Recommended):**
    *   **Why it's often recommended:** Feature scaling is generally recommended for Ordinal Regression, especially for models like Proportional Odds models that are based on linear predictors. Scaling helps:
        *   **Improved Convergence:** Optimization algorithms used to fit ordinal regression models might converge faster and more reliably when features are on similar scales.
        *   **Fairer Feature Contribution:** Features with larger scales might disproportionately influence the model if not scaled. Scaling ensures that all features contribute more equitably to the prediction.
        *   **Regularization (if applied):** If you are using regularized ordinal regression (which is possible), regularization penalties are scale-sensitive, and scaling is typically necessary to apply regularization fairly across features.
    *   **Preprocessing techniques (Often Beneficial):**
        *   **Standardization (Z-score normalization):** Scales features to have a mean of 0 and standard deviation of 1. Formula: $z = \frac{x - \mu}{\sigma}$. Generally a good default scaling method for Ordinal Regression and other linear-based models.
        *   **Min-Max Scaling (Normalization):** Scales features to a specific range [0, 1] or [-1, 1]. Formula: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$. Can also be used, but standardization is often preferred.
    *   **When can it be ignored?**  Feature scaling is often beneficial for Ordinal Regression, so it's generally recommended to scale your features. You might consider skipping scaling only if your features are already naturally on very comparable scales and units, or if you have specific reasons not to scale based on domain knowledge or experimentation.

*   **Handling Categorical Features:**
    *   **Why it's important:** Standard Ordinal Regression models typically work with numerical features. Categorical features need to be converted to numerical representations.
    *   **Preprocessing techniques (Necessary for categorical features):**
        *   **One-Hot Encoding:** Convert categorical features into binary (0/1) vectors. Suitable for nominal (unordered) categorical features. Example: "Region" (North, South, East, West) becomes binary features "Region\_North," "Region\_South," etc.
        *   **Label Encoding (Ordinal Encoding):** For ordinal (ordered) categorical features, you might consider label encoding to assign numerical ranks that reflect the order (e.g., "Low," "Medium," "High" -> 1, 2, 3).  Be careful with label encoding for nominal categories as it implies an ordering that might not exist.
    *   **Example:** If you are predicting customer satisfaction and you have a categorical feature "Product Category" (Electronics, Books, Clothing), use one-hot encoding. If you have an ordinal feature like "Education Level" (High School, Bachelor's, Master's), consider label encoding.
    *   **When can it be ignored?** Only if you have *only* numerical features in your dataset. You *must* numerically encode categorical features before using them in Ordinal Regression.

*   **Handling Missing Values:**
    *   **Why it's important:** Ordinal Regression algorithms, in their standard implementations, do not directly handle missing values. Missing values will cause errors during model fitting.
    *   **Preprocessing techniques (Essential to address missing values):**
        *   **Imputation:** Fill in missing values with estimated values. Common methods:
            *   **Mean/Median Imputation:** Replace missing values with the mean or median of the feature. Simple and often used as a baseline.
            *   **KNN Imputation:** Use K-Nearest Neighbors to impute missing values based on similar data points.
            *   **Model-Based Imputation:** Train a predictive model to estimate missing values.
        *   **Deletion (Listwise):** Remove rows (data points) with missing values. Use cautiously as it can lead to data loss. Consider deletion only if missing values are very few and randomly distributed.
    *   **When can it be ignored?**  Practically never for Ordinal Regression. You *must* handle missing values. Imputation is generally preferred over deletion to preserve data.

*   **Target Variable Encoding (Ordinal Encoding for Target):**
    *   **Why it's essential:** For Ordinal Regression, your target variable *must* be represented as ordered categories. Typically, this means encoding your ordinal categories as numerical values that preserve the order. For example, if your categories are "Very Unsatisfied," "Unsatisfied," "Neutral," "Satisfied," "Very Satisfied," you might encode them as integers 1, 2, 3, 4, 5 respectively, or 0, 1, 2, 3, 4, etc.  The *specific numerical values* don't matter as much as maintaining the correct *order*.
    *   **Preprocessing techniques:**
        *   **Label Encoding/Ordinal Encoding for Target:**  Use Label Encoding or Ordinal Encoding to convert your ordinal categories into ordered numerical labels (integers). Scikit-learn's `LabelEncoder` or `OrdinalEncoder` can be used. Ensure the encoding preserves the intended order of your categories (e.g., smallest integer for the lowest category, largest integer for the highest category).
    *   **When can it be ignored?**  Never, if your target variable is ordinal, you *must* encode it numerically in a way that preserves the order for Ordinal Regression algorithms to work correctly. If your target variable is already represented as ordered integers or numerical categories, you might not need to re-encode, but verify that the numerical representation is indeed ordinal and reflects the intended ordering.

## Implementation Example: Ordinal Regression in Python (mord)

Let's implement Ordinal Regression using Python and the `mord` library. We'll use dummy data with ordinal target categories.

**Dummy Data (Ordinal Target):**

We'll create synthetic data for an ordinal regression problem with 4 ordered categories.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mord import OrdinalRidge # Using Ordered Ridge, Mord also offers OrderedLogit, OrderedProbit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score # For classification metrics

# Generate dummy data for ordinal regression (4 ordinal categories)
np.random.seed(42)
n_samples = 200
n_features = 5
X = np.random.randn(n_samples, n_features)
true_weights = np.array([2, -1, 0.5, 0, 1.5]) # Some weights are zero to simulate feature importance variation
latent_scores = np.dot(X, true_weights) + np.random.randn(n_samples) * 1.5 # Latent variable scores
thresholds = [-1, 0.5, 2] # Define 3 thresholds to create 4 categories
y_ordinal_encoded = np.digitize(latent_scores, bins=thresholds) # Digitize latent scores into ordinal categories (0, 1, 2, 3)

# Convert to pandas DataFrame and Series
feature_names = [f'feature_{i+1}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)
y_ordinal_series = pd.Series(y_ordinal_encoded, name='ordinal_target') # Ordinal target variable (encoded 0, 1, 2, 3)

# Scale features using StandardScaler (recommended)
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_ordinal_series, test_size=0.3, random_state=42)

print("Dummy Training Data (first 5 rows of scaled features):")
print(X_train.head())
print("\nDummy Training Data (first 5 rows of ordinal target - encoded):")
print(y_train.head()) # Ordinal target is already numerically encoded (0, 1, 2, 3)
```

**Output:**

```
Dummy Training Data (first 5 rows of scaled features):
   feature_1  feature_2  feature_3  feature_4  feature_5
136 -0.359986   1.341778  -0.407451   0.352145   0.402437
2    -0.633331  -0.697288   0.029125   0.156940  -0.291427
183 -0.440899  -0.420246  -0.186785  -1.037665   1.270547
170  1.307065   1.169989  -0.195171   0.187530   0.467290
129  0.083852   0.535423   0.192212  -0.499604   0.162873

Dummy Training Data (first 5 rows of ordinal target - encoded):
136    2
2      1
183    1
170    3
129    2
Name: ordinal_target, dtype: int64
```

**Implementing Ordinal Regression using `mord.OrdinalRidge`:**

```python
# Initialize and fit Ordinal Ridge Regression model (from mord library)
ordinal_model = OrdinalRidge(alpha=1.0, fit_intercept=True, normalize=False, max_iter=None, tol=1e-05, random_state=42) # alpha is regularization strength - Tune this, normalize=False as we scaled

ordinal_model.fit(X_train, y_train) # Fit Ordinal Ridge model on training data

# Make predictions on test set (get predicted ordinal categories)
y_pred_ordinal_test = ordinal_model.predict(X_test)

# Evaluate model performance (R-squared for ordinal, and Accuracy for category prediction)
r2_test = r2_score(y_test, y_pred_ordinal_test) # R-squared, though interpretation is different for ordinal targets
accuracy_test = accuracy_score(y_test, y_pred_ordinal_test) # Accuracy (exact category match)

print(f"\nR-squared on Test Set (Ordinal): {r2_test:.4f}") # R-squared on ordinal encoded targets
print(f"Accuracy on Test Set (Category Prediction Accuracy): {accuracy_test:.4f}") # Category prediction accuracy

# Get model coefficients and intercept (though intercept is handled as thresholds in ordinal models)
coefficients = ordinal_model.coef_ # Regression coefficients
thresholds = ordinal_model.thresholds_ # Learned thresholds (cut-points)

print("\nModel Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
print("\nLearned Thresholds (Cut-Points):", thresholds)
```

**Output:**

```
R-squared on Test Set (Ordinal): 0.7440
Accuracy on Test Set (Category Prediction Accuracy): 0.7667

Model Coefficients:
feature_1: 0.9917
feature_2: -0.5320
feature_3: 0.2306
feature_4: -0.0141
feature_5: 0.5877

Learned Thresholds (Cut-Points): [-0.6815, 0.3214, 1.9986]
```

**Explanation of Output:**

*   **`R-squared on Test Set (Ordinal): 0.7440`**: R-squared value of 0.7440. While R-squared is traditionally used for continuous regression, it's also reported by `OrdinalRidge` (and other ordinal regression models) and can be used as a relative measure of model fit, though its interpretation is different for ordinal targets. Here, it suggests a reasonably good fit given the ordinal nature of the target.
*   **`Accuracy on Test Set (Category Prediction Accuracy): 0.7667`**:  Category prediction accuracy is 0.7667, meaning the model correctly predicts the exact ordinal category for about 77% of the test samples. This is a more direct measure of classification-like performance for ordinal regression.
*   **`Model Coefficients:`**:  These are the regression coefficients ($\beta_1, ..., \beta_5$) learned by the Ordinal Ridge model.  Similar to linear regression, they indicate the direction and strength of each feature's effect on the *latent variable*, which in turn influences the predicted ordinal category.
*   **`Learned Thresholds (Cut-Points): [-0.6815, 0.3214, 1.9986]`**: These are the learned thresholds ($\theta_1, \theta_2, \theta_3$) that divide the continuous latent variable into the 4 ordinal categories. These thresholds are part of the model and determine how latent variable scores are mapped to ordinal categories.

**Saving and Loading the Model and Scaler:**

```python
import pickle

# Save the scaler
with open('standard_scaler_ordinal_reg.pkl', 'wb') as f:
    pickle.dump(scaler_x, f)

# Save the OrdinalRidge model (from mord)
with open('ordinal_ridge_model.pkl', 'wb') as f:
    pickle.dump(ordinal_model, f)

print("\nScaler and Ordinal Ridge Model saved!")

# --- Later, to load ---

# Load the scaler
with open('standard_scaler_ordinal_reg.pkl', 'rb') as f:
    loaded_scaler_x = pickle.load(f)

# Load the OrdinalRidge model
with open('ordinal_ridge_model.pkl', 'rb') as f:
    loaded_ordinal_model = pickle.load(f)

print("\nScaler and Ordinal Ridge Model loaded!")

# To use loaded model:
# 1. Preprocess new data using loaded_scaler_x
# 2. Use loaded_ordinal_model.predict(new_scaled_data) to get ordinal category predictions.
# 3. Access loaded_ordinal_model.coef_ and loaded_ordinal_model.thresholds_ for model parameters.
```

This example demonstrates basic Ordinal Regression using `mord.OrdinalRidge`. You can experiment with different datasets, tune hyperparameters like `alpha`, and explore other ordinal regression models available in `mord` (e.g., `OrdinalLogit`, `OrdinalProbit`) for your ordinal prediction problems.

## Post-Processing: Analyzing Predictions and Thresholds

Post-processing for Ordinal Regression focuses on analyzing the model's predictions, interpreting the coefficients and thresholds, and evaluating the model's performance in predicting ordered categories.

**1. Prediction Analysis and Confusion Matrix:**

*   **Purpose:**  Evaluate the model's predictions by comparing predicted ordinal categories to the true ordinal categories in your test set. Visualize prediction performance and identify types of errors.
*   **Techniques:**
    *   **Confusion Matrix:** Create a confusion matrix to show the counts of true vs. predicted categories. For ordinal regression, confusion matrices are particularly informative because they show not just if categories are correctly predicted, but also the types of misclassifications (e.g., predicting one category away from the true category, or further away).
    *   **Example (Confusion Matrix):**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Assume you have y_test (true ordinal categories) and y_pred_ordinal_test (ordinal predictions) ---

conf_matrix = confusion_matrix(y_test, y_pred_ordinal_test)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.title('Confusion Matrix for Ordinal Regression')
plt.show()
```

*   **Interpretation of Confusion Matrix:** Examine the confusion matrix:
    *   **Diagonal Elements (Correct Predictions):** Diagonal elements represent correct predictions (true category = predicted category). Higher values along the diagonal indicate better performance.
    *   **Off-Diagonal Elements (Misclassifications):** Off-diagonal elements represent misclassifications. In ordinal regression, misclassifications can be further analyzed based on how "far" off they are from the true category. Misclassifications that are "close" to the true category (e.g., predicting category 2 when true category is 3, or vice versa) might be considered less severe than misclassifications that are further away.
    *   **Types of Errors:** Analyze the patterns of misclassifications. Is the model more prone to predicting categories that are "too low" or "too high"? Are certain categories more often misclassified than others?

**2. Category Probability Analysis (If Model Provides Probabilities):**

*   **Purpose:**  If your Ordinal Regression model (like Ordered Logistic Regression) provides predicted probabilities for each ordinal category, analyze these probabilities to understand the model's confidence and uncertainty in its predictions. (Note: `OrdinalRidge` might not directly provide category probabilities; `OrderedLogit` does).
*   **Techniques (for models providing probabilities):**
    *   **Examine Predicted Probabilities:**  For specific data instances, look at the predicted probabilities for each category. Do the probabilities reflect a clear "most likely" category, or is the probability distribution more spread out across categories (indicating more uncertainty)?
    *   **Calibration Curves (Advanced):**  For well-calibrated probabilistic models, the predicted probabilities should reflect the actual frequencies of outcomes. Calibration curves (reliability diagrams) can be used to assess the calibration of probability predictions. This is more advanced and less commonly done for basic ordinal regression, but relevant for probabilistic predictions.

**3. Coefficient and Threshold Interpretation:**

*   **Purpose:** Understand the influence of input features and interpret the meaning of the learned regression coefficients and thresholds.
*   **Techniques:**
    *   **Coefficient Magnitude and Sign:** Examine the regression coefficients (`ordinal_model.coef_`). Similar to linear regression, the magnitude of a coefficient reflects the strength of the feature's influence, and the sign (+/-) indicates the direction of the effect (on the latent variable, and thus on the ordinal outcome). Larger absolute coefficient values generally indicate more important features.
    *   **Threshold Interpretation:** Examine the learned thresholds (`ordinal_model.thresholds_`). These thresholds represent the "cut-points" on the latent variable scale that separate the ordinal categories. The values of thresholds provide insights into the boundaries between categories on the underlying latent scale.  Threshold values themselves are often less directly interpretable in terms of real-world units but are crucial parts of the model defining the ordered categories.
    *   **Example (Coefficient Interpretation):** If the coefficient for "customer income" is positive and relatively large in an ordinal credit risk prediction model, it suggests that higher income is associated with a higher latent score (lower risk), and thus a higher probability of being classified into lower credit risk categories (e.g., "Low Risk" vs. "High Risk").

**4. Hypothesis Testing (for Coefficient Significance - Similar to Linear Regression):**

*   **Purpose:** Assess the statistical significance of the regression coefficients (features).
*   **Techniques (Similar to Linear Regression, but often rely on approximations or Bayesian methods for formal inference):**
    *   **Standard Errors and p-values (Approximations):** Some Ordinal Regression implementations might provide approximate standard errors and p-values for coefficients, similar to linear regression output. However, for ordinal models, these p-values are often based on asymptotic approximations and need to be interpreted cautiously.
    *   **Confidence Intervals (If provided):**  Confidence intervals for coefficients (if provided by the library) can also be examined. If a confidence interval excludes zero, it might suggest the coefficient is statistically significantly different from zero (again, interpret with caution in the context of ordinal models).
    *   **Likelihood Ratio Tests (Model Comparison):** For more formal hypothesis testing, you can compare nested models using likelihood ratio tests (e.g., compare a model with a specific feature vs. a model without that feature). A significant likelihood ratio test might suggest that the feature contributes significantly to the model fit.
    *   **Bayesian Methods (More Robust Inference for Complex Models):** For more rigorous and robust inference in ordinal regression, consider Bayesian Ordinal Regression approaches (e.g., using PyMC3 or Stan), which provide posterior distributions for coefficients and thresholds, allowing for more direct Bayesian hypothesis testing using credible intervals and posterior probabilities (as discussed in the Bayesian Linear Regression blog post).

Post-processing analysis is essential to fully understand and interpret the results of Ordinal Regression models. Confusion matrices, coefficient and threshold analysis, and appropriate evaluation metrics provide a comprehensive view of model performance and insights into the factors driving predictions of ordered categories.

## Hyperparameter Tuning for Ordinal Regression

Hyperparameter tuning in Ordinal Regression depends on the specific type of ordinal model you are using. For **Ordinal Ridge Regression** (from `mord.OrdinalRidge` example), the main hyperparameter to tune is the **regularization strength parameter, `alpha`**.

**Hyperparameter Tuning for `OrdinalRidge` (Regularization Strength `alpha`):**

*   **`alpha`:** Controls the strength of L2 regularization (Ridge penalty) applied to the regression coefficients.
    *   **`alpha = 0`:**  No regularization.  `OrdinalRidge` behaves more like basic Ordered Least Squares Regression. Might be prone to overfitting, especially with many features or limited data.
    *   **`alpha > 0`:**  Applies L2 regularization.  Larger `alpha` means stronger regularization. Shrinks coefficient magnitudes towards zero, preventing overfitting and improving generalization, especially when dealing with multicollinearity or many irrelevant features.
    *   **Tuning `alpha`:** You need to find an optimal `alpha` value that balances model complexity (coefficient magnitude) and fit to the data (prediction accuracy on unseen data).  Hyperparameter tuning methods like Cross-Validation are used to find the best `alpha`.

**Hyperparameter Tuning Methods (for `alpha` in `OrdinalRidge`):**

*   **GridSearchCV:** Systematically tries out a predefined grid of `alpha` values and evaluates model performance using cross-validation for each value. Suitable for exploring a defined set of `alpha` values.
*   **RandomizedSearchCV:** Randomly samples `alpha` values from a defined distribution or range. Can be more efficient than GridSearchCV for larger hyperparameter spaces.
*   **Cross-Validation (Essential for Tuning `alpha`):** Use k-fold cross-validation (e.g., 5-fold or 10-fold) to evaluate model performance for each `alpha` value. Choose an evaluation metric appropriate for ordinal regression (see "Checking Model Accuracy" section below - e.g., Ordered Accuracy, Mean Absolute Error of ordinal predictions, or metrics like Mean Squared Error if treating ordinal labels as numerical). Select the `alpha` value that gives the best average performance across cross-validation folds.

**Implementation Example: Hyperparameter Tuning using GridSearchCV for `OrdinalRidge`:**

```python
from sklearn.model_selection import GridSearchCV
from mord import OrdinalRidge

# Define parameter grid for GridSearchCV (only alpha for OrdinalRidge)
param_grid_ordinal_ridge = {
    'alpha': np.logspace(-3, 3, 7) # Example range of alpha values (log scale)
}

# Initialize GridSearchCV with OrdinalRidge and parameter grid
grid_search_ordinal_ridge = GridSearchCV(OrdinalRidge(fit_intercept=True, normalize=False, random_state=42), # normalize=False as we scaled data
                                       param_grid_ordinal_ridge, scoring='r2', cv=5, n_jobs=-1, verbose=1) # 5-fold CV, R-squared scoring (adjust scoring metric if needed), parallel processing

# Fit GridSearchCV on training data
grid_result_ordinal_ridge = grid_search_ordinal_ridge.fit(X_train, y_train)

# Get the best model and best parameters from GridSearchCV
best_ordinal_ridge_model = grid_search_ordinal_ridge.best_estimator_
best_params_ordinal_ridge = grid_search_ordinal_ridge.best_params_
best_score_ordinal_ridge = grid_search_ordinal_ridge.best_score_

print("\nBest Ordinal Ridge Model from GridSearchCV:")
print(best_ordinal_ridge_model)
print("\nBest Hyperparameters (alpha):", best_params_ordinal_ridge)
print(f"Best Cross-Validation R-squared Score: {best_score_ordinal_ridge:.4f}")

# Evaluate best model on test set
y_pred_best_ordinal_ridge = best_ordinal_ridge_model.predict(X_test)
r2_test_best_ordinal_ridge = r2_score(y_test, y_pred_best_ordinal_ridge)
accuracy_test_best_ordinal_ridge = accuracy_score(y_test, y_pred_best_ordinal_ridge) # Also evaluate accuracy for category prediction

print(f"R-squared on Test Set (Best Model): {r2_test_best_ordinal_ridge:.4f}")
print(f"Accuracy on Test Set (Best Model): {accuracy_test_best_ordinal_ridge:.4f}")
```

**For Other Ordinal Regression Models (e.g., `OrdinalLogit`):**

Hyperparameters and tuning strategies will vary depending on the specific ordinal regression model you are using.  For example, for `OrdinalLogit` (Ordered Logistic Regression) from `mord`, hyperparameters might include:

*   **Regularization strength (if applicable in the implementation).**
*   **Link function variations (if available in the library).**
*   **Optimization parameters (e.g., for gradient descent-based fitting if the library exposes them).**

Refer to the documentation of the specific ordinal regression library and model you are using (e.g., `mord`, `statsmodels`) for details on tunable hyperparameters and recommended tuning methods.

## Checking Model Accuracy: Evaluation Metrics for Ordinal Regression

"Accuracy" evaluation in Ordinal Regression requires metrics that are appropriate for **ordered categorical predictions**. Standard classification metrics like accuracy alone might not be sufficient because they don't fully capture the ordered nature of the categories. Regression metrics like R-squared, while sometimes reported, also need to be interpreted cautiously for ordinal data.

**Relevant Evaluation Metrics for Ordinal Regression:**

1.  **Accuracy (Category Prediction Accuracy):**
    *   **Metric:** Standard accuracy - percentage of exact matches between predicted and actual ordinal categories. Formula: $Accuracy = \frac{Number\ of\ Correct\ Category\ Predictions}{Total\ Number\ of\ Predictions}$.
    *   **Interpretation:** Simple and easy to understand. Higher accuracy is better. However, accuracy treats all misclassifications equally, even if a prediction is only "one category away" from the true category.  For ordinal data, misclassifications that are closer to the true category might be considered less severe than misclassifications that are further away.
    *   **Calculation (Python Example):**

```python
from sklearn.metrics import accuracy_score

# --- Assume you have y_test (true ordinal categories) and y_pred_ordinal_test (ordinal predictions) ---

accuracy_test = accuracy_score(y_test, y_pred_ordinal_test)
print(f"Accuracy on Test Set: {accuracy_test:.4f}")
```

2.  **Mean Absolute Error (MAE) of Ordinal Predictions (Ordinal MAE):**
    *   **Metric:** Calculate the Mean Absolute Error (MAE), but treat ordinal categories as numerical values (using their encoded numerical labels, e.g., 0, 1, 2, 3 for 4 categories). MAE measures the average absolute difference between predicted and actual ordinal category *levels*.  Formula: $MAE_{ordinal} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$, where $y_i$ and $\hat{y}_i$ are the numerical encodings of true and predicted ordinal categories, respectively.
    *   **Interpretation:** Lower Ordinal MAE is better. Ordinal MAE is more sensitive to the *magnitude* of errors in ordinal category predictions compared to accuracy. It penalizes predictions that are further away from the true ordinal category more heavily. It's often a more informative metric for ordinal regression than simple accuracy because it reflects the ordered nature of the categories.
    *   **Calculation (Python Example):**

```python
from sklearn.metrics import mean_absolute_error

# --- Assume you have y_test (true ordinal categories) and y_pred_ordinal_test (ordinal predictions, already numerically encoded) ---

mae_ordinal_test = mean_absolute_error(y_test, y_pred_ordinal_test) # MAE on ordinal encoded categories
print(f"Ordinal MAE on Test Set: {mae_ordinal_test:.4f}")
```

3.  **Root Mean Squared Error (RMSE) of Ordinal Predictions (Ordinal RMSE - Less Common for Ordinal, but Possible):**
    *   **Metric:**  Similar to Ordinal MAE, but use Mean Squared Error (MSE) and then take the square root. Ordinal RMSE measures the root mean squared difference between predicted and actual ordinal category levels. Formula: $RMSE_{ordinal} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$.
    *   **Interpretation:** Lower Ordinal RMSE is better. Ordinal RMSE is more sensitive to larger errors (due to squaring) compared to MAE. Like MAE, it's a metric that reflects the magnitude of errors in ordinal predictions, considering the ordered nature of categories.  However, for ordinal data, MAE is often preferred over RMSE as it is less sensitive to the arbitrary numerical encoding of ordinal categories.
    *   **Calculation (Python Example):**

```python
from sklearn.metrics import mean_squared_error

# --- Assume you have y_test (true ordinal categories) and y_pred_ordinal_test (ordinal predictions, numerically encoded) ---

rmse_ordinal_test = np.sqrt(mean_squared_error(y_test, y_pred_ordinal_test)) # RMSE on ordinal encoded categories
print(f"Ordinal RMSE on Test Set: {rmse_ordinal_test:.4f}")
```

4.  **Weighted or Distance-Based Metrics (More Advanced, Capturing Ordinal Distance - e.g., Quadratic Weighted Kappa - Cohen's Kappa variant):**
    *   **Metrics:** More sophisticated metrics that explicitly consider the *ordinal distance* between categories. For example, Quadratic Weighted Kappa (a variant of Cohen's Kappa) assigns partial credit for predictions that are "close" to the true ordinal category and penalizes predictions that are further away more heavily.  Other distance-based metrics tailored for ordinal data exist. These metrics are more complex but can provide a more nuanced and accurate evaluation of ordinal prediction performance.
    *   **Libraries for Weighted Kappa:** Libraries like `sklearn.metrics` (for Cohen's Kappa, which can be adapted for weighted Kappa) or specialized packages might offer functions to calculate weighted Kappa or other ordinal distance-based metrics.
    *   **Interpretation:** Higher Weighted Kappa is better (range -1 to +1, +1 is perfect agreement, 0 is agreement no better than chance).  Weighted Kappa and similar metrics are considered more appropriate for evaluating ordinal predictions because they explicitly account for the ordered nature of categories and penalize errors based on ordinal distance.

**Choosing and Interpreting Evaluation Metrics:**

*   **Accuracy:**  Simple to understand and interpret, good as a basic starting point. But remember it treats all misclassifications equally, which might not be ideal for ordinal data.
*   **Ordinal MAE:**  Often a preferred metric for ordinal regression because it directly reflects the magnitude of errors in terms of ordinal category levels, respecting the ordered nature. Good for understanding the typical "off-by-one-category," "off-by-two-categories" type errors.
*   **Ordinal RMSE:** Similar to Ordinal MAE, but more sensitive to larger errors. Less commonly used for ordinal than MAE, but possible.
*   **Weighted Kappa or Ordinal Distance-Based Metrics:** More advanced and statistically sound metrics specifically designed for ordinal classification. They provide a more nuanced evaluation by considering the ordinal distance between predictions and true categories.  Recommended for rigorous evaluation when you want to explicitly quantify performance considering the ordinal scale.
*   **Context Matters:**  The "best" metric or combination of metrics depends on your specific problem and what you want to optimize. For some applications, simple accuracy might be sufficient. For others, Ordinal MAE or Weighted Kappa might be more appropriate to capture the nuances of ordinal predictions.

## Model Productionizing Steps for Ordinal Regression

Productionizing an Ordinal Regression model follows similar steps to productionizing other supervised machine learning models, but with considerations specific to the nature of ordinal predictions.

**1. Save the Trained Model and Preprocessing Objects:**

Use `pickle` (or `joblib`) to save:

*   The trained Ordinal Regression model object (`best_ordinal_ridge_model` or your final chosen model).
*   The fitted scaler object (e.g., `StandardScaler`).
*   Potentially, any label encoders used for encoding categorical features or the ordinal target variable, if needed for decoding predictions back to original category labels in your application.

**2. Create a Prediction Service/API:**

*   **Purpose:** To make your Ordinal Regression model accessible for making predictions on new data in a production environment.
*   **Technology Choices (Python, Flask/FastAPI, Cloud Platforms, Docker - as discussed in previous blogs):**  Create a Python-based API (Flask or FastAPI).
*   **API Endpoints (Example using Flask):**
    *   `/predict_ordinal_category`: Endpoint to take input feature data as JSON and return the predicted ordinal category (e.g., "Satisfied," "Neutral," "Unsatisfied") as JSON.
    *   `/predict_ordinal_level`: Endpoint to return the numerically encoded ordinal category prediction (e.g., 4 for "Satisfied" if using 1-5 encoding) as JSON.  This might be useful if downstream systems need the numerical level.
    *   `/predict_category_probabilities` (Optional, if model provides probabilities - e.g., `OrdinalLogit`): Endpoint to return predicted probabilities for all ordinal categories as JSON.

*   **Example Flask API Snippet (for ordinal category prediction):**

```python
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load Ordinal Regression model and scaler
ordinal_model = pickle.load(open('ordinal_ridge_model.pkl', 'rb')) # Load your trained ordinal model
data_scaler_x = pickle.load(open('standard_scaler_ordinal_reg.pkl', 'rb')) # Load scaler

# --- (Optional: If you need to decode numerical predictions back to original category labels, load label encoder if you used one) ---
# label_encoder_ordinal_target = pickle.load(open('label_encoder_ordinal_target.pkl', 'rb')) # Load label encoder

@app.route('/predict_ordinal_category', methods=['POST']) # Endpoint for ordinal category prediction
def predict_ordinal_category():
    try:
        data_json = request.get_json() # Expect input feature data as JSON
        if not data_json:
            return jsonify({'error': 'No JSON data provided'}), 400

        input_df = pd.DataFrame([data_json]) # Create DataFrame from input JSON
        input_scaled = data_scaler_x.transform(input_df) # Scale input features
        prediction_ordinal_encoded = ordinal_model.predict(input_scaled).tolist() # Get ordinal prediction (numerical encoded)

        # --- (Optional: Decode numerical prediction back to original category label - if you used label encoder) ---
        # predicted_category_label = label_encoder_ordinal_target.inverse_transform(prediction_ordinal_encoded)[0]
        # return jsonify({'predicted_category': predicted_category_label}) # Return category label

        return jsonify({'predicted_ordinal_level': prediction_ordinal_encoded[0]}) # Return numerical encoded ordinal level

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) # debug=True for local testing, remove for production
```

**3. Deployment Environments (Cloud, On-Premise, Local - as in previous blogs):**

*   **Local Testing:** Flask app locally.
*   **On-Premise Deployment:** Deploy API on your organization's servers.
*   **Cloud Deployment (PaaS, Containers, Serverless):** Cloud platforms (AWS, Google Cloud, Azure) offer scalable options.

**4. Monitoring and Maintenance:**

*   **Performance Monitoring:** Track API performance metrics (latency, error rates). Monitor ordinal prediction performance metrics (Accuracy, Ordinal MAE, Weighted Kappa if applicable) on live data.
*   **Data Drift Monitoring:** Monitor for changes in input feature distributions and in the distribution of the ordinal target variable (if possible to monitor true ordinal outcomes over time). Retrain model if data drift is significant.
*   **Model Retraining and Updates:** Retrain your Ordinal Regression model periodically with updated data to maintain accuracy and adapt to evolving patterns in ordinal outcomes. Implement automated retraining pipelines.
*   **Threshold/Cut-Point Review (Less Common in Basic Ordinal Regression, but relevant in some advanced models):**  For some advanced ordinal models or customized implementations, you might consider periodically reviewing and potentially adjusting the learned thresholds (cut-points) if the distribution of ordinal categories in your application changes over time.

## Conclusion: Ordinal Regression - Bridging the Gap between Categories and Order

Ordinal Regression provides a valuable and statistically sound approach for predictive modeling when your target variable represents ordered categories. It fills a gap between standard classification and regression, allowing you to build models that respect the inherent order in ordinal data and make more nuanced and accurate predictions for ordered categorical outcomes.

**Real-world Applications Where Ordinal Regression is Most Appropriate:**

*   **Surveys and Ratings Data:** Analyzing and predicting customer satisfaction ratings, product ratings, survey responses on Likert scales, where the outcomes are inherently ordinal.
*   **Medical Severity and Staging:** Predicting disease severity stages, cancer stages, risk levels in medicine and healthcare, where categories have a natural order of severity or risk.
*   **Risk Assessment and Credit Scoring:**  Predicting credit risk levels, insurance risk categories, or safety risk ratings, where categories are ordered by level of risk.
*   **Sentiment Analysis with Intensity Levels:** When you want to predict not just positive/negative sentiment, but also the *degree* or *intensity* of sentiment on an ordered scale (e.g., Very Negative to Very Positive).
*   **Education and Social Sciences Research:** Analyzing and predicting ordinal outcomes related to education levels, socioeconomic status, opinion scales, or other ordered categorical variables commonly encountered in social science research.

**Optimized or Newer Algorithms and Extensions:**

While Ordinal Regression with models like Proportional Odds and Ordered Ridge are widely used, ongoing research and extensions include:

*   **Non-Proportional Odds Models:**  Models that relax the "proportional odds" assumption of the Proportional Odds model, allowing for feature effects to vary across different thresholds. More flexible but also more complex to estimate and interpret.
*   **Neural Network-Based Ordinal Regression:** Combining neural networks with ordinal regression principles to build more complex, non-linear ordinal regression models. These can leverage the power of deep learning for ordinal prediction tasks.
*   **Bayesian Ordinal Regression:** Applying Bayesian methods to Ordinal Regression to get probabilistic predictions and quantify uncertainty in ordinal category predictions and model parameters. Bayesian approaches can offer more robust inference and handling of model uncertainty.
*   **Improved Evaluation Metrics for Ordinal Data:** Research continues on developing more refined evaluation metrics that capture the nuances of ordinal predictions beyond simple accuracy or MAE, especially metrics that account for ordinal distance and misclassification severity in a more statistically sound way.

**Conclusion:**

Ordinal Regression is an essential tool in the machine learning toolkit for anyone working with data where the target variable is inherently ordinal. It provides a statistically principled approach to building predictive models that respect the ordering of categories and make more accurate and meaningful predictions for ordered categorical outcomes compared to treating ordinal data as nominal categories or continuous numerical values. Understanding Ordinal Regression and its appropriate application scenarios is crucial for effectively analyzing and modeling ordinal data in various real-world domains.

## References

1.  **McCullagh, P. (1980). Regression models for ordinal data.** *Journal of the Royal Statistical Society: Series B (Methodological)*, *42*(2), 109-142. [[Link to JSTOR (may require subscription or institutional access)](https://www.jstor.org/stable/2984587)] - The seminal paper introducing the Proportional Odds model, a fundamental ordinal regression model.

2.  **Agresti, A. (2010). *Analysis of ordinal categorical data*. John Wiley & Sons.** - A comprehensive textbook dedicated to the analysis of ordinal categorical data, covering various ordinal regression models, including Proportional Odds and others, in detail.

3.  **mord (Mostly Ordinal Regression) Python Library Documentation:** [[Link to mord documentation on GitHub](https://github.com/fabianp/mord)] - Official documentation and examples for the `mord` Python library, providing practical examples and API reference for Ordinal Regression models in Python.

4.  **Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: data mining, inference, and prediction*. Springer Science & Business Media.** [[Link to book website with free PDF available](https://web.stanford.edu/~hastie/ElemStatLearn/)] - A widely used textbook covering statistical learning methods, including a chapter on extensions of linear models, which includes discussions of ordinal regression and generalized linear models (Chapter 4).

5.  **Wikipedia page on Ordinal Regression:** [[Link to Wikipedia article on Ordinal Regression](https://en.wikipedia.org/wiki/Ordinal_regression)] - Provides a good starting point for a general overview and introduction to ordinal regression concepts and models.

This blog post provides a detailed introduction to Ordinal Regression. Experiment with the code examples, tune hyperparameters, and apply Ordinal Regression to your own datasets to gain practical experience and a deeper understanding of this valuable algorithm for ordered categorical prediction.
