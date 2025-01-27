---
title: "Fisher Linear Discriminant Analysis: A Comprehensive Guide"
excerpt: "Fisher's Linear Discriminant Algorithm"
# permalink: /courses/classification/fsd/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
tags: 
  - Machine Learning
  - Classification Model
---


{% include download file="fisher_lda.ipynb" alt="download fisher lda code" text="Download Code" %}

## Introduction to Fisher Linear Discriminant Analysis (LDA)

Imagine you have a collection of objects, and you want to sort them into different groups based on their features. For example, you might want to classify emails as either "spam" or "not spam" based on the words they contain, or distinguish between different types of flowers based on their petal length and width. That's where Linear Discriminant Analysis (LDA) comes in handy!

LDA is a powerful and classic technique used in machine learning and statistics for **classification** and **dimensionality reduction**. The core idea is to find a way to project your data (objects described by features) into a lower-dimensional space while **maximizing the separation between different classes**. In simple terms, it tries to find the best line (or hyperplane in higher dimensions) that separates your groups so that they are as far away from each other as possible.

**Real-world examples:**
*   **Medical Diagnosis:** LDA can be used to classify patients into different disease groups based on their symptoms and test results.
*   **Face Recognition:** It can help identify individuals by analyzing the features extracted from their facial images.
*   **Customer Segmentation:** Businesses can use LDA to group their customers into different segments based on their buying behavior and demographics, enabling personalized marketing.
*   **Natural Language Processing:**  Classifying documents into categories (e.g. sports, politics, or technology) based on the words they contain.
*   **Fraud detection:** Identifying fraudulent transactions by analyzing user behavior patterns.

## The Mathematics Behind LDA

Now, let's dive into the math. Don't worry, we'll break it down so that everyone can understand it.

The primary goal of LDA is to find a **projection vector** or a **line** (usually represented by 'w') that maximizes the separation between the means of different classes while minimizing the variance (or spread) within each class, after projecting the data onto that vector/line. This means we want the means of projected classes to be as far apart as possible, and the data points within each projected class to be as close to their means as possible.

**Key Concepts:**

*   **Between-class Scatter Matrix (Sb):** This matrix measures how spread apart the means of your different classes are. If class means are far from each other, Sb will be large.
*   **Within-class Scatter Matrix (Sw):** This matrix measures how scattered the data points are within each class. If data points are far from their class means, Sw will be large.

**The Objective Function:**

LDA aims to maximize the following:

  $$ J(w) = \frac {wᵀ S_b w} {wᵀ S_w w} $$

Where:

*   `w` is the projection vector that we are looking for.
*   `S_b` is the between-class scatter matrix, as mentioned above.
*   `S_w` is the within-class scatter matrix.
*   `wᵀ` is the transpose of w vector.

**Simple Explanation of the equation:**
The numerator (wᵀ S_b w) represents the variance between the classes, and the denominator (wᵀ S_w w) represents the variance within the classes. So, the equation is trying to maximize the variance between classes while minimizing the variance within classes. In other words, it's trying to find a projection line 'w' that will separate classes better.

**How to calculate Sb and Sw**
Let's assume we have 'k' number of classes and 'N' be the number of samples across all the classes.
*   **Overall mean vector:** $$ μ = (1/N) Σ(xᵢ) $$, where the summation is done across all the samples xᵢ.
*   **Class specific mean vectors:**  $$ μⱼ = (1/Nⱼ) Σ(xᵢ) $$, where the summation is done across samples belonging to the j-th class, and Nⱼ is the total number of samples in class j.
*   **Between class scatter matrix (S_b):**  $$ S_b =  Σⱼ Nⱼ(μⱼ - μ)(μⱼ - μ)ᵀ $$ , where j ranges from 1 to k.
*   **Within class scatter matrix (S_w):** $$ S_w = Σⱼ Σᵢ(xᵢ - μⱼ)(xᵢ - μⱼ)ᵀ $$, where j ranges from 1 to k and i ranges over the samples belonging to the jth class.

**Finding the optimal w:**

The projection vector `w` that maximizes the objective function `J(w)` can be found by solving a generalized eigenvalue problem which has the solution as follows:
$$ S_b w = λ S_w w $$

Where λ represents the eigenvalues and w represents the corresponding eigenvectors. The eigenvector associated with the largest eigenvalue will give you the direction in feature space that maximizes the class separation.
If `S_w` is invertible, we can multiply both sides of the equation by the inverse of `S_w`, denoted by S_w⁻¹, and we get
$$ S_w⁻¹ S_b w = λw $$
Here $$ S_w⁻¹ S_b $$ is a matrix and the above equation now resembles an Eigen value equation.

<a title="Amélia O. F. da S., CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:LDA_Projection_Illustration_01.gif"><img width="512" alt="An animation depicting the projection of 2D points onto an LDA axis" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/LDA_Projection_Illustration_01.gif?20220605211907"></a>

_An illustration of LDA. Notice the original data points on the left are difficult to classify using a vertical line and how the same data points are easier to classify on the right, after projection._
## Prerequisites and Assumptions

Before using LDA, it's important to understand the assumptions it makes and the preprocessing steps that may be necessary:

*   **Assumptions:**
    *   **Normality:** The data within each class is assumed to follow a multivariate normal distribution (a bell curve in multiple dimensions). This assumption is often violated in real-world data. However, LDA is somewhat robust to deviations from normality, but it's something to be aware of.
    *   **Equal Covariance Matrices:** All classes should have the same covariance matrix. This means that the spread and shape of the data within each class is similar. If this is violated significantly, LDA might not perform optimally.
    *   **Linear Separability:** LDA assumes the classes are linearly separable (they can be separated by a straight line or a hyperplane in higher dimensions). If the classes are highly non-linear, LDA may not be the best choice.

*   **Testing Assumptions:**
    *   **Normality:** You can visually inspect the data using histograms or Q-Q plots. Statistical tests like the Shapiro-Wilk test can also be used to test normality but keep in mind that these tests are sensitive to sample size.
    *   **Equal Covariance Matrices:** Box's M test can be used to test the equality of covariance matrices between groups, however, this test is sensitive to deviations from normality.
*   **Python Libraries:**
    *   `scikit-learn` (`sklearn`):  Provides the `LinearDiscriminantAnalysis` class for implementation.
    *   `numpy`: For numerical computation and array handling.
    *   `pandas`: For data manipulation.

## Data Preprocessing

LDA is sensitive to the scale of features because it relies on calculating variances. So data preprocessing is important.
*   **Feature Scaling:** It's a good idea to scale your data, particularly when features have different scales (e.g., height in meters and weight in kilograms). Standardization (subtracting the mean and dividing by the standard deviation) is a common choice. Without scaling, features with larger scales will disproportionately affect the algorithm.
*   **Why Standardization?** LDA uses covariance matrices.  If your feature values are very large, their contribution to the covariance matrix will also be large. Standardization makes sure that each feature has the same amount of influence on calculating covariance matrices.
*   **Cases where scaling might be ignored:** If your features are already on similar scales, you might get away without scaling them. For example, if your features are all proportions or percentages, scaling might not be necessary.
    *   **Example:**
    Consider a dataset of house prices with two features: area in square feet (ranging from 500 to 5000) and number of bedrooms (ranging from 1 to 5). Without feature scaling, the area will dominate the LDA computation due to its larger range and variance. After standardization, both the area and the number of bedrooms will have the same scale, allowing them to contribute equally to the analysis.

## Implementation Example

Here's how to implement LDA with Python using scikit-learn:

```python
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Generate some dummy data
np.random.seed(42)
n_samples = 150
features = 2
class_0_samples = n_samples // 3
class_1_samples = n_samples // 3
class_2_samples = n_samples - class_0_samples - class_1_samples

X_class0 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.2], [0.2, 1]], size=class_0_samples)
X_class1 = np.random.multivariate_normal(mean=[3, 3], cov=[[1, -0.2], [-0.2, 1]], size=class_1_samples)
X_class2 = np.random.multivariate_normal(mean=[6, 1], cov=[[1, 0], [0, 1]], size=class_2_samples)

X = np.vstack((X_class0, X_class1, X_class2))
y = np.array([0] * class_0_samples + [1] * class_1_samples + [2] * class_2_samples)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Apply LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = lda.predict(X_test_scaled)

# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the model and scaler for later
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

with open('scaler.pkl', 'wb') as f:
  pickle.dump(scaler, f)

# 8. Load the model and scaler for later
with open('lda_model.pkl', 'rb') as f:
    loaded_lda = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
# Now you can use loaded_lda and loaded_scaler for predictions
new_test_data = [[0.5, 0.5], [4.0, 2.8], [5.9, 0.8]]
new_test_data_scaled = loaded_scaler.transform(new_test_data)
new_test_pred = loaded_lda.predict(new_test_data_scaled)
print("prediction for the new data", new_test_pred)
```

**Output:**

```
Accuracy: 0.9555555555555556
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        12
           1       0.90      1.00      0.95        18
           2       1.00      0.89      0.94        15

    accuracy                           0.96        45
   macro avg       0.97      0.96      0.96        45
weighted avg       0.96      0.96      0.96        45

prediction for the new data [0 1 2]
```

**Explanation:**
*   **Data Generation:** We generate data samples of 3 classes, each distributed around a different mean using a multivariate normal distribution. We then split the data into training and testing sets.
*   **Scaling:** Before fitting the model, we standardize our training and test data with StandardScaler.
*   **LDA Model:** We create an LDA object and train the model using the scaled training data.
*   **Prediction:** We use the trained LDA model to predict classes on our scaled test data.
*   **Accuracy:** The output `accuracy` represents the ratio of correct predictions and the `classification report` presents other metrics like `precision`, `recall`, and `f1-score` (more on this below)
*   **Saving/Loading Model:** We saved both the LDA model object and the scaler object using pickle and also loaded and used them for prediction.

## Post-Processing

*   **Feature Importance:** After LDA, the projection vectors (`lda.coef_` attribute) can be used to understand which features contribute most to separating different classes. Each element in `lda.coef_[i]` will represent weight of feature on `i`th class. A larger magnitude (absolute value) corresponds to a larger contribution of that feature. However, keep in mind, the `lda.coef_` is generated by projecting the feature in a lower dimension (for binary class projection goes to one dimensional line). Thus, it is hard to evaluate how original features effect on class separation. Feature importance is not straightforward in LDA, like tree models where importance is defined as how much that feature contributes to the decision.
*   **Hypothesis Testing:** After finding out most impactful variables, to make sure that observed impact is statistically significant and not due to random chance, one can do A/B testing, or chi-squared test depending on nature of variables. For example, the two groups in A/B test can be two classes being discriminated.
*   **Variable Selection:** In case of many input features, LDA can also be used as a variable selection method. By selecting features which has highest weight in `lda.coef_`, the user can get a reduced dimension for the feature space, and can perform other analysis, like regression, other kind of classification, or cluster analysis etc.

## Tweakable Parameters and Hyperparameters

LDA has several tweakable parameters which are often set during model initialization. These can affect the model output, so understanding their role is useful.
*   **solver** (`svd`, `lsqr`, `eigen`): This determines the algorithm used for optimization.
    *   `svd` is the default option, generally works well in most cases, performs singular value decomposition.
    *    `lsqr` is more suited for data with large number of samples and features where computations are heavy, this method uses least square method.
    *   `eigen` finds the best line using Eigen value decomposition (mentioned above), the data is not directly computed, but solved by eigenvalue equation.
    *   The choice of `solver` may affect the speed of computation and may also affect model accuracy marginally.
*   **shrinkage** (`None` or `float`): This parameter adds regularization to the within-class scatter matrix. In other words, it adds value on the diagonal matrix of the `Sw`. It is useful when there are large number of features compared to number of samples and the covariance matrices may be singular.
    *   If `None`, no shrinkage is applied.
    *   A float between 0 and 1 will apply shrinkage. 0 means no shrinkage (same as none) and 1 means full shrinkage.
    *   With increase in `shrinkage`, the model will become less sensitive to outliers, and generalize better. But if shrinkage is too high, the model might not perform well in training data and result in poor prediction. `shrinkage` parameter has similar effect like regularization in regression algorithms.
*   **priors** (`None` or array-like): This parameter sets the prior probabilities of each class. By default, the prior probability is equal to number of samples of each class divided by total samples.
    *   If `None`, prior probabilities are determined by proportion of each class in training data.
    *   An array like object should be passed if user wants to explicitly set priors.
    *   Setting the priors will change the decision boundary, the classes with large prior will be classified more often.

**Hyperparameter Tuning:**
You can use `GridSearchCV` or `RandomizedSearchCV` from scikit-learn to tune the `solver`, `shrinkage`, and `priors` parameter. Here's an example of how to do this:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Define parameter grid
param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 0.1, 0.5, 0.9],
     'priors': [None, [0.2, 0.3, 0.5], [0.1, 0.3, 0.6]]
}

# Initialize LDA model
lda = LinearDiscriminantAnalysis()

# Initialize Grid Search
grid = GridSearchCV(lda, param_grid, cv=5, scoring='accuracy')

# Fit Grid Search to training data
grid.fit(X_train_scaled, y_train)

# Print best parameters and score
print("Best parameters found:", grid.best_params_)
print("Best score found:", grid.best_score_)
```

## Model Evaluation: Accuracy Metrics

Evaluating a classification model is very important step to see whether the model is performing well. In general, a confusion matrix is calculated, and different metrics are derived from the confusion matrix. Here are some common ones:

*   **Confusion Matrix:**  A table that shows how many samples were correctly or incorrectly classified. Each row represents the true 


   |                     | Predicted Positive | Predicted Negative |
   |---------------------|--------------------|--------------------|
   | **Actual Positive** | True Positive (TP) | False Negative (FN) |
   | **Actual Negative** | False Positive (FP) | True Negative (TN) |


*   **Accuracy:** The proportion of correct predictions out of all predictions.
   $$ Accuracy = (TP + TN) / (TP + TN + FP + FN) $$
*   **Precision (for positive class):**  The proportion of correctly predicted positive samples out of all samples predicted as positive.
    $$ Precision = TP / (TP + FP) $$
*   **Recall or Sensitivity (for positive class):** The proportion of correctly predicted positive samples out of all actual positive samples.
    $$ Recall = TP / (TP + FN) $$
*   **F1-score:** A harmonic mean of precision and recall.
    $$ F1_score = 2 * (Precision * Recall) / (Precision + Recall) $$
*   **Classification Report:** A report provided by libraries such as scikit-learn which displays all of these metrics in a readable manner. This can give you a good insight of which classes are not being predicted well.
*   **ROC curve and AUC:** ROC or Receiver operating characteristics curve plots the true positive rate (or recall) vs the false positive rate at all the thresholds. AUC or Area under the curve represents the area under the ROC curve. AUC represents how well the model distinguishes between the classes. A perfect classifier will have AUC = 1.

In our example output, the model is performing well with 0.96 average accuracy, and has high precision and recall for all the classes.
## Model Productionizing

Here's a basic outline of how to deploy an LDA model:

1.  **Local Testing:**
    *   Use the code above for local testing. Use `pickle` to load the model and the scaler into production.
    *   Ensure to scale the test input data using the same scaler object that was used for training the model.
    ```python
        # Load the model and scaler
        with open('lda_model.pkl', 'rb') as f:
            loaded_lda = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)

        # Prepare test data
        test_data = np.array([[1, 1], [3, 3], [6, 2]])

        #Scale the data using the scaler object
        test_data_scaled = loaded_scaler.transform(test_data)

        # Make predictions
        predictions = loaded_lda.predict(test_data_scaled)
        print("predictions:", predictions)
    ```
2.  **On-Premise Deployment:**
    *   Package your model as a Python library or a Docker container, and deploy it to your on-premise servers.
    *   Create API endpoints using frameworks like Flask or FastAPI that load the model and provide predictions.
        ```python
        from flask import Flask, request, jsonify
        import numpy as np
        import pickle
        import os
        app = Flask(__name__)

        # Load the model and scaler
        model_path = "lda_model.pkl"
        scaler_path = "scaler.pkl"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                loaded_lda = pickle.load(f)
            with open(scaler_path, 'rb') as f:
               loaded_scaler = pickle.load(f)
        else:
            raise Exception("model or scaler files are not present in current directory")

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            input_data = np.array(data['data'])
            input_data_scaled = loaded_scaler.transform(input_data)
            prediction = loaded_lda.predict(input_data_scaled).tolist()
            return jsonify({'prediction': prediction})


        if __name__ == '__main__':
            app.run(debug=True, host='0.0.0.0', port=5000)

        ```
        *   To access this API using a client, use requests
            ```python
                import requests
                import json

                url = 'http://127.0.0.1:5000/predict' #localhost address, if running the flask app locally

                data_to_send = {
                    "data" : [[1, 1], [3, 3], [6, 2]]
                }

                headers = {'Content-type': 'application/json'}

                response = requests.post(url, data=json.dumps(data_to_send), headers=headers)

                if response.status_code == 200:
                    print("response from server:", response.json())
                else:
                    print(f"Request failed with status code {response.status_code}")
            ```
3.  **Cloud Deployment:**
    *   Use cloud services like AWS, Azure, or Google Cloud to deploy your model as an API.
    *   Cloud services provide scalability, monitoring, and integration with other services.
    *   Use serverless platforms like AWS Lambda or Azure Functions for a cost-effective and scalable way to deploy your model.
    *   Use services like AWS SageMaker, Google AI platform or Azure ML for simplified ML deployment.

## Conclusion

Fisher Linear Discriminant Analysis (LDA) is a valuable technique for classification and dimensionality reduction. Its simplicity and interpretability make it a good choice in various scenarios.

**Real-world Applications:**
*   It is widely used for classification tasks in many fields such as medical diagnosis and image classification and signal processing.
*   It is used as a preprocessing step in many other algorithms.

**Alternatives and Advancements:**

*   **Quadratic Discriminant Analysis (QDA):** If the equal covariance assumption is violated, QDA can be a better option. However, this assumes that covariance matrix for each class is not same.
*   **Support Vector Machines (SVM):** SVMs are powerful classifiers but require careful hyperparameter tuning.
*   **Neural Networks:** Can handle more complex decision boundaries compared to linear models.
*   **Regularized LDA:** Some methods offer regularized LDA versions that reduce overfitting.

Despite being a classic algorithm, LDA remains relevant due to its computational efficiency and strong performance in many use cases.

## References

1.  Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic Problems". *Annals of Eugenics*. **7** (2): 179–188.
2.  Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: data mining, inference, and prediction*. Springer Science & Business Media.
3.  Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern classification*. John Wiley & Sons.
4.  Scikit-learn documentation for LinearDiscriminantAnalysis: [https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
