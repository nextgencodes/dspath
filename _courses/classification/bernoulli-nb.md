---
title: "Bernoulli Naive Bayes: A Simple Guide to Binary Classification"
excerpt: "Bernoulli Naive Bayes Algorithm"
# permalink: /courses/classification/bernoulli-nb/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
  - Classification
tags: 
  - Machine Learning
  - Classification Model
---

{% include download file="bernoulli_nb.ipynb" alt="download bernoulli naive bayes code" text="Download Code" %}

## Introduction: What is a Bernoulli Naive Bayes?

Welcome! Today, we'll be exploring a powerful yet easy-to-understand machine learning algorithm called Bernoulli Naive Bayes. If you're new to machine learning, don't worry – we'll take it step by step.

## Introduction: What is Bernoulli Naive Bayes and Why Should You Care?

Imagine you have a bunch of emails, and you want to automatically sort them into "spam" or "not spam."  Or perhaps you want to classify user reviews as "positive" or "negative." This is where classification algorithms come into play, and Bernoulli Naive Bayes is an excellent choice for these types of *binary classification* problems. 

**Binary classification** simply means categorizing things into one of two categories (like yes/no, true/false, etc.).

Bernoulli Naive Bayes is a simplified version of the Naive Bayes algorithm (there are others like Gaussian and Multinomial Naive Bayes) that’s particularly well-suited when your data is represented as **binary features**. Think of it as a checklist: either a feature is present (1) or absent (0) for a given data point. 

Here are some real-world applications:

*   **Spam Detection:** Does an email contain certain keywords like "free," "money," or "urgent"?
*   **Sentiment Analysis:** Does a customer review contain words that suggest positive or negative sentiment?
*   **Document Classification:** Does a document contain keywords that indicate it's about a particular topic?
*   **Medical Diagnosis:** Does a patient have symptoms that suggest a specific disease (present/absent)?

The beauty of Bernoulli Naive Bayes lies in its **simplicity** and **speed**, making it a great starting point for many classification tasks.

## The Math Behind the Algorithm: Probability and Assumptions

Let's get a little mathematical, but don't worry, it won't be too scary. The algorithm is based on **Bayes' Theorem**, which tells us how to update our beliefs when new evidence arrives.

### Bayes' Theorem

Bayes' Theorem is expressed as:

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$

Where:
*   `P(A|B)`: This is the **posterior probability**. It is the probability of event `A` happening given that event `B` has already happened. (What we are trying to find out)
*   `P(B|A)`: This is the **likelihood**. It is the probability of event `B` happening given that event `A` has happened.
*   `P(A)`: This is the **prior probability**. It is the probability of event `A` happening independently.
*   `P(B)`: This is the **evidence** or **marginal probability**. It is the probability of event `B` happening.

**Example:** Imagine you're trying to determine if a person has a disease (event A) given they have a specific symptom (event B).

*   `P(A|B)`:  The probability of having the disease given the symptom is present. (What we want to know).
*   `P(B|A)`: The probability of having the symptom given that the person has the disease. 
*   `P(A)`: The probability of having the disease in the population.
*   `P(B)`: The probability of having the symptom in the population.

### Bernoulli Naive Bayes Simplification

Bernoulli Naive Bayes applies Bayes' Theorem to the problem of classification. We want to find the probability of a data point belonging to a class given its features. Here are the simplifications that make this specific Bayes algorithm called "Naive":

1. **Feature Independence:**  It assumes that all features are independent of each other, given the class.  This is a simplification that rarely holds true in real-world data, but it makes calculations a lot easier and often performs well despite this assumption (hence "naive").
2.  **Binary Features:** Bernoulli Naive Bayes specifically works with binary features, each having two options either 0 or 1 (absent/present). We do not consider the number of times a feature is present.

For a document with features (words) \( x_1, x_2, ..., x_n \), the probability of it belonging to a class 'c' is given by:
$$P(c | x_1, x_2, ..., x_n) \propto P(c) \prod_{i=1}^{n} P(x_i|c)$$

Where:
*   \( P(c|x_1, x_2, ..., x_n) \) is the posterior probability of the document belonging to class 'c' given features.
*   \( P(c) \) is the prior probability of class 'c'.
*   \( P(x_i|c) \) is the likelihood, or the probability of feature \( x_i \) being present given the class is 'c'. 
    *   If the feature \( x_i \) is present in document, it has a value of 1. Then we use  \( P(x_i=1|c) \).
    *   If the feature \( x_i \) is absent from document, it has a value of 0. Then we use \( P(x_i=0|c) = 1 - P(x_i=1|c) \)

**Example:**

Suppose we want to classify emails as spam or not spam. Our features might be the presence of the words "free", "money," and "urgent."

*   **Class (c):** Spam or Not Spam.
*   **Features (x):** `x1` (word "free"), `x2` (word "money"), `x3` (word "urgent"). Each feature is either 1 (present) or 0 (absent) in the email.

The Bernoulli Naive Bayes algorithm calculates the probability of the email belonging to the "spam" class (or "not spam") using the equation above and then picks the class with the highest probability. 

## Prerequisites and Preprocessing

Before using Bernoulli Naive Bayes, let's cover some essential prerequisites:

### Assumptions
The main assumption of Bernoulli Naive Bayes is the **conditional independence** of features. This implies, given the class label, features are independent of each other.
* **Checking for Assumption:** This assumption is generally not testable, but rather it's something we must be mindful of. If features are strongly correlated (e.g., multiple very similar words), the naive Bayes assumption is violated.
* **Violation of Assumption:** When assumption is violated, it may lead to underperformance. You should consider other algorithms or feature engineering. 

### Prerequisites
1.  **Binary Features:** The features must be binary (0 or 1, present or absent). If the data contains features with multiple discrete values, or real values, this algorithm will not work as is.
2.  **Categorical Target Variable:** Target variable must be categorical variable. For this algorithm, it needs to be binary category.
3. **Understanding of Probability:** You need to have basic understanding of probability concept, especially Bayes theorem.

### Python Libraries

*   **Scikit-learn (`sklearn`):** This is the workhorse for machine learning in Python and provides the `BernoulliNB` class for our algorithm.
*   **NumPy:**  Used for numerical computations and array manipulation.
*   **Pandas:**  Used for data handling and manipulation.

**Code Example of Required Libraries:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
import joblib # To save and load the model
```

## Data Preprocessing

Data preprocessing is an essential step to ensure the performance of the machine learning algorithm. Let's see what needs to be done for the Bernoulli Naive Bayes.

### Required Preprocessing

1.  **Binary Feature Conversion:** If your data isn't already in a binary format, you'll need to transform it. Here are some common ways:

    *   **Text Data (Bag of Words):** Convert your text data into a matrix where each row represents a document (e.g., an email or review), and each column represents a word or "token." The cells contain either 0 or 1, depending on whether the word appears in the document. 
        *   **Code Example:**
            ```python
            from sklearn.feature_extraction.text import CountVectorizer
            
            documents = ["this is the first document", "this document is the second", "and this is the third one"]
            vectorizer = CountVectorizer(binary=True)
            X = vectorizer.fit_transform(documents)
            print(vectorizer.vocabulary_)
            print(X.toarray())
            ```
            In this example, all document has either `1` or `0` in the matrix based on the presence of the words in the vocabulary of training corpus.

    *   **Categorical Features:** If you have categorical features, you'll need to one-hot encode them and then convert the resulting features to binary. For example, if the categorical features has value `yes` and `no`, it can be converted into 1 or 0.

    *   **Numerical data:** The numerical data needs to be converted to a binary data. For example, a specific medical test can have result as `positive` or `negative` or any continuous value, but this should be converted to either 1 or 0 based on the threshold set. This step is very important because Bernoulli Naive Bayes only accepts binary data.

2.  **Missing Value Handling:**  Decide how to deal with missing values, as these are often denoted with NaN and will not be binary. Options are to impute with 0, remove them, or use a different imputation strategy.

### When Preprocessing Can Be Ignored

*   **Already Binary Data:** If your data is already in binary format (e.g., a presence/absence checklist), you can skip conversion.

### Why Specific Preprocessing Is Required?

Bernoulli Naive Bayes uses probability calculation based on the features. The presence (1) or absence (0) of the features directly affects the calculation. It **can not process** numerical values like 0.5 or 10. If the values are not binary, then this will break the model. The model makes its prediction based on the multiplication of probabilities for each feature in the document or the record. For the algorithm to work properly, each of the data point must have binary values.

## Implementation Example

Let's implement Bernoulli Naive Bayes using a dummy dataset. We will create a dataframe with spam or not spam tag as a target variable. The features will be the presence of the words in the email.

```python
# Create a sample dataset
data = {
    'email': [
        "free money urgent offer",
        "hello how are you",
        "get free gift now",
        "meeting scheduled for today",
        "urgent action needed",
        "let's have a coffee",
        "claim your reward",
        "check my document attached"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
}
df = pd.DataFrame(data)

# Convert emails to binary feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['email'])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the model
joblib.dump(model, 'bernoulli_naive_bayes_model.pkl')

# Load the model
loaded_model = joblib.load('bernoulli_naive_bayes_model.pkl')

# Predict using the loaded model
y_pred_loaded = loaded_model.predict(X_test)
print(f"Prediction using loaded model: {y_pred_loaded}")
```

**Explanation of Output:**

*   **Accuracy:**  This is the percentage of correctly classified examples (emails). In this case, it tells you how well the model classifies emails as spam or not spam. For example, 1.0 accuracy indicates 100% accuracy.
*   **Classification Report:** This provides a detailed breakdown of the performance:
    *   **Precision:**  The ratio of true positives to all predicted positives. It tells us how many of the emails classified as spam were actually spam.
    *   **Recall:** The ratio of true positives to all actual positives. It tells us how many of the total actual spam emails did the model correctly identify.
    *   **F1-Score:**  The harmonic mean of precision and recall. It provides a balance between precision and recall.
    *   **Support:** The number of actual occurrences of the class in the test set.

*   **Saving and Loading the Model:** This demonstrates how to save a trained model and load it for later use using `joblib`.

## Post-Processing and Feature Importance

Once your model is trained, you might want to do some post-processing:

1.  **Feature Importance:** In the context of text data, you can examine which words have the highest impact on the classification. In Bernoulli Naive Bayes, these are the words with large \(P(x_i|c)\) values, for the different classes.
    *   **Code Example:**
        ```python
        feature_log_prob = model.feature_log_prob_ #probability of each feature
        feature_names = vectorizer.get_feature_names_out()
        for i, category in enumerate(model.classes_):
            print(f"Most significant words for class: {category}")
            indices = feature_log_prob[i,:].argsort()[::-1][:5] #top 5 most significant features
            print(feature_names[indices])
        ```
        This gives top 5 significant words for each class based on the probability.
2. **Hypothesis Testing:** if there are multiple models, you can use hypothesis testing to check if the accuracy is significantly better for one model than the other.
    *   **Example:** You can use a paired t-test or a chi-squared test for comparison between classification results.

## Hyperparameter Tuning

Bernoulli Naive Bayes has a few hyperparameters that you can adjust:

1.  **`alpha` (Laplace smoothing):**  This is a smoothing parameter that helps handle unseen words or features during the test set. Adding 1 to the word count of each feature in every class before calculating probability prevents zero probabilities, especially if some words do not appear in the training corpus. This technique is also known as Laplace correction or add-1 smoothing. 
    *   **Effect:** Higher `alpha` values lead to more smoothing, potentially preventing overfitting but could also lead to underfitting. Default value is `1`. Smaller value will give model better training accuracy, but could also potentially lead to overfitting.
    *   **Tuning:** Use grid search or random search to explore different values (e.g., `[0.1, 0.5, 1.0, 2.0]`)
2. **`binarize`:** This is the threshold for converting continuous data to binary data. The default value is `0.0`.
    *   **Effect:** This is very important when dealing with continuous data, and you want to convert to binary before using this algorithm. If the number is greater than threshold value, it will be 1 otherwise it will be 0.

**Code Example for Hyperparameter Tuning:**

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}

# Perform grid search with cross validation
grid_search = GridSearchCV(BernoulliNB(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Train the final model with the best parameters
best_model = BernoulliNB(**grid_search.best_params_)
best_model.fit(X_train, y_train)
```

## Model Evaluation

Evaluating the model is crucial to know how well it is performing. You can use below accuracy matrics to evaluate:

1.  **Accuracy:** As we discussed earlier, this is the proportion of correct classifications.
    
    $$ Accuracy = \frac{Number \ of \ Correct \ Predictions}{Total \ Number \ of \ Predictions} $$

2.  **Precision, Recall, F1-Score:** These give a detailed breakdown of per-class performance, as explained before.

3.  **Confusion Matrix:** A table showing true positives, false positives, true negatives, and false negatives. It helps you understand the types of errors the model is making.

4.  **Cross Validation:** Split training dataset into `k` sets, and perform train and test on `k-1` training set, and `1` test set. This helps in model evaluation without making model overfit on the train data. Scikit-learn provide api for implementing cross validation.

    * **Code Example**
    ```python
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=5) # 5 fold cross validation
        print(f"Cross validation scores: {scores}")
        print(f"Average cross validation score: {np.mean(scores)}")
    ```

## Model Productionization

Once you're satisfied with your model's performance, here's how to get it into production:

1.  **Cloud Deployment (Example: AWS):**

    *   **Save Model:** As shown previously, save the trained model.
    *   **Create API:**  Use a framework like Flask or FastAPI to create an API that loads the model and accepts new data for predictions.
    *   **Deploy:** Deploy this API to a service like AWS Elastic Beanstalk, Lambda, or SageMaker.
2.  **On-Premises Deployment:**
    *   **Package:** Package your model and API into a container (e.g., Docker).
    *   **Deploy:** Deploy the container to a server on your infrastructure.
3. **Local Testing:**
    *   **Local Server:** Run your API on a local server for testing using Python.

**Code Example of Local API using Flask:**

```python
from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('bernoulli_naive_bayes_model.pkl')
vectorizer = CountVectorizer(binary=True) #recreate vectorizer for transforming test data
texts = ["free money urgent offer", "hello how are you", "get free gift now", "meeting scheduled for today", "urgent action needed", "let's have a coffee", "claim your reward", "check my document attached"]
vectorizer.fit(texts) # need to fit on training corpus to create vocabulary

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```
    This example shows how to host a local server which can take text input and return 0 or 1, which is prediction of the model.

## Conclusion

Bernoulli Naive Bayes is a simple, fast, and effective classification algorithm, especially useful for binary classification with binary features. It's widely used in areas like spam detection, sentiment analysis, and medical diagnosis. While it does have the strong assumption of feature independence, it often delivers surprisingly good results.  

While many newer and optimized algorithms exist (such as more complex deep learning models), Bernoulli Naive Bayes continues to be valuable for its interpretability and efficiency in specific applications. For example, some newer algorithms are computationally very expensive and needs good amount of memory and infrastructure to train the model. For the small to medium dataset, this algorithm can be the starting point to build any classification model.

## References

1.  [Scikit-learn Documentation on Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
2.  [Wikipedia: Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
