---
title: "Apriori Algorithm: Uncovering Hidden Relationships in Data"
excerpt: "Apriori Algorithm"
# permalink: /courses/association/apriori/
last_modified_at: 2025-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
tags: 
  - Machine Learning
  - Association Rule Mining
  - Data Mining
---

{% include download file="apriori.ipynb" alt="download apriori code" text="Download Code" %}


## Introduction: What is the Apriori Algorithm?

Imagine you're a store owner trying to understand what items people often buy together. If you knew this, you could place those items close to each other, or recommend them together to increase sales. The Apriori algorithm helps to find these "hidden patterns" or relationships in data.

The **Apriori algorithm** is a classic algorithm in data mining, specifically designed for **association rule mining**. It's used to find frequent itemsets in a dataset. In simple terms, it helps you discover which items or things tend to occur together frequently. For example, "people who buy bread also often buy milk." The name 'Apriori' refers to the fact that the algorithm makes use of prior knowledge about the dataset and also because it uses a "prior" step to find frequently occurring item sets.

**Real-World Examples:**

*   **Market Basket Analysis:** In retail, it's used to find what products are frequently purchased together. This information can be used for store layout, product recommendations, and targeted marketing.
*   **Website Navigation:** Analyzing which pages users visit together on a website can help improve navigation and user experience.
*   **Medical Diagnosis:** Finding which symptoms or diseases frequently occur together can help identify associated health problems.
*   **Bioinformatics**: Discover patterns in gene expression data, identifying genes that are often expressed together.
*   **Telecommunications:** Understanding what services subscribers are using together can help for marketing and product development.

## The Math Behind the Apriori Algorithm

The Apriori algorithm relies on some key concepts and mathematical calculations to find the frequent itemsets. Let's explore these concepts.

**Key Concepts:**

*   **Itemset:** A set of items. For example, {bread, milk} or {laptop, charger, mouse}.
*   **Support:** The frequency of an itemset in the dataset. It is calculated by the number of times the itemset appears in the dataset divided by the total number of transactions. A transaction here is one basket in the market or one instance in the dataset.
*   **Frequent Itemset:** An itemset with a support value greater than or equal to a specified minimum support threshold, which is set by the user.

**Support Calculation:**

The support of an itemset is calculated as:

$$
\text{Support}(X) = \frac{\text{Number of transactions containing } X}{\text{Total number of transactions}}
$$

Where,
* X is an item set
* Support(X) is the value of support for the given itemset.

**Example:**

Let's take an example of the following dataset showing the purchased items by customers, let's take an example for an itemset {Milk, Bread}:
1. {Milk, Bread, Eggs}
2. {Milk, Bread}
3. {Eggs, Bread, Butter}
4. {Milk, Bread, Butter}
5. {Milk, Butter}

Total number of transactions = 5
Number of transactions that contains {Milk, Bread} = 3
Support({Milk,Bread}) = 3/5 = 0.6

**Apriori Principle:**

The algorithm uses the Apriori principle, which states that "all subsets of a frequent itemset must also be frequent." This principle makes the algorithm efficient.

**Steps of the Apriori Algorithm:**

1.  **Initial Scan:** The algorithm scans the dataset and counts the frequency of each individual item.
2.  **Find Frequent 1-Itemsets:** Based on the minimum support threshold, the algorithm finds frequent itemsets of size 1.
3.  **Generate Candidate Itemsets:** It uses the frequent itemsets of size *k* to generate candidate itemsets of size *k+1*.
4.  **Prune Candidate Itemsets:** The algorithm uses the Apriori principle to prune infrequent itemsets and keeps only frequent item sets for next step.
5.  **Repeat:** Repeat steps 3 and 4 until no more frequent itemsets are found.

{% include figure popup=true image_path="/assets/images/courses/Apriori.png" caption="Apriori Algorithm" %}

## Prerequisites and Preprocessing

**Assumptions:**

*   **Transactional Data:**  The data should be in a transactional format, where each transaction is a set of items. For example, list of items bought in a single transaction.
*   **Binary Data:** In most cases, the algorithm assumes that the items are binary, which means that either the item is present in the transaction or not.

**Preprocessing:**

*   **Data Transformation:** The data needs to be converted into a list of itemsets. This involves preparing your data such that each transaction is a separate list of items.
*  **Removing duplicates:** Duplicate items within the same transaction can be removed.
*  **Handling categorical values**: If categorical values are present they can be converted into itemsets by creating new items using one hot encoding.
*   **Minimal Preprocessing:** Apart from converting your data into the correct format, the preprocessing requirements are minimal, as the algorithm directly uses the presence and absence of the items.

**Python Libraries:**

*   `mlxtend`: A popular library for association rule mining, provides an easy-to-use implementation of the Apriori algorithm. You will need to `pip install mlxtend`
*   `pandas`: For data handling and data manipulation.

## Tweakable Parameters and Hyperparameters

The Apriori algorithm has a few parameters:

*   **`min_support`:** The minimum support threshold. This parameter controls how often an itemset must occur in the dataset to be considered "frequent".
    *   **Effect:** A higher `min_support` means that fewer itemsets will be deemed frequent, leading to fewer rules being generated. A lower value will give more rules and may increase the compute time.
*   **`use_colnames`**: If set to True, then the column names will be used in the itemsets, and index will not be used. Default is False.
   * **Effect:** If set to True the column names will be included in output, and not the indexes.
*  **`max_len`**: Maximum length of the itemsets generated by the algorithm. Default value is `None`, meaning no limit.
    *  **Effect:** Higher value will create longer frequent item sets. This is used to reduce the computation time, or generate rules up to a specific length.

## Data Preprocessing

As mentioned, preprocessing requirements are minimal. Here are some key points:

*   **Data should be transactional:** Each record should be a transaction or a set of items.
*  **Data should be categorical**: The algorithm does not work directly with numerical data and any numerical data needs to be converted into a categorical one.
*  **Data preparation:** Make sure that the data is in the correct form, before training the algorithm.

**Examples:**

*   **Shopping Cart Data:** A dataset where each row is a customer transaction and each column represents an item (1 for bought, 0 for not bought).
*   **Movie Ratings:** A dataset where each row is a user and each column represents a movie the user has rated (1 for rated, 0 for not rated).
*   **Website clicks:**  A dataset with list of pages clicked by users.

## Implementation Example

Let's implement the Apriori algorithm using `mlxtend` and `pandas` with dummy data:

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```


```python
# Sample data
dataset = [['Milk', 'Bread', 'Eggs'],
           ['Milk', 'Bread'],
           ['Eggs', 'Bread', 'Butter'],
           ['Milk', 'Bread', 'Butter'],
           ['Milk', 'Butter']]
```


```python
# Convert data into correct format
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
```


```python
# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=2)
```


```python
# Print results
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
```
`Output:`
```
    Frequent Itemsets:
        support         itemsets
    0      0.8          (Bread)
    1      0.6         (Butter)
    2      0.4           (Eggs)
    3      0.8           (Milk)
    4      0.4  (Bread, Butter)
    5      0.4    (Bread, Eggs)
    6      0.6    (Bread, Milk)
    7      0.4   (Milk, Butter)
    
    Association Rules:
       antecedents consequents  antecedent support  consequent support  support  \
    0    (Butter)     (Bread)                 0.6                 0.8      0.4   
    1      (Eggs)     (Bread)                 0.4                 0.8      0.4   
    2     (Bread)      (Milk)                 0.8                 0.8      0.6   
    3      (Milk)     (Bread)                 0.8                 0.8      0.6   
    4    (Butter)      (Milk)                 0.6                 0.8      0.4   
    
       confidence      lift  representativity  leverage  conviction  \
    0    0.666667  0.833333               1.0     -0.08         0.6   
    1    1.000000  1.250000               1.0      0.08         inf   
    2    0.750000  0.937500               1.0     -0.04         0.8   
    3    0.750000  0.937500               1.0     -0.04         0.8   
    4    0.666667  0.833333               1.0     -0.08         0.6   
    
       zhangs_metric  jaccard  certainty  kulczynski  
    0      -0.333333      0.4  -0.666667    0.583333  
    1       0.333333      0.5   1.000000    0.750000  
    2      -0.250000      0.6  -0.250000    0.750000  
    3      -0.250000      0.6  -0.250000    0.750000  
    4      -0.333333      0.4  -0.666667    0.583333  
    
```

```python
# Save and load
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)

loaded_frequent_itemsets = pd.read_csv('frequent_itemsets.csv')
loaded_rules = pd.read_csv('association_rules.csv')

print("\nLoaded Frequent Itemsets:\n", loaded_frequent_itemsets)
print("\nLoaded Association Rules:\n", loaded_rules)
```
`Output:`
```
    
    Loaded Frequent Itemsets:
        support                        itemsets
    0      0.8            frozenset({'Bread'})
    1      0.6           frozenset({'Butter'})
    2      0.4             frozenset({'Eggs'})
    3      0.8             frozenset({'Milk'})
    4      0.4  frozenset({'Bread', 'Butter'})
    5      0.4    frozenset({'Bread', 'Eggs'})
    6      0.6    frozenset({'Bread', 'Milk'})
    7      0.4   frozenset({'Milk', 'Butter'})
    
    Loaded Association Rules:
                  antecedents           consequents  antecedent support  \
    0  frozenset({'Butter'})  frozenset({'Bread'})                 0.6   
    1    frozenset({'Eggs'})  frozenset({'Bread'})                 0.4   
    2   frozenset({'Bread'})   frozenset({'Milk'})                 0.8   
    3    frozenset({'Milk'})  frozenset({'Bread'})                 0.8   
    4  frozenset({'Butter'})   frozenset({'Milk'})                 0.6   
    
       consequent support  support  confidence      lift  representativity  \
    0                 0.8      0.4    0.666667  0.833333               1.0   
    1                 0.8      0.4    1.000000  1.250000               1.0   
    2                 0.8      0.6    0.750000  0.937500               1.0   
    3                 0.8      0.6    0.750000  0.937500               1.0   
    4                 0.8      0.4    0.666667  0.833333               1.0   
    
       leverage  conviction  zhangs_metric  jaccard  certainty  kulczynski  
    0     -0.08         0.6      -0.333333      0.4  -0.666667    0.583333  
    1      0.08         inf       0.333333      0.5   1.000000    0.750000  
    2     -0.04         0.8      -0.250000      0.6  -0.250000    0.750000  
    3     -0.04         0.8      -0.250000      0.6  -0.250000    0.750000  
    4     -0.08         0.6      -0.333333      0.4  -0.666667    0.583333  
    
```

**Explanation:**

*   **Frequent Itemsets:** The output of frequent itemsets shows the items and combinations of items that occur more than the specified min\_support value, along with their support.
    
*   **Association Rules:** Association rules show the items that are associated with each other using confidence, lift, leverage, and conviction:
    
    *   **Antecedents**: Items present before the item to be predicted.
        
    *   **Consequents**: Items to be predicted using the antecedents.
        
    *   **Antecedent Support**: Support of the antecedent itemset.
        
    *   **Consequent Support**: Support of the consequent itemset.
        
    *   **Support:** Support of the union of antecedent and consequent itemsets.
        
    *   **Confidence**: Confidence indicates how often the consequent appears given the antecedent is present. Calculated as Support(Antecedent Union Consequent) / Support(Antecedent)
        
    *   **Lift**: Lift is calculated as (Support(Antecedent U Consequent)) / Support(Antecedent) \* Support(Consequent). Lift value of 1 indicates no relationship between items, Lift greater than 1 indicates that they are more likely to be bought together, and lift less than 1 indicates that they are less likely to be bought together.
        
    *   **Leverage**: Leverage is calculated as Support(Antecedent U Consequent) - (Support(Antecedent) \* Support(Consequent)). Leverage shows the difference between the observed frequency of A and B appearing together and the frequency that would be expected if A and B were independent. A leverage value of zero indicates independence.
        
    *   **Conviction**: Conviction is measured as the probability that if the consequent is false the antecedent is false also, calculated as (1 - Support(Consequent))/(1-Confidence). A high conviction value means a consequent is highly dependent on the antecedent.
        
*   **Pickle:** CSV files are saved to save output, and loaded back again for future use. to\_csv is used to save data and read\_csv is used to load data from csv files.
    

Post-Processing
---------------

*   **Rule Evaluation:** Filter and sort rules using different metrics such as support, confidence, and lift to find most useful rules.
    
*   **Visualization:** Visualize the association rules using graphs.
    
*   **AB Testing:** Use AB tests to test if the association rules will improve the business.
    
*   Other statistical testing methods can also be applied if needed.
    

Hyperparameter Tuning
---------------------

Hyperparameter tuning for Apriori algorithm is usually not required as it only has a single hyperparameter, which is min\_support. However, you can perform parameter exploration by checking the model output based on different values of min\_support. The value should be chosen based on the dataset, and it needs to be fine tuned.

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample data
dataset = [['Milk', 'Bread', 'Eggs'],
           ['Milk', 'Bread'],
           ['Eggs', 'Bread', 'Butter'],
           ['Milk', 'Bread', 'Butter'],
           ['Milk', 'Butter']]

# Convert data into correct format
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Explore effect of min_support
min_supports = [0.2, 0.4, 0.6]

for min_support in min_supports:
  # Find frequent itemsets
  frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
  print(f"Frequent Itemsets with min_support = {min_support}:\n", frequent_itemsets)
  ```

  Checking Model Accuracy
-----------------------

The Apriori algorithm is not about "accuracy" in the traditional sense of prediction. It focuses on finding relationships, which are measured using the following:

*   **Support:** The measure of how often an itemset appears in a dataset.
    
*   **Confidence:** The measure of how often a rule is true. It is calculated as Support(Antecedent Union Consequent)/Support(Antecedent)
    
*   **Lift:** The ratio of the observed co-occurrence of items to the expected co-occurrence. Greater than 1 indicates that they are more likely to be bought together. It is calculated as (Support(Antecedent U Consequent)) / (Support(Antecedent) \* Support(Consequent)).
    
*   **Leverage:** It shows the difference between the observed frequency of A and B appearing together and the frequency that would be expected if A and B were independent. Calculated as Support(Antecedent U Consequent) - (Support(Antecedent) \* Support(Consequent)).
    
*   **Conviction**: Conviction is measured as the probability that if the consequent is false the antecedent is false also. It is calculated as (1 - Support(Consequent))/(1-Confidence).
    

Productionizing Steps
---------------------

*   **Local Testing:** Create a script for testing the algorithm on a small sample of your data.
    
*   **On-Prem:** Deploy the model on a server, and use containers to manage the deployment.
    
*   **Cloud:** Deploy the model on a cloud platform.
    
*   **Integration:** Integrate the output of the algorithm into the main system, and deploy changes to production.
    
*   **Real-time/Batch Processing:** Develop a data pipeline for generating association rules with fresh data on daily basis, which can be used by other services.
    

Conclusion
----------

The Apriori algorithm is a valuable tool for finding interesting patterns in transactional data. Despite its age, it is still a very useful algorithm in various industries for discovering important relationships in data. While it can be computationally expensive on very large datasets, there are more optimized algorithms that have been developed, which are more scalable and computationally efficient. However, Apriori is a foundational algorithm for association rule mining, and it is valuable to understand before moving to other advanced algorithms.

References
----------

1.  **mlxtend Documentation:** [http://rasbt.github.io/mlxtend/](https://www.google.com/url?sa=E&q=http://rasbt.github.io/mlxtend/)
    
2.  **Wikipedia:** [https://en.wikipedia.org/wiki/Apriori\_algorithm](https://www.google.com/url?sa=E&q=https://en.wikipedia.org/wiki/Apriori_algorithm)
    
3.  **Data Mining Book:** [https://www-users.cse.umn.edu/~kumar001/dmbook/index.php](https://www.google.com/url?sa=E&q=https://www-users.cse.umn.edu/~kumar001/dmbook/index.php)