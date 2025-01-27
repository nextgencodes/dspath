---
title: "Unveiling the Eclat Algorithm: Finding Hidden Patterns in Your Data"
excerpt: "Eclat Algorithm"
# permalink: /courses/association/eclat/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
  - Data Mining
tags: 
  - Machine Learning
  - Frequent Itemset
  - Association Rule Mining
---

{% include download file="eclat.ipynb" alt="download eclat code" text="Download Code" %}


## Introduction: What is the Eclat Algorithm?

Imagine you're a store owner trying to understand what products your customers often buy together. Knowing these patterns can help you strategically place items, create targeted promotions, and ultimately boost sales. This is where the Eclat algorithm comes in handy. 

Eclat, short for **Equivalence Class Transformation**, is a clever technique used in the field of data mining to find **frequent itemsets** within a dataset.  A frequent itemset is simply a collection of items that appear together in a significant number of transactions. 

Think of it like this:
*   **Transactions:** Each customer purchase is a transaction (e.g., a basket of items).
*   **Items:** The products in those baskets are the items (e.g., milk, bread, eggs).
*   **Frequent Itemset:** A combination of items often found together (e.g., milk and bread) is a frequent itemset.

**Real-World Examples:**
*   **Retail:**  Identifying which products are frequently bought together to optimize shelf placement and suggest complementary items to customers.
*   **E-commerce:**  Recommending "frequently bought together" items on online stores.
*   **Medical Diagnosis:** Finding combinations of symptoms that are often associated with a specific disease.
*   **Web Usage Analysis:** Discovering which pages on a website users frequently access together.

## The Mathematics Behind Eclat

At its heart, the Eclat algorithm uses a simple yet powerful concept: **vertical data representation** and **set intersections**.

Instead of looking at the data horizontally (transaction by transaction), Eclat flips it vertically. Let's say you have the following transaction data.

| Transaction ID | Items          |
| -------------- | -------------- |
| T1             | A, B, C        |
| T2             | A, C           |
| T3             | A, B           |
| T4             | B, D           |
| T5             | B, C           |

Eclat converts it into vertical representation which looks like this.

| Items | Transaction IDs |
| --- | --- |
| A | T1, T2, T3 |
| B | T1, T3, T4, T5 |
| C | T1, T2, T5 |
| D | T4 |

Now, to find frequent itemsets, we use the concept of support.

**Support:** The support of an itemset is the fraction of transactions that contain the itemset. For example, the itemset {A,C} appears in transactions T1 and T2. So the support of the itemset {A,C} is 2/5 = 0.4.

We define a minimum support threshold (e.g., 30%). Itemsets that have support greater than or equal to the minimum support are considered frequent.

**How Eclat uses Set Intersection**
The core operation of Eclat is based on intersecting the transaction IDs of each item.
* Initially, we identify the 1 itemset. In the above example, {A},{B},{C}, and {D} are 1-itemset. We calculate the support of the 1-itemset and filter out the infrequent ones.
* For each frequent 1-itemset, we start by intersecting the transaction IDs of item A with that of item B which will give the transaction IDs for itemset {A,B}. The count of the resulting transaction IDs gives us the support for itemset {A,B}.
* We do this for all combinations of frequent 1-itemset and generate the frequent 2-itemset.
* We iteratively repeat this process to generate the frequent 3-itemset, 4-itemset and so on.
* The process stops when no more frequent itemsets can be generated.

**Mathematical Representation:**

If we denote the set of transactions in which item *i* appears by *T(i)* and an itemset by *I = {i1, i2, ..., ik}*, then the support of I can be denoted as.

$$
   \text{support}(I) = \frac{|T(i_1) \cap T(i_2) \cap ... \cap T(i_k)|}{\text{Total number of Transactions}}
$$

Where:

*   \|...\| denotes the cardinality of the set (number of elements).
    
*      ∩    denotes the intersection of the sets.
    

Prerequisites and Preprocessing
-------------------------------

Before diving into Eclat, let's look at the necessary preparations:

**Assumptions:**

1.  **Transactional Data:** The algorithm assumes your data is in a transactional format, where each transaction is a set of items.
    
2.  **Binary Representation:** Eclat works with the presence or absence of an item in a transaction, not quantities.
    

**Testing Assumptions:**

1.  **Data Inspection:** Manually check your data to ensure it aligns with the transactional format.
    
2.  **Data Summary:** Create summaries to understand the data structure, i.e., the number of transactions, unique items and number of items per transaction.
    

**Python Libraries:**

*   `pandas`: For data manipulation and loading.
    
*   `mlxtend`: For the actual Eclat algorithm implementation.

Data Preprocessing
------------------

Unlike some algorithms, Eclat doesn't require heavy preprocessing like normalization or scaling. This is because Eclat is working with a binary representation of items present in a transaction. It is based on presence and absence, rather than numerical values.

**Minimal Preprocessing:**

1.  **Data Conversion:** You need to convert your transactional data into a format suitable for the algorithm. This usually involves creating a binary matrix where rows are transactions, columns are unique items, and values are 1 (item present) or 0 (item absent).
    
2.  **Handling Missing Data:** This is crucial. If the item is missing in the transaction, you should fill it with 0, if you are generating binary matrix in the previous step.
    
3.  **Data Cleaning:** Address inconsistencies in naming or formatting (e.g., "apple" vs "Apple").
    

**Example:**Let's say you have purchase data.
Let's say you have purchase data.

| Transaction ID | Items   |
| -------------- | -------- |
| T1            |  Bread, Milk     |
| T2            |  Bread, Diaper, Beer        |
| T3            | Milk, Diaper, Beer        |
| T4            |  Bread, Milk    |

You'd preprocess this into a binary matrix:

| Transaction ID | Beer | Bread | Diaper | Milk |
| -------------- | ---- | ----- | ------ | ---- |
| T1           |  0   |  1    |  0    | 1    |
| T2           |  1   |  1    |  1     |  0   |
| T3           |  1   |  0    |  1    | 1    |
| T4           | 0    | 1     |  0    | 1    |

## Implementation Example

Let's see how Eclat works with some dummy data. We will use the same data from previous example.

```python
import pandas as pd
from pyECLAT import ECLAT
```


```python
# Dummy transactional data
transactions = pd.DataFrame( [
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk']
])
```


```python
# Run Eclat
eclat = ECLAT(data=transactions)
#Binarizing Data Frame
eclat.df_bin
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Beer</th>
      <th>None</th>
      <th>Milk</th>
      <th>Diaper</th>
      <th>Bread</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# count items in each column
items_total = eclat.df_bin.astype(int).sum(axis=0)
items_total

# count items in each row
items_per_transaction = eclat.df_bin.astype(int).sum(axis=1)
items_per_transaction
```

```python
# Loading items per column stats to the DataFrame
df = pd.DataFrame({'items': items_total.index, 'transactions': items_total.values}) 
# cloning pandas DataFrame for visualization purpose  
df_table = df.sort_values("transactions", ascending=False)
#  Top 5 most popular products/items
df_table.head(5).style.background_gradient(cmap='Blues')
```


`Output:`

<style type="text/css">
#T_d6cd8_row0_col1, #T_d6cd8_row1_col1 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_d6cd8_row2_col1, #T_d6cd8_row3_col1 {
  background-color: #3787c0;
  color: #f1f1f1;
}
#T_d6cd8_row4_col1 {
  background-color: #f7fbff;
  color: #000000;
}
</style>
<table id="T_d6cd8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d6cd8_level0_col0" class="col_heading level0 col0" >items</th>
      <th id="T_d6cd8_level0_col1" class="col_heading level0 col1" >transactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d6cd8_level0_row0" class="row_heading level0 row0" >2</th>
      <td id="T_d6cd8_row0_col0" class="data row0 col0" >Milk</td>
      <td id="T_d6cd8_row0_col1" class="data row0 col1" >3</td>
    </tr>
    <tr>
      <th id="T_d6cd8_level0_row1" class="row_heading level0 row1" >4</th>
      <td id="T_d6cd8_row1_col0" class="data row1 col0" >Bread</td>
      <td id="T_d6cd8_row1_col1" class="data row1 col1" >3</td>
    </tr>
    <tr>
      <th id="T_d6cd8_level0_row2" class="row_heading level0 row2" >0</th>
      <td id="T_d6cd8_row2_col0" class="data row2 col0" >Beer</td>
      <td id="T_d6cd8_row2_col1" class="data row2 col1" >2</td>
    </tr>
    <tr>
      <th id="T_d6cd8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d6cd8_row3_col0" class="data row3 col0" >Diaper</td>
      <td id="T_d6cd8_row3_col1" class="data row3 col1" >2</td>
    </tr>
    <tr>
      <th id="T_d6cd8_level0_row4" class="row_heading level0 row4" >1</th>
      <td id="T_d6cd8_row4_col0" class="data row4 col0" >None</td>
      <td id="T_d6cd8_row4_col1" class="data row4 col1" >0</td>
    </tr>
  </tbody>
</table>





```python
# to have a same origin
df_table["all"] = "Tree Map" 
```


```python
# the item shoud appear at least at 5% of transactions
min_support = 5/100
# start from transactions containing at least 2 items
min_combination = 2
# up to maximum items per transaction
max_combination = max(items_per_transaction)
rule_indices, rule_supports = eclat.fit(min_support=min_support,
                                                 min_combination=min_combination,
                                                 max_combination=max_combination,
                                                 separator=' & ',
                                                 verbose=True)
```
`Output:`
```
    Combination 2 by 2
    6it [00:00, 96.98it/s]
    Combination 3 by 3
    4it [00:00, 87.88it/s]
```

```python
result = pd.DataFrame(rule_supports.items(),columns=['Item', 'Support'])
result.sort_values(by=['Support'], ascending=False)
```

`Output:`


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Beer &amp; Diaper</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Milk &amp; Bread</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beer &amp; Bread</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Beer &amp; Milk</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Milk &amp; Diaper</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Diaper &amp; Bread</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Beer &amp; Milk &amp; Diaper</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Beer &amp; Diaper &amp; Bread</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>


**Explanation:**
*   `support`: The support column shows the fraction of transactions containing the itemsets. For example, Similarly, itemsets {Bread, Milk} appears in 50% of transactions.
*   `itemsets`: The itemsets column shows all frequent combinations found.

## Post-Processing and Analysis

After running Eclat, you might want to analyze results further.

**Finding Important Rules:**
*   **Association Rule Mining:** Combine Eclat with association rule mining algorithms like Apriori to extract association rules (e.g., if a customer buys bread, they are likely to buy milk).
*   **Hypothesis Testing:** You can use hypothesis testing to check if the support of a particular item set is statistically significant. For example, if your itemset {Bread, Milk} has a higher support than usual, then you can perform hypothesis testing to show that the relationship is not due to random chance, and that they are significantly associated with each other.

## Tweaking Parameters and Hyperparameters

Eclat has a few parameters that impact its performance:

*   **`min_support`:** This is the most important parameter. It sets the minimum support threshold for itemsets to be considered frequent.
    *   *Effect:* A higher `min_support` will result in fewer frequent itemsets, potentially missing some less frequent patterns. A lower `min_support` will increase the number of frequent itemsets and may result in very large result with less useful patterns and require more computation.
    *   *Example:* If set to 0.1, only item sets appearing in at least 10% of all transactions will be considered.
*   **`min_combination`:** This parameter helps in specifying the minimum number of item in the itemset.
    *   *Effect:* This helps in controlling the length of itemset. It limits the depth of exploration and can reduce computation complexity.
    *   *Example*: If set to 2, the minimum size of the frequent itemsets will be 2 (e.g., {A,B}, {C,D}).
*   **`max_combination`:** This parameter helps in specifying the maximum number of item in the itemset.
    *   *Effect:* This helps in controlling the length of itemset. It limits the depth of exploration and can reduce computation complexity.
    *   *Example*: If set to 2, the maximum size of the frequent itemsets will be 2 (e.g., {A,B}, {C,D}).

**Hyperparameter Tuning**

Eclat doesn't have extensive hyperparameters for tuning. However, you might experiment with different values of min_support to find one which gives meaningful insights for your dataset. This can be done using cross validation technique which we are not implementing here.

## Model Evaluation

Eclat itself doesn't have a traditional "accuracy" metric since it's not a predictive model. Instead, we assess the usefulness of its results:

*   **Support:**  The support of the generated item sets itself is a measure of relevance.
*   **Association Rules Metrics:**  For association rules generated after Eclat (using Apriori or similar), metrics like confidence, lift, and leverage are used to measure rule strength.
    *   **Confidence**:  It indicates how often the rule has been found to be true.
    *   **Lift:**  It shows the association between the occurrence of the rule with the occurrence of the consequent. The lift value of 1 means that there is no association between them, and lift value more than 1 means they are positively correlated, and lift value less than 1 means that they are negatively correlated.
    *   **Leverage:**  It is the difference between the frequency that they appear together in the dataset, and what we would expect if they were statistically independent.
*   **Subject Matter Expertise:** Domain knowledge is important in evaluating whether the identified itemsets make sense and offer real value.

## Model Productionizing

Deploying Eclat models in production involves a few key steps:

1.  **Data Pipeline:** Create a robust data pipeline for processing incoming transaction data.
2.  **Periodic Updates:** Regularly retrain the model (or re-run Eclat) on new data to capture evolving trends.
3.  **API Integration:** Expose the model predictions or association rules as an API for other systems to use.
4.  **Cloud Deployment:** Use cloud services like AWS, Google Cloud, or Azure to deploy and scale your model.
5.  **On-Premise Deployment:** Use docker container or similar method to deploy the model within your own infrastructure if cloud is not suitable.

Here's a simple example of creating an API endpoint for predicting frequent itemsets. We are using a flask library to create a dummy API here. The actual model can be deployed in a similar way.

```python
from flask import Flask, request, jsonify
import pandas as pd
from pyECLAT import ECLAT
from mlxtend.preprocessing import TransactionEncoder
import json

app = Flask(__name__)

# Dummy data and model (replace with actual data and model loading)
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk']
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
#frequent_itemsets = ECLAT(df, min_combination=0.5, use_colnames=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_data()
        print(json.loads(data))
        data = pd.DataFrame(json.loads(data)['data'])
        # Run Eclat
        eclat = ECLAT(data=data)
        # the item shoud appear at least at 5% of transactions
        min_support =  0.5 # Default min_support is 0.5
        # start from transactions containing at least 2 items
        min_combination = 2
        # up to maximum items per transaction
        max_combination = 10
        rule_indices, rule_supports = eclat.fit(min_support=min_support,
                                                         min_combination=min_combination,
                                                         max_combination=max_combination,
                                                         separator=' & ',
                                                         verbose=True)
        print(rule_supports)
        result = pd.DataFrame(rule_supports.items(),columns=['Item', 'Support'])
        result.sort_values(by=['Support'], ascending=False)
        itemsets_dict = result.to_dict(orient='records')
        return jsonify({'result':itemsets_dict})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)
```
To test the API, save this code as `app.py` and run `python app.py`, then use a curl command in terminal as following to get the API output.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data":[
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk']
]}' http://0.0.0.0:5000/predict
```

## Conclusion

The Eclat algorithm is a simple and powerful tool for uncovering hidden patterns in transactional data. It is widely used in various industries for making data-driven decisions and is still used today. While there may be optimized and newer techniques, Eclat remains a core algorithm in the field. Some optimized and new algorithms are FP-Growth, and Apriori algorithms. The choice of algorithms depends on the data structure, size, and computational resources available.

## References

1.  Zaki, Mohammed J., et al. "An efficient algorithm for mining association rules." *Proceedings of the 7th international conference on information and knowledge management*. 1998.
2.  Borgelt, Christian. "Efficient implementations of the eclat algorithm." *Proceedings of the ICDM workshop on frequent itemset mining implementations*. 2005.
3. mlxtend library documentation: http://rasbt.github.io/mlxtend/


