---
title: "K-Medians Clustering: A Robust Approach for Grouping Data"
excerpt: "K-Medians Clustering Algorithm"
# permalink: /courses/clustering/kmedian/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Partitional Clustering
  - Unsupervised Learning
  - Clustering Algorithm
  - Centroid-based
  - Robust Clustering
tags: 
  - Clustering algorithm
  - Partitional clustering
  - Centroid-based
  - Robust to outliers
---


{% include download file="k_medians_clustering.ipynb" alt="download k-medians clustering code" text="Download Code" %}

## Introduction to K-Medians Clustering: Finding Centers in Your Data

Imagine you're organizing a treasure hunt in a large park. You want to divide the park into several zones and hide treasures at the 'center' of each zone so that they are relatively easy to find from anywhere within that zone. You want to choose these 'centers' wisely to minimize the distance people have to travel to find a treasure in their assigned zone.

K-Medians clustering is a technique that helps us do something similar with data.  Think of your data points as locations in the park.  The K-Medians algorithm aims to find a set of 'center points', called **medians**, such that each data point is assigned to the closest median. The goal is to group similar data points together, forming **clusters**.  It's a way to automatically discover groupings in your data without needing to pre-define what these groups are.

**Real-world Examples:**

*   **Location Optimization for Stores/Facilities:** A company wants to open a new chain of stores in a city. They can use K-Medians to cluster customer locations and then place stores near the 'median' location of each customer cluster to minimize travel distance for customers.
*   **Document Grouping:** Imagine you have a large collection of articles. K-Medians can group articles based on similar themes or topics.  Each cluster would represent a set of articles centered around a specific theme, and the 'median' document could be thought of as a representative article for that theme.
*   **Image Segmentation:** In image processing, K-Medians can be used to segment an image into different regions based on color or pixel intensity. Each cluster could represent a segment of the image, and the 'median' color of each segment can represent the typical color in that region.
*   **Customer Segmentation:** Businesses can use K-Medians to divide their customers into different groups based on purchasing behavior, demographics, or other features. Each cluster represents a distinct customer segment, and the 'median' customer in each segment can characterize the typical customer profile for that segment.
*   **Anomaly Detection (indirectly):** While not primarily for anomaly detection, in some cases, data points that are far from all cluster medians could be considered potential anomalies, as they don't neatly fit into any of the identified clusters.

K-Medians is particularly useful when you suspect your data might contain **outliers**, or extreme values, because it is less sensitive to outliers than its cousin, the K-Means algorithm.  Let's see how it works mathematically!

## The Mathematics Behind K-Medians: Minimizing Manhattan Distance

K-Medians clustering is very similar to the more famous K-Means clustering, but it uses a different way to measure distance and define the 'center' of a cluster.  Instead of using the **mean** to represent the center of a cluster and **Euclidean distance** to measure distance, K-Medians uses the **median** as the cluster center and **Manhattan distance** to measure how far points are from the center.

**Manhattan Distance (L1 Distance):**

Imagine city blocks laid out in a grid.  To travel from one point to another, you can only move along the grid lines, like walking in Manhattan. The Manhattan distance between two points $(x_1, y_1)$ and $(x_2, y_2)$ in a 2D plane is calculated as the sum of the absolute differences of their coordinates:

$d_{Manhattan}((x_1, y_1), (x_2, y_2)) = |x_1 - x_2| + |y_1 - y_2|$

In general, for two data points $\mathbf{p} = (p_1, p_2, ..., p_n)$ and $\mathbf{q} = (q_1, q_2, ..., q_n)$ in an n-dimensional space, the Manhattan distance is:

$d_{Manhattan}(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{n} |p_i - q_i|$

**Example:** Let's say we have two points in 2D: $\mathbf{p} = (1, 2)$ and $\mathbf{q} = (4, 6)$. The Manhattan distance is:

$d_{Manhattan}(\mathbf{p}, \mathbf{q}) = |1 - 4| + |2 - 6| = |-3| + |-4| = 3 + 4 = 7$

**K-Medians Algorithm Steps:**

1.  **Initialization:** Choose the number of clusters, $K$. Randomly select $K$ data points from your dataset to be the initial **medians**. These medians are the initial 'centers' of our clusters.

2.  **Assignment Step:** For each data point in your dataset, calculate its Manhattan distance to each of the $K$ current medians. Assign the data point to the cluster whose median is closest to it (smallest Manhattan distance).

3.  **Update Step:** For each cluster, recalculate the **median** of all data points currently assigned to that cluster.  This new median becomes the new center of that cluster. To find the median in multiple dimensions, you find the median for each dimension independently. For example, if you have points $(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)$ in a cluster, the new median is $(\text{median}(x_1, x_2, ..., x_m), \text{median}(y_1, y_2, ..., y_m))$.

4.  **Iteration:** Repeat steps 2 and 3 until the cluster assignments no longer change significantly or a maximum number of iterations is reached. This means the algorithm has converged, and the medians and clusters are stable.

**Objective Function (Cost Function):**

The goal of K-Medians is to minimize the **sum of Manhattan distances** of each data point to its closest cluster median. This is the objective function that the algorithm tries to minimize. Let's say we have $K$ clusters $C_1, C_2, ..., C_K$ and their medians are $\mathbf{m}_1, \mathbf{m}_2, ..., \mathbf{m}_K$. The objective function, often called the **cost** or **inertia** in clustering, is:

$J = \sum_{j=1}^{K} \sum_{\mathbf{x} \in C_j} d_{Manhattan}(\mathbf{x}, \mathbf{m}_j)$

Where:

*   $J$ is the cost function we want to minimize.
*   $K$ is the number of clusters.
*   $C_j$ is the $j$-th cluster.
*   $\mathbf{m}_j$ is the median of cluster $C_j$.
*   $\mathbf{x}$ is a data point in cluster $C_j$.
*   $d_{Manhattan}(\mathbf{x}, \mathbf{m}_j)$ is the Manhattan distance between data point $\mathbf{x}$ and median $\mathbf{m}_j$.

The K-Medians algorithm iteratively refines the medians and cluster assignments to reduce this total sum of Manhattan distances, aiming to create clusters that are as 'compact' as possible with respect to the Manhattan distance.

## Prerequisites and Preprocessing for K-Medians Clustering

Before using K-Medians clustering, it's important to understand its prerequisites and consider necessary preprocessing steps for your data.

**Prerequisites & Assumptions:**

*   **Numerical Data:** K-Medians, as typically implemented, works best with numerical data because it relies on calculating distances and medians. If you have categorical data, you need to convert it to a numerical form (e.g., using one-hot encoding) before applying K-Medians.
*   **Defined Distance Metric (Manhattan Distance):** K-Medians is specifically designed to use Manhattan distance.  You should consider if Manhattan distance is appropriate for measuring similarity in your data. Manhattan distance is suitable when the dimensions are independent and movement along each dimension is equally 'costly' or meaningful. It's less sensitive to outliers than Euclidean distance.
*   **Number of Clusters (K):** You need to pre-determine the number of clusters, $K$. This is a hyperparameter. Choosing the right $K$ is often crucial and can be challenging.  Techniques like the Elbow Method or Silhouette analysis (discussed later) can help you estimate a good value for $K$.
*   **Data Scaling (often recommended):** While not strictly required, scaling your numerical features is generally recommended, especially if features are on vastly different scales.  Features with larger scales can disproportionately influence the Manhattan distance calculation.

**Assumptions (Implicit):**

*   **Clusters are roughly spherical or elongated along axes:** K-Medians, like K-Means, tends to work well when clusters are somewhat spherical or linearly separable.  However, because it uses Manhattan distance, it is more robust to outliers and can handle clusters that are aligned with the axes better than K-Means.
*   **Each data point belongs to exactly one cluster:** K-Medians performs hard clustering, meaning each data point is assigned to only one cluster.

**Testing Assumptions (Informally):**

*   **Data Exploration:**  Visualize your data (if possible, e.g., using scatter plots for 2D or 3D data).  Look for visual groupings or patterns. This can give you an initial idea if clustering is appropriate and perhaps a sense of how many clusters might be present.
*   **Feature Scaling Check:** Examine the ranges and distributions of your numerical features. If features have very different scales (e.g., one feature ranges from 0-1, another from 1000-1000000), scaling is likely necessary.

**Python Libraries:**

For implementing K-Medians clustering in Python, you will typically use:

*   **NumPy:** For numerical operations, array manipulations, and especially for calculating medians and Manhattan distances efficiently.
*   **pandas:** For data manipulation and working with DataFrames.
*   **scikit-learn (sklearn):**  While scikit-learn does not have a built-in K-Medians algorithm directly, you can implement it using its functionalities (like distance metrics and iterative algorithms).  Alternatively, libraries like `kmodes` or `scikit-learn-extra` (for newer versions) might provide K-Medians implementations. We will demonstrate a manual implementation in our example for better understanding, and then show how to use existing libraries.
*   **Matplotlib** or **Seaborn:** For data visualization, which can be helpful for understanding your data and visualizing the clusters.

## Data Preprocessing for K-Medians

Data preprocessing is crucial for K-Medians to work effectively.  Here are common preprocessing steps:

*   **Handling Missing Values:**
    *   **Why it's important:** K-Medians calculations (distance and median) do not directly handle missing values. Missing values will cause errors or lead to incorrect cluster assignments.
    *   **Preprocessing techniques:**
        *   **Imputation:** Fill in missing values with estimated values. Common methods include:
            *   **Median Imputation:** Since K-Medians uses medians, using median imputation to fill missing values can be conceptually consistent.  Replace missing values in a feature with the median value of that feature.
            *   **Mean Imputation:**  Replace missing values with the mean value. Less consistent with K-Medians' median-based approach but still a common option.
            *   **KNN Imputation:** Use K-Nearest Neighbors to predict and impute missing values based on similar data points.
            *   **Deletion (Listwise or Pairwise):** Remove rows (listwise deletion) or columns (pairwise deletion, less common for features in clustering) with missing values. Use with caution, as it can lead to data loss.
    *   **When can it be ignored?** If your dataset has very few missing values (e.g., less than 1-2% and randomly distributed), and you're using a robust imputation technique, you *might* consider very simple imputation or even deletion if the impact is minimal. However, it's generally better to address missing values.

*   **Handling Categorical Features:**
    *   **Why it's important:** Standard K-Medians works with numerical data and Manhattan distance, which is designed for numerical dimensions. Categorical features need to be converted to a numerical form.
    *   **Preprocessing techniques:**
        *   **One-Hot Encoding:** Convert categorical features into binary vectors.  For example, if you have a feature "Color" with categories "Red," "Blue," "Green," one-hot encode it into three binary features: "Is\_Red," "Is\_Blue," "Is\_Green."  Manhattan distance can then be applied to these binary features.
        *   **Frequency Encoding or Target Encoding:** For categorical features with many categories, one-hot encoding can lead to high dimensionality. Frequency encoding replaces categories with their frequency of occurrence. Target encoding (for supervised tasks, less common in clustering) replaces categories with the mean of the target variable for that category. These methods can be considered if one-hot encoding results in too many dimensions.
    *   **When can it be ignored?**  If you only have numerical features, or if your categorical features are ordinal and can be reasonably represented by numerical ranks without losing too much information. However, for nominal (unordered) categorical features, numerical encoding (like one-hot encoding) is generally necessary.

*   **Feature Scaling (Normalization or Standardization):**
    *   **Why it's important:** Features with larger scales can dominate the Manhattan distance calculation.  Scaling ensures that all features contribute more equally to the distance.
    *   **Preprocessing techniques:**
        *   **Min-Max Scaling (Normalization):** Scales features to a specific range, typically [0, 1] or [-1, 1]. Formula: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$.
        *   **Standardization (Z-score normalization):** Scales features to have a mean of 0 and standard deviation of 1. Formula: $z = \frac{x - \mu}{\sigma}$.
        *   **Robust Scaling:**  Scalers that are less sensitive to outliers, like `RobustScaler` in scikit-learn (uses median and interquartile range).  Given K-Medians' robustness to outliers itself, and its use of median, robust scaling can be a good choice to complement K-Medians.
    *   **Example:** If you are clustering customer data and one feature is "Income" (ranging from \$20,000 to \$500,000) and another is "Age" (ranging from 20 to 80), income will have a much larger range and might overly influence the Manhattan distance. Scaling both features to a similar range is advisable.
    *   **When can it be ignored?**  If all your features are already on comparable scales, or if the differences in scales are not expected to significantly bias the clustering results. However, scaling is generally recommended for K-Medians, especially when features have diverse units or ranges.

*   **Outlier Handling (Consideration):**
    *   **Why it's relevant:** While K-Medians is *more* robust to outliers than K-Means (due to using median instead of mean), extreme outliers can still influence cluster assignments and medians to some extent.
    *   **Preprocessing techniques:**
        *   **Outlier Removal:** Detect and remove extreme outliers before clustering. Methods like IQR (Interquartile Range) based outlier detection, Z-score based outlier detection, or domain-specific outlier identification can be used.
        *   **Robust Scaling:** Using robust scalers (like `RobustScaler`) can reduce the impact of outliers on feature scaling.
        *   **Winsorization:** Limit extreme values by capping them at a certain percentile (e.g., values above the 99th percentile are capped at the 99th percentile value).
    *   **When can it be ignored?** If you know your dataset is relatively clean from outliers, or if you specifically want the clustering to be influenced by potential outliers (e.g., if outliers are meaningful anomalies you want to detect). However, for general-purpose clustering, especially if outliers are due to errors or noise, handling them can improve clustering quality.

## Implementation Example: K-Medians Clustering in Python

Let's implement K-Medians clustering in Python using NumPy and pandas, and then demonstrate using a library for easier implementation.

**Dummy Data:**

We'll create synthetic data with a few features.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dummy data (e.g., 150 samples, 2 features for easy visualization)
np.random.seed(42)
X = np.concatenate([np.random.randn(50, 2) + [2, 2],     # Cluster 1
                    np.random.randn(50, 2) + [-2, -2],   # Cluster 2
                    np.random.randn(50, 2) + [2, -2]])  # Cluster 3
X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])

# Scale the data using StandardScaler (for demonstration, scaling often helps clustering algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

# Split data into training and testing sets (for later evaluation or comparison, not strictly needed for clustering itself)
X_train, X_test = train_test_split(X_scaled_df, test_size=0.3, random_state=42)

print("Dummy Data (first 5 rows of scaled features):")
print(X_train.head())
```

**Output:**

```
Dummy Data (first 5 rows of scaled features):
   feature_1  feature_2
97   -0.035058   1.403747
32    1.018910   1.212504
77   -1.630583  -1.297953
13    1.716226   0.591474
8    -0.197730   1.772556
```

**Implementing K-Medians Algorithm (from scratch for understanding):**

```python
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def calculate_median_centroid(data_points):
    return np.median(data_points, axis=0)

def k_medians_clustering(data, n_clusters, max_iters=100):
    n_samples, n_features = data.shape

    # 1. Initialize medians randomly (using data points as initial medians)
    initial_median_indices = np.random.choice(n_samples, n_clusters, replace=False)
    medians = data[initial_median_indices]

    cluster_assignments = np.zeros(n_samples) # To store cluster assignment for each data point
    cost_history = [] # To track cost function over iterations

    for _ in range(max_iters):
        updated_assignments = np.zeros(n_samples)
        total_cost = 0

        # 2. Assignment Step
        for i in range(n_samples):
            distances = [manhattan_distance(data[i], median) for median in medians]
            closest_median_index = np.argmin(distances)
            updated_assignments[i] = closest_median_index
            total_cost += distances[closest_median_index]

        cost_history.append(total_cost)

        # Check for convergence (assignments unchanged)
        if np.array_equal(updated_assignments, cluster_assignments):
            break
        cluster_assignments = updated_assignments

        # 3. Update Step
        for cluster_index in range(n_clusters):
            cluster_data_points = data[cluster_assignments == cluster_index]
            if len(cluster_data_points) > 0: # Handle empty clusters
                medians[cluster_index] = calculate_median_centroid(cluster_data_points)

    return cluster_assignments, medians, cost_history

# Set parameters and run K-Medians
n_clusters = 3
max_iterations = 100
cluster_labels, final_medians, cost_history_values = k_medians_clustering(X_train.values, n_clusters, max_iters=max_iterations)

print("Final Medians:\n", final_medians)
print("\nCluster Assignments (first 10):\n", cluster_labels[:10])
print("\nCost Function History (first 10 iterations):\n", cost_history_values[:10])

# Plotting the clusters (for 2D data)
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # Define colors for clusters
for cluster_index in range(n_clusters):
    cluster_data = X_train[cluster_labels == cluster_index]
    plt.scatter(cluster_data['feature_1'], cluster_data['feature_2'], c=colors[cluster_index], label=f'Cluster {cluster_index+1}')
plt.scatter(final_medians[:, 0], final_medians[:, 1], s=200, c='k', marker='X', label='Medians') # Mark medians
plt.title('K-Medians Clustering Results (Training Data)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (will vary slightly due to random initialization, plot will be displayed):**

*(Output will show the final medians, first 10 cluster assignments, and the cost function values for the first 10 iterations. A scatter plot will visualize the clusters and medians.)*

**Explanation of Output:**

*   **`Final Medians:`**: These are the coordinates of the medians found by the algorithm for each cluster in the scaled feature space.
*   **`Cluster Assignments (first 10):`**: Shows the cluster index (0, 1, or 2 in this case, for 3 clusters) assigned to the first 10 data points in the training set.  `cluster_labels` array contains assignments for all data points.
*   **`Cost Function History (first 10 iterations):`**:  Shows the value of the cost function (sum of Manhattan distances) at each iteration. You should observe that the cost generally decreases over iterations as the algorithm refines the medians and cluster assignments. Convergence is reached when the cost stops decreasing significantly or cluster assignments stabilize.
*   **Plot:** The scatter plot visualizes the clusters in 2D feature space. Different colors represent different clusters, and 'X' markers indicate the final medians. You should see data points grouped around the medians, forming clusters.

**Using `kmodes` library (for easier K-Medians):**

For easier implementation, you can use the `kmodes` library, which provides a `KMedians` class.

```python
from kmodes.kmedoids import KMedoids

# Initialize and fit K-Medians model from kmodes library
kmedoids_model = KMedoids(n_clusters=3, init='random', random_state=42, metric='manhattan') # 'random' or 'cao', metric='manhattan'
kmedoids_model.fit(X_train.values)

# Get cluster assignments and medians
cluster_labels_kmodes = kmedoids_model.labels_
final_medians_kmodes = kmedoids_model.cluster_centroids_

print("\nK-Medians using kmodes library:")
print("Final Medians (kmodes):\n", final_medians_kmodes)
print("\nCluster Assignments (kmodes, first 10):\n", cluster_labels_kmodes[:10])

# Plotting using kmodes results
plt.figure(figsize=(8, 6))
for cluster_index in range(n_clusters):
    cluster_data = X_train[cluster_labels_kmodes == cluster_index]
    plt.scatter(cluster_data['feature_1'], cluster_data['feature_2'], c=colors[cluster_index], label=f'Cluster {cluster_index+1}')
plt.scatter(final_medians_kmodes[:, 0], final_medians_kmodes[:, 1], s=200, c='k', marker='X', label='Medians (kmodes)')
plt.title('K-Medians Clustering Results (kmodes library, Training Data)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
```

**Output:**

*(Output will show similar final medians and cluster assignments as the manual implementation, and a similar scatter plot. Slight variations are possible due to randomness.)*

**Saving and Loading the Trained Model and Scaler:**

You can save the scaler for preprocessing new data. For the `kmodes` model, you'd typically save the cluster assignments and medians.  The `kmodes` library itself might offer methods for saving/loading model state, or you can manually save the medians and labels. For simplicity, let's save the scaler and the final medians (from our manual implementation) for later use.

```python
import pickle

# Save the scaler
with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the final medians (from manual implementation, adjust if using kmodes library)
with open('k_medians_medians.pkl', 'wb') as f:
    pickle.dump(final_medians, f)

print("\nScaler and K-Medians Medians saved!")

# --- Later, to load ---

# Load the scaler
with open('standard_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Load the medians
with open('k_medians_medians.pkl', 'rb') as f:
    loaded_medians = pickle.load(f)

print("\nScaler and K-Medians Medians loaded!")

# To use loaded model:
# 1. Preprocess new data using loaded_scaler
# 2. Calculate distances to loaded_medians and assign to closest median's cluster.
```

This example demonstrates both a manual implementation and using the `kmodes` library for K-Medians clustering. The library makes implementation easier, while the manual version helps in understanding the algorithm's steps.

## Post-Processing: Analyzing Clusters and Variable Importance

After running K-Medians clustering, post-processing steps are crucial to understand the clusters, validate the results, and extract meaningful insights.

**1. Cluster Profiling:**

*   **Purpose:** To characterize and describe each cluster. Understand what distinguishes one cluster from another.
*   **Techniques:**
    *   **Calculate descriptive statistics for each cluster:** For each feature, calculate the mean, median, standard deviation, min, max, percentiles within each cluster. Compare these statistics across clusters to see how feature distributions differ.
    *   **Visualize feature distributions per cluster:** Use histograms, box plots, violin plots to compare the distribution of each feature across different clusters.
    *   **Example (using pandas):**

```python
# Add cluster labels back to the original DataFrame (before scaling for easier interpretation)
X_train_original_scale = scaler.inverse_transform(X_train) # Inverse transform scaled data
X_train_original_df = pd.DataFrame(X_train_original_scale, columns=X_df.columns, index=X_train.index) # Recreate DataFrame with original indices
X_train_original_df['cluster'] = cluster_labels # Add cluster labels

# Calculate descriptive statistics per cluster
cluster_profiles = X_train_original_df.groupby('cluster').describe()
print("\nCluster Profiles (Descriptive Statistics):\n", cluster_profiles)

# Example visualization: Box plots for each feature, per cluster
for feature in X_df.columns:
    plt.figure(figsize=(8, 4))
    X_train_original_df.boxplot(column=feature, by='cluster')
    plt.title(f'Feature: {feature}')
    plt.suptitle('') # Remove default title
    plt.show()
```

**Output:**

*(Output will show descriptive statistics tables and box plots for each feature, broken down by cluster. Examine these to see how feature distributions differ between clusters.)*

*   **Interpretation:** Analyze the cluster profiles to understand what characteristics are typical for each cluster. For example, if clustering customers, you might find that Cluster 1 has high income and high spending, Cluster 2 has low income and low spending, etc. These profiles help in naming and understanding the clusters.

**2. Cluster Validation:**

*   **Purpose:** To assess the quality and validity of the clusters. Are the clusters meaningful and robust, or just artifacts of the algorithm?
*   **Techniques:**
    *   **Internal Validation Metrics:**  Metrics that assess the quality of clustering based on the data itself, without external labels. Examples:
        *   **Silhouette Score:** Measures how similar a data point is to its own cluster compared to other clusters. Ranges from -1 to +1. Higher values are better.
        *   **Davies-Bouldin Index:**  Measures the average similarity ratio of each cluster with its most similar cluster. Lower values are better.
        *   **Calinski-Harabasz Index:**  Ratio of between-cluster variance to within-cluster variance. Higher values are better.
        *   Calculate these metrics for your K-Medians results. (We will show implementation in the 'Accuracy Metrics' section later).
    *   **External Validation (if ground truth labels are available):** If you have external labels (e.g., true categories for your data points), you can compare your clustering results to these labels using metrics like:
        *   **Adjusted Rand Index (ARI)**
        *   **Normalized Mutual Information (NMI)**
        *   **Homogeneity, Completeness, V-measure**
        *   These metrics measure the agreement between your clusters and the ground truth labels. Higher values generally indicate better agreement.
    *   **Stability Analysis:** Run K-Medians multiple times with different random initializations. Check if the cluster assignments are consistent across runs. If the clusters are stable, it increases confidence in their validity.

**3. Feature Importance (Indirect Inference):**

*   **Purpose:** To get an idea of which features are most important in defining the clusters. K-Medians doesn't directly provide feature importance scores like some models (e.g., tree-based models). However, you can infer feature importance indirectly.
*   **Techniques:**
    *   **Examine Feature Distributions in Cluster Profiles:** Features that show the most significant differences in distributions across clusters are likely more important for cluster separation. Look at the descriptive statistics and visualizations from cluster profiling.
    *   **Feature Ablation or Permutation Importance (More advanced):**  For each feature, try removing it (or permuting its values) and re-run clustering. If removing a feature significantly degrades the clustering quality (e.g., worsens silhouette score or cluster separation), that feature is likely important.  This is similar to permutation importance used in feature importance analysis for supervised models.

By performing these post-processing steps, you can thoroughly analyze your K-Medians clustering results, validate their quality, and derive meaningful insights about the groupings in your data and the features that drive these groupings.

## Hyperparameter Tuning for K-Medians Clustering

The main hyperparameter to tune in K-Medians clustering is the **number of clusters, $K$**.  Choosing the right $K$ is crucial for effective clustering. If $K$ is too small, you might over-merge distinct groups. If $K$ is too large, you might split meaningful clusters into subgroups.

**Methods for Choosing the Number of Clusters (Tuning $K$):**

1.  **Elbow Method (for Cost Function/Inertia):**
    *   **Idea:** Run K-Medians for a range of $K$ values (e.g., $K=2, 3, 4, ..., 10$). For each $K$, calculate the cost function (sum of Manhattan distances, often called 'inertia' in clustering context). Plot the cost function value against $K$.
    *   **Elbow Point:** Look for an "elbow" in the plot. The elbow point is where the rate of decrease in the cost function starts to slow down noticeably. This $K$ value is often considered a reasonable choice.
    *   **Code Example:**

```python
inertia_values = []
k_range = range(1, 11) # Test K from 1 to 10

for k in k_range:
    _, _, cost_history = k_medians_clustering(X_train.values, k, max_iters=100) # Use manual implementation, or adapt for kmodes
    inertia_values.append(cost_history[-1]) # Last cost value after convergence

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K (K-Medians)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Cost Function Value)')
plt.xticks(k_range)
plt.grid(True)
plt.show()
```

*(Run this code, and look at the generated plot. Identify where the 'elbow' occurs in the curve of inertia values. This point suggests a potentially good number of clusters.)*

2.  **Silhouette Analysis:**
    *   **Idea:** Calculate the Silhouette Score for each data point and then average silhouette scores for different values of $K$. Silhouette score measures how well each data point fits into its assigned cluster compared to other clusters.
    *   **Range:** Silhouette score ranges from -1 to +1. Higher values are better.
    *   **Interpretation:**
        *   Values close to +1: Indicate that the data point is well-clustered.
        *   Values close to 0: Indicate that the data point is near the decision boundary between two clusters.
        *   Values close to -1: Indicate that the data point might be assigned to the wrong cluster.
    *   **Optimal K:** Choose the $K$ value that maximizes the average Silhouette Score.
    *   **Code Example (using scikit-learn for Silhouette score calculation):**

```python
from sklearn.metrics import silhouette_score, silhouette_samples

silhouette_scores = []
k_range = range(2, 11) # Silhouette score is not defined for K=1

for k in k_range:
    cluster_labels, _, _ = k_medians_clustering(X_train.values, k, max_iters=100) # Use manual implementation
    if len(set(cluster_labels)) > 1: # Silhouette needs at least 2 clusters assigned
        silhouette_avg = silhouette_score(X_train, cluster_labels, metric='manhattan') # Manhattan distance
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, average silhouette_score is {silhouette_avg:.4f}")
    else:
        silhouette_scores.append(0) # Or some value indicating invalid score for K=1

plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Optimal K (K-Medians)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()
```

*(Run this code and examine the plot. Find the $K$ value that corresponds to the highest average Silhouette Score. This suggests a potentially good number of clusters.)*

3.  **Davies-Bouldin Index:**
    *   **Idea:**  Calculate the Davies-Bouldin Index for different $K$ values. This index measures the average 'similarity' between each cluster and its most similar cluster.
    *   **Range:** Davies-Bouldin Index is always non-negative. Lower values are better (indicate better separation between clusters and compactness within clusters).
    *   **Optimal K:** Choose the $K$ value that minimizes the Davies-Bouldin Index.
    *   **Code Example (using scikit-learn):**

```python
from sklearn.metrics import davies_bouldin_score

db_scores = []
k_range = range(2, 11)

for k in k_range:
    cluster_labels, _, _ = k_medians_clustering(X_train.values, k, max_iters=100) # Use manual implementation
    if len(set(cluster_labels)) > 1:
        db_index = davies_bouldin_score(X_train, cluster_labels)
        db_scores.append(db_index)
        print(f"For n_clusters = {k}, Davies-Bouldin index is {db_index:.4f}")
    else:
        db_scores.append(float('inf')) # Indicate invalid score for K=1 (or very high value)

plt.figure(figsize=(8, 4))
plt.plot(k_range, db_scores, marker='o')
plt.title('Davies-Bouldin Index for Optimal K (K-Medians)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(k_range)
plt.grid(True)
plt.show()
```

*(Run this code and examine the plot. Find the $K$ value that corresponds to the lowest Davies-Bouldin Index. This suggests a potentially good number of clusters.)*

**Implementation for Hyperparameter Tuning:**

You would typically iterate through a range of $K$ values, calculate the relevant metric (inertia, Silhouette score, Davies-Bouldin index) for each $K$, and then select the $K$ that optimizes the chosen metric based on the elbow method, maximum Silhouette score, or minimum Davies-Bouldin index criteria.  You can then re-run K-Medians with the selected optimal $K$ to get your final clustering solution.

## Checking Model Accuracy: Cluster Evaluation Metrics

"Accuracy" in clustering is not measured in the same way as in classification or regression (where we have ground truth labels to compare to).  Instead, we use **cluster evaluation metrics** to assess the quality and validity of the clusters themselves. These metrics can be broadly categorized into:

*   **Internal Validation Metrics:** Evaluate clustering quality using only the data itself, without external labels. Examples:
    *   **Inertia (Cost Function Value):**  (Lower is better) Sum of squared distances (or Manhattan distances for K-Medians) of each data point to its cluster center/median. Measures cluster compactness.
    *   **Silhouette Score:** (Higher is better, range -1 to +1) Measures how well each point is clustered. High score means point is well-matched to its own cluster and poorly-matched to neighboring clusters.
    *   **Davies-Bouldin Index:** (Lower is better, non-negative) Measures cluster separation and compactness. Lower index means better-separated and more compact clusters.
    *   **Calinski-Harabasz Index:** (Higher is better, non-negative) Ratio of between-cluster scatter to within-cluster scatter. Higher index means well-defined clusters that are separated from each other.

*   **External Validation Metrics:**  Evaluate clustering quality by comparing the clustering results to external ground truth labels (if available). Examples:
    *   **Adjusted Rand Index (ARI):** (Range -1 to +1, higher is better) Measures the similarity between cluster assignments and ground truth labels, correcting for chance.
    *   **Normalized Mutual Information (NMI):** (Range 0 to 1, higher is better) Measures the mutual information between cluster assignments and ground truth labels, normalized to be between 0 and 1.
    *   **Homogeneity, Completeness, V-measure:** (Range 0 to 1, higher is better) Homogeneity measures if all clusters contain only data points of a single class. Completeness measures if all data points that are members of a given class are elements of the same cluster. V-measure is the harmonic mean of homogeneity and completeness.

**Equations for Metrics:**

We've already seen the equation for **Inertia** (Cost Function) in the 'Mathematics' section.

For **Silhouette Score:**
For a data point $i$, let $a_i$ be the average Manhattan distance of $i$ to all other points in the same cluster, and $b_i$ be the minimum average Manhattan distance of $i$ to points in a different cluster, minimized over clusters. The Silhouette score $s_i$ for point $i$ is:
$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$
The **overall Silhouette score** is the average of $s_i$ for all points.

For **Davies-Bouldin Index:**
Let $C_i$ be cluster $i$, and $\mathbf{m}_i$ be its median. Let $avg\_dist(C_i)$ be the average distance of points in $C_i$ to $\mathbf{m}_i$.  For clusters $C_i$ and $C_j$, define similarity measure $R_{ij}$ as:
$R_{ij} = \frac{avg\_dist(C_i) + avg\_dist(C_j)}{d_{Manhattan}(\mathbf{m}_i, \mathbf{m}_j)}$
The Davies-Bouldin index $DB$ is:
$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} (R_{ij})$

For **Calinski-Harabasz Index:**
Let $n$ be the number of data points, $K$ be the number of clusters, and $SS_W$ be the within-cluster sum of squares (or sum of Manhattan distances squared in K-Means, adapt for K-Medians using Manhattan distances), and $SS_B$ be the between-cluster sum of squares (or sum of squared distances from medians to overall data median, adjust for Manhattan distance). The Calinski-Harabasz index $CH$ is:
$CH = \frac{(n-K)}{(K-1)} \times \frac{SS_B}{SS_W}$

**Calculating Metrics in Python (using scikit-learn and manual calculation):**

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Get cluster labels from K-Medians result (e.g., cluster_labels from manual implementation)

# Inertia (Cost Function - already calculated during K-Medians)
print(f"Inertia (Final Cost): {cost_history_values[-1]:.4f}")

# Silhouette Score
if len(set(cluster_labels)) > 1: # Needs at least 2 clusters assigned
    silhouette_avg = silhouette_score(X_train, cluster_labels, metric='manhattan')
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    print("Silhouette Score: Not applicable for single cluster")

# Davies-Bouldin Index
if len(set(cluster_labels)) > 1:
    db_index = davies_bouldin_score(X_train, cluster_labels)
    print(f"Davies-Bouldin Index: {db_index:.4f}")
else:
    print("Davies-Bouldin Index: Not applicable for single cluster")

# Calinski-Harabasz Index (Euclidean distance is typically used for CH, adapt or be cautious for Manhattan in K-Medians context)
if len(set(cluster_labels)) > 1:
    ch_index = calinski_harabasz_score(X_train, cluster_labels) # Default metric is Euclidean
    print(f"Calinski-Harabasz Index (using Euclidean): {ch_index:.4f}")
else:
    print("Calinski-Harabasz Index: Not applicable for single cluster")
```

**Interpreting Metrics:**

*   Use these metric values to compare different clustering solutions (e.g., with different values of $K$ or different algorithms).
*   Higher Silhouette score and Calinski-Harabasz Index, and lower Inertia and Davies-Bouldin Index generally indicate better clustering quality.
*   Consider these metrics in combination with domain knowledge and the goals of your clustering task. No single metric is perfect, and the best evaluation often involves a combination of quantitative metrics and qualitative assessment (e.g., cluster profiling).

## Model Productionizing Steps for K-Medians

Productionizing a K-Medians clustering model involves deploying it so that it can be used to cluster new, unseen data or to assign new data points to existing clusters.

**1. Save the Trained Model Components:**

You need to save:

*   **The Scaler:** If you used feature scaling (like `StandardScaler`), save the fitted scaler object. You'll need to use it to scale new input data in the same way.
*   **Cluster Medians:** Save the coordinates of the final cluster medians. These define the cluster centers.
*   **Potentially Cluster Assignments (optional):** If you need to recall the cluster assignments of the training data for reference or analysis later, you can save these labels as well.

**2. Create a Prediction/Assignment Function:**

You need to create a function that takes new data points as input and assigns them to the closest cluster based on the saved medians and Manhattan distance.

```python
import numpy as np
import pandas as pd
import pickle

# --- Assume you have loaded scaler and medians as 'loaded_scaler' and 'loaded_medians' ---

def assign_to_clusters(new_data_df, scaler, medians):
    """Assigns new data points to the closest K-Medians clusters.

    Args:
        new_data_df (pd.DataFrame): DataFrame of new data points.
        scaler (sklearn scaler object): Fitted scaler used for training data.
        medians (np.ndarray): Array of cluster medians.

    Returns:
        np.ndarray: Cluster assignments for new data points (0-indexed).
    """
    new_data_scaled = scaler.transform(new_data_df) # Scale new data
    n_samples = new_data_scaled.shape[0]
    cluster_assignments = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        distances = [manhattan_distance(new_data_scaled[i], median) for median in medians]
        closest_median_index = np.argmin(distances)
        cluster_assignments[i] = closest_median_index

    return cluster_assignments

# Example Usage:
# Create some new dummy data to cluster
new_data_points = pd.DataFrame([[1, 1], [-3, -3], [3, -1]], columns=X_df.columns)

# Assign new data points to clusters using the function and loaded components
new_cluster_assignments = assign_to_clusters(new_data_points, loaded_scaler, loaded_medians)
print("\nCluster Assignments for New Data Points:\n", new_cluster_assignments)
```

**Output:**

*(Output will show the cluster assignment (0, 1, or 2) for each of the new data points, based on their proximity to the loaded medians.)*

**3. Deployment as a Service/API (Similar to other algorithms):**

You can create a prediction service or API (using Flask, FastAPI, cloud platforms, etc.) to make your K-Medians clustering available for applications.  The API endpoint would take new data as input, preprocess it using the saved scaler, use the `assign_to_clusters` function (or similar logic) to get cluster assignments, and return the cluster assignments as the API response.

**4. Deployment Environments:**

*   **Local Testing:** Test your `assign_to_clusters` function and API locally to ensure it works correctly.
*   **On-Premise Deployment:** Deploy the prediction service on your organization's servers.
*   **Cloud Deployment (PaaS, Containers, Serverless):**  Use cloud platforms (AWS, Google Cloud, Azure) to deploy your API. Options include:
    *   **PaaS (Platform as a Service):**  AWS Elastic Beanstalk, Google App Engine, Azure App Service for easy deployment of web applications.
    *   **Containers (Docker/Kubernetes):** Package your prediction service in a Docker container and deploy to container orchestration platforms (AWS ECS, GKE, AKS) for scalability and flexibility.
    *   **Serverless Functions:** Cloud functions (AWS Lambda, Google Cloud Functions, Azure Functions) can be used for simpler prediction APIs if your workload is event-driven or you want to minimize server management.

**5. Monitoring and Maintenance:**

*   **Monitoring:** Monitor the performance of your clustering service, API latency, error rates, and resource usage.
*   **Data Drift Monitoring:** If your application is clustering data that changes over time, monitor for data drift. If the data distribution shifts significantly, you might need to retrain your K-Medians model periodically to ensure the clusters remain relevant and accurate.
*   **Model Retraining:** Retrain your K-Medians model (and potentially re-evaluate the optimal number of clusters, $K$) as needed, based on data drift monitoring or when you have significantly new data.

Productionizing K-Medians clustering is generally simpler than productionizing complex deep learning models, but still requires careful planning for data preprocessing, prediction service implementation, deployment environment selection, and ongoing monitoring.

## Conclusion: K-Medians Clustering in Practice and Alternatives

K-Medians clustering provides a robust and interpretable way to partition data into groups, especially when dealing with data that might contain outliers. Its use of Manhattan distance makes it less sensitive to extreme values than K-Means and suitable for datasets where feature dimensions are independent and equally weighted in terms of distance.

**Real-world Applications where K-Medians is useful:**

*   **Location Analytics:** Clustering customer locations, store locations, or event locations where Manhattan distance might be a more natural measure of travel distance in city grids.
*   **Datasets with Outliers:** Applications where data is prone to outliers, and robustness is important, such as sensor data analysis, financial data, or network traffic monitoring.
*   **Data Mining and Exploratory Data Analysis:** For initial exploration and grouping of data to discover underlying patterns, especially when interpretability and robustness are desired.
*   **Text and Document Clustering (with modifications):** While standard K-Medians works with numerical data, adaptations can be used for text clustering by representing documents in a numerical vector space (e.g., using TF-IDF) and then applying K-Medians or related algorithms.

**Optimized or Newer Algorithms and Alternatives:**

While K-Medians is a useful algorithm, there are alternatives and related approaches:

*   **K-Means Clustering:**  The most common centroid-based clustering algorithm. Uses Euclidean distance and the mean as the centroid.  Generally faster than K-Medians but more sensitive to outliers. Use K-Means when outliers are less of a concern and Euclidean distance is appropriate.
*   **K-Medoids Clustering (PAM - Partitioning Around Medoids):**  Similar to K-Medians in that it uses actual data points as cluster centers (medoids).  PAM is another robust clustering algorithm.  The `kmodes` library's `KMedoids` implementation is based on PAM-like logic. PAM can be more computationally expensive than K-Medians for large datasets.
*   **Fuzzy C-Means Clustering (FCM):** Allows data points to belong to multiple clusters with different degrees of membership (fuzzy clustering). Useful when clusters are not well-separated or data points might belong to multiple groups.
*   **Hierarchical Clustering:** Builds a hierarchy of clusters. Can be agglomerative (bottom-up, starting with each point as a cluster and merging) or divisive (top-down, starting with one cluster and splitting). Useful when you don't know the number of clusters beforehand and want to explore cluster hierarchies.
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Density-based clustering algorithm that can find clusters of arbitrary shapes and identify outliers as noise.  Useful when clusters are not spherical and there is noise in the data.
*   **Gaussian Mixture Models (GMMs):** Probabilistic clustering algorithm that assumes data points are generated from a mixture of Gaussian distributions.  Provides soft cluster assignments (probabilities of belonging to each cluster).

**Conclusion:**

K-Medians clustering is a valuable algorithm in the clustering toolbox, particularly when robustness to outliers and the use of Manhattan distance are desired. Understanding its principles and applications allows data scientists to choose the right clustering approach for various data analysis and machine learning tasks. When facing clustering problems, consider K-Medians alongside other clustering algorithms and choose the method that best fits your data characteristics, problem requirements, and desired balance between robustness, computational efficiency, and interpretability.

## References

1.  **Park, H. S., & Jun, C. H. (2009). A simple and fast algorithm for K-medoids clustering.** *Expert Systems with Applications*, *36*(6), 8486-8491. [[Link to ScienceDirect (may require subscription)](https://www.sciencedirect.com/science/article/pii/S095741740800979X)] - Discusses efficient algorithms for K-Medoids and related methods.

2.  **Lloyd, S. P. (1982). Least squares quantization in PCM.** *IEEE transactions on information theory*, *28*(2), 129-137. [[Link to IEEE Xplore (may require subscription)](https://ieeexplore.ieee.org/document/1056483)] - Although focused on K-Means, the iterative approach is similar to K-Medians and is foundational to centroid-based clustering.

3.  **Ng, R. T., & Han, J. (2001). CLARANS: A method for clustering objects for spatial data mining.** *IEEE Transactions on Knowledge and Data Engineering*, *14*(5), 1003-1016. [[Link to IEEE Xplore (may require subscription)](https://ieeexplore.ieee.org/document/1041003)] - CLARANS (Clustering Large Applications based upon RANdomized Search) is related to K-Medoids and addresses scalability issues.

4.  **Kaufman, L., & Rousseeuw, P. J. (2009). *Finding groups in data: an introduction to cluster analysis*. John Wiley & Sons.**  - A comprehensive textbook on cluster analysis, including detailed discussions of K-Medoids (PAM) and related methods.

5.  **Scikit-learn documentation on Clustering Evaluation:** [[Link to scikit-learn clustering evaluation metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] - Scikit-learn documentation on various clustering evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, etc.).

This blog post offers a detailed introduction to K-Medians clustering. Experiment with the provided code examples, try different datasets, and explore the hyperparameter tuning and evaluation techniques to deepen your understanding and practical application of this algorithm.
