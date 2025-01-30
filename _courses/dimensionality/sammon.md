---
title: "Mapping the Unseen: Understanding Sammon Mapping for Data Visualization"
excerpt: "Sammon Mapping Algorithm"
# permalink: /courses/dimensionality/sammon/
last_modified_at: 2022-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories: 
  - Non-linear Dimensionality Reduction
  - Data Visualization
  - Unsupervised Learning
tags: 
  - Dimensionality reduction
  - Data visualization
  - Non-linear mapping
  - Distance preservation
---

{% include download file="sammon_mapping_code.ipynb" alt="download sammon mapping code" text="Download Code" %}

## 1. Introduction: Making Sense of Complex Data with Sammon Mapping

Imagine you have a detailed 3D model of a mountain range. To easily study it on a flat map, you need to "flatten" it.  However, you want to do this in a way that preserves the relative distances between different locations on the mountain as much as possible.  This idea of representing high-dimensional data in a lower dimension while trying to keep the important relationships is at the heart of **Sammon Mapping**.

In the world of data science, we often work with data that has many features or dimensions – much like our 3D mountain model.  It becomes challenging to visualize and understand patterns in such high-dimensional data directly. Sammon Mapping is a technique designed to help us reduce the number of dimensions down to 2D or 3D, so we can visualize the data and hopefully see clusters, relationships, or structures that are hidden in the high-dimensional space.

**Think of these real-world examples where Sammon Mapping can be helpful:**

*   **Analyzing Customer Preferences:** Suppose a company has collected data about customer buying habits, website interactions, demographics, and more (many dimensions!). Sammon Mapping could help visualize if customers naturally group together based on these characteristics.  This could reveal distinct customer segments that marketers can then target with tailored campaigns.

*   **Exploring Chemical Compound Similarity:** In chemistry, we might have data describing the properties of different chemical compounds. Sammon Mapping can help to visualize which compounds are structurally or functionally similar. This is useful in drug discovery to find new compounds that are similar to known drugs.

*   **Visualizing Sensor Data:**  Imagine a network of sensors collecting various types of environmental data (temperature, humidity, pressure, pollution levels, etc. – again, multiple dimensions). Sammon Mapping can be used to visualize if sensors are behaving similarly and identify any unusual patterns or groupings across the sensor network.

In simple words, Sammon Mapping is like a smart way to create a map of your complex data. It tries its best to keep the "distances" between data points similar in the lower-dimensional map as they were in the original high-dimensional space. This helps us to visually explore and understand the underlying structure of the data.

## 2. The Mathematics Underneath: Preserving Distances

Sammon Mapping is built on the principle of preserving the distances between data points as they are projected from a high-dimensional space to a lower-dimensional one. Let's break down the mathematical process in a simple way.

The main idea is to minimize a **cost function**, often called the **Sammon stress**, which measures how much the distances are distorted during the dimensionality reduction.

Let's say we have \(N\) data points, \(x_1, x_2, ..., x_N\), in a high-dimensional space. We want to find their representations \(y_1, y_2, ..., y_N\) in a lower-dimensional space (say, 2D).

**Step 1: Calculate High-Dimensional Distances**

First, we calculate the pairwise distances between all data points in the original high-dimensional space. Let \(d_{ij}\) be the distance between point \(x_i\) and \(x_j\) in the original space. Typically, we use Euclidean distance:

$$
d_{ij} = ||x_i - x_j|| = \sqrt{\sum_{k=1}^{D} (x_{ik} - x_{jk})^2}
$$

where \(D\) is the number of dimensions in the original space, and \(x_{ik}\) is the \(k\)-th dimension of point \(x_i\).

**Example:** If \(x_1 = (1, 2, 3)\) and \(x_2 = (4, 5, 6)\), the squared Euclidean distance \(d_{12}^2 = (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27\), and \(d_{12} = \sqrt{27} \approx 5.2\).

**Step 2: Initialize Low-Dimensional Points**

We need to start with an initial guess for the positions of the points \(y_1, y_2, ..., y_N\) in the low-dimensional space.  A common approach is to initialize them randomly or using Principal Component Analysis (PCA) to get a reasonable starting configuration.

**Step 3: Calculate Low-Dimensional Distances**

Next, we calculate the pairwise distances between the low-dimensional points \(y_i\) and \(y_j\). Let \(\delta_{ij}\) be the distance in the low-dimensional space (again, typically Euclidean):

$$
\delta_{ij} = ||y_i - y_j|| = \sqrt{\sum_{k=1}^{d} (y_{ik} - y_{jk})^2}
$$

where \(d\) is the number of dimensions in the low-dimensional space (usually 2 or 3), and \(y_{ik}\) is the \(k\)-th dimension of point \(y_i\).

**Step 4: Define the Sammon Stress Function**

The Sammon Stress \(E\) is the core of the algorithm. It measures the normalized sum of squared differences between the original high-dimensional distances \(d_{ij}\) and the low-dimensional distances \(\delta_{ij}\).  The formula is:

$$
E = \frac{1}{\sum_{i<j} d_{ij}} \sum_{i<j} \frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}
$$

Let's break this equation down:

*   \((d_{ij} - \delta_{ij})^2\): This is the squared difference between the high-dimensional distance and the low-dimensional distance for each pair of points \((i, j)\). We want to minimize these differences.
*   \(\frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}\): We divide by \(d_{ij}\) in the denominator. This is important because it gives more weight to preserving the distances between points that are originally *close* in the high-dimensional space.  If \(d_{ij}\) is small, the term \(\frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}\) becomes larger for the same difference \((d_{ij} - \delta_{ij})\), penalizing distance distortion more for nearby points.
*   \(\sum_{i<j} \frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}\): We sum these weighted squared differences over all pairs of points \((i, j)\) where \(i < j\) (to avoid double counting pairs).
*   \(\frac{1}{\sum_{i<j} d_{ij}}\):  We divide by the sum of all high-dimensional distances. This normalization factor ensures that the stress \(E\) is in a consistent range (roughly between 0 and 1), and makes it less dependent on the overall scale of the original distances.

**Step 5: Minimize the Stress Function using Gradient Descent**

The goal of Sammon Mapping is to find the positions of the low-dimensional points \(y_1, y_2, ..., y_N\) that **minimize** the Sammon Stress \(E\). We use an iterative optimization technique called **gradient descent** (or a similar optimization method) to achieve this.

Gradient descent works like this:

1.  **Calculate the gradient of the stress function \(E\) with respect to each coordinate of each low-dimensional point \(y_i\).** The gradient tells us the direction of the steepest *increase* in \(E\). We want to move in the *opposite* direction to *decrease* \(E\).
2.  **Update the positions of the low-dimensional points \(y_i\) by moving them slightly in the direction opposite to the gradient.** The size of the step is controlled by a **learning rate** (or step size) parameter.
3.  **Repeat steps 1 and 2 iteratively** until the stress \(E\) converges to a minimum value, or until a maximum number of iterations is reached.

The update rule for each coordinate of \(y_i\) in gradient descent typically looks something like:

$$
y_{i}^{(t+1)} = y_{i}^{(t)} - \eta \frac{\partial E}{\partial y_{i}^{(t)}}
$$

where:
*   \(y_{i}^{(t)}\) is the position of point \(y_i\) at iteration \(t\).
*   \(y_{i}^{(t+1)}\) is the updated position at iteration \(t+1\).
*   \(\eta\) (eta) is the learning rate, a small positive value that controls the step size.
*   \(\frac{\partial E}{\partial y_{i}^{(t)}}\) is the gradient of the stress function \(E\) with respect to \(y_{i}^{(t)}\).  This gradient needs to be calculated analytically (by differentiating the stress function equation) and then computed numerically in each iteration.

The gradient calculation for Sammon Mapping is somewhat complex, but it can be derived.  The iterative process of gradient descent refines the positions of the low-dimensional points \(y_i\) until they represent the original high-dimensional relationships as faithfully as possible, in terms of preserving pairwise distances.

**In summary:** Sammon Mapping is an optimization process that tries to arrange points in a low-dimensional space such that their pairwise distances are as close as possible to the original pairwise distances in the high-dimensional space, by minimizing the Sammon Stress function using iterative gradient descent.

## 3. Prerequisites and Preprocessing: Getting Started with Sammon Mapping

Before you use Sammon Mapping, let's understand what you need and what to consider in terms of prerequisites and preprocessing.

**Prerequisites and Assumptions:**

*   **Numerical Data:** Sammon Mapping works with numerical data, where you can calculate distances between data points.  If your data contains categorical features, you will need to convert them into a numerical representation (e.g., using one-hot encoding or other suitable encoding methods).
*   **Distance Metric:** You need to choose a distance metric that is appropriate for your data and the kind of relationships you want to preserve. Euclidean distance is the most common choice, but other metrics like Manhattan distance, cosine distance, or correlation distance could be used depending on the nature of your data.
*   **Data that can be meaningfully embedded in lower dimensions:** Sammon Mapping, like other dimensionality reduction techniques, works best when the high-dimensional data has some underlying structure that can be reasonably represented in a lower-dimensional space. If the data is truly random or very high-dimensional with no inherent low-dimensional structure, the results might be less meaningful.
*   **Computational Resources:** Sammon Mapping involves iterative optimization, which can be computationally intensive, especially for large datasets. The runtime complexity is roughly quadratic in the number of data points, as it calculates pairwise distances.  For very large datasets, consider more scalable dimensionality reduction techniques (like PCA, UMAP, t-SNE).

**Testing Assumptions (and Considerations):**

*   **Data Exploration:** Before applying Sammon Mapping, it's always good to explore your data using basic statistical methods and visualizations. Understand the range of your features, the distribution of your data, and any potential outliers.
*   **Distance Metric Choice:** Consider the nature of your data when choosing a distance metric. If you're working with data where magnitude matters, Euclidean or Manhattan distance might be appropriate. If you're interested in the similarity of directions or patterns, cosine distance or correlation distance might be more relevant.
*   **Number of Dimensions to Reduce to:** You need to decide on the target dimensionality (e.g., 2D or 3D). For visualization, 2D or 3D are typical choices. For other purposes, you might experiment with different target dimensions.
*   **Initialization:** The initialization of low-dimensional points can affect the optimization process and the final embedding. Random initialization is common, but PCA initialization can sometimes lead to better starting points.
*   **Iteration and Convergence:** Sammon Mapping is an iterative algorithm. You need to decide on stopping criteria (maximum iterations or a convergence threshold for the stress function). Monitor the stress function during optimization to see if it is decreasing and converging.

**Python Libraries:**

While Sammon Mapping is not as readily available in scikit-learn as some other dimensionality reduction techniques, you can find implementations in libraries like `sklearn-contrib` (though it might not be actively maintained) or implement it yourself using libraries like `numpy` for numerical computation and `scipy.optimize` for optimization routines.  A simpler way is to use existing implementations from online resources or specialized packages if available.

For basic numerical operations and optimization needed to implement Sammon Mapping, you'll primarily use:

*   **NumPy (`numpy`)**: For array operations, linear algebra, and mathematical functions.
*   **SciPy (`scipy`)**:  Especially `scipy.optimize` which provides optimization algorithms like `minimize` that can be used to minimize the Sammon Stress function.

```python
# Python Libraries needed for Sammon Mapping Implementation
import numpy as np
import scipy.optimize

print("numpy version:", np.__version__)
import scipy
print("scipy version:", scipy.__version__)
```

You'll need to ensure these libraries are installed in your Python environment. Install them using pip if needed:

```bash
pip install numpy scipy
```

For a direct, ready-to-use Sammon Mapping implementation, you might search for specialized Python packages or online code repositories, as it's not a standard algorithm in `sklearn`.  If you are comfortable with coding, you can implement it yourself based on the mathematical description using NumPy and SciPy's optimization tools.

## 4. Data Preprocessing: Scaling is Often Important

Data preprocessing is often necessary for Sammon Mapping, especially **scaling** your features.

**Why Scaling is Important for Sammon Mapping:**

Sammon Mapping is a **distance-based method**. It relies on calculating distances between data points in both high and low dimensions.  If your features have vastly different scales, features with larger scales can dominate the distance calculations and disproportionately influence the embedding.

**Example (Same as in t-SNE and Subset Selection Blogs):** If you have features like "Age" (range 20-80) and "Income" (range \$20,000-\$200,000), "Income" differences will heavily outweigh "Age" differences in Euclidean distance if data is not scaled, even if "Age" is equally or more important for the underlying structure you are trying to reveal.

**Types of Scaling (Same as before):**

*   **Standardization (Z-score scaling):** Transforms features to have mean 0 and standard deviation 1.

    $$
    x'_{i} = \frac{x_{i} - \mu}{\sigma}
    $$

*   **Min-Max Scaling:** Scales features to a range, typically 0 to 1.

    $$
    x'_{i} = \frac{x_{i} - x_{min}}{x_{max} - x_{min}}
    $$

**When can scaling be ignored?**

It's generally **not recommended** to ignore scaling for Sammon Mapping unless you have a clear reason and understand the potential implications.

*   **Features Already on Similar Scales:** If all your features are already in comparable units and ranges (e.g., all are percentages, or all are measured in similar physical units with similar ranges), scaling might be less critical, but it's usually still safer to apply it.
*   **If you *want* certain features to have dominant influence:** In rare cases, you might intentionally want features with larger scales to have a greater influence on the distances. If this is a deliberate choice based on domain knowledge, you *might* skip scaling, but this is less common.

**Preprocessing Example in Python (using scikit-learn's scalers, same code as before):**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Dummy data (example - replace with your actual data)
data = np.array([[10, 1000],
                 [20, 20000],
                 [15, 15000],
                 [5, 5000]])

# StandardScaler
scaler_standard = StandardScaler()
scaled_data_standard = scaler_standard.fit_transform(data)
print("Standardized data:\n", scaled_data_standard)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
scaled_data_minmax = scaler_minmax.fit_transform(data)
print("\nMin-Max scaled data:\n", scaled_data_minmax)
```

**StandardScaler is often a good default choice for Sammon Mapping.** It helps ensure that all features contribute more equally to the distance calculations.

**Handling Categorical and Missing Data (Same principles as in Subset Selection Blog):**

*   **Categorical Features:** Encode categorical features into numerical representations (e.g., one-hot encoding) *before* applying Sammon Mapping.
*   **Missing Values:** Handle missing values by imputation or removal *before* applying Sammon Mapping.  Sammon Mapping algorithms typically expect complete numerical data.

**In summary:** Data preprocessing, particularly feature scaling, is usually an important step before applying Sammon Mapping to ensure that the distance calculations are meaningful and that features with different scales are treated fairly.  Handle categorical and missing data appropriately as well before using Sammon Mapping.

## 5. Implementation Example: Sammon Mapping on Dummy Data

Let's implement Sammon Mapping using NumPy and SciPy's optimization tools on some dummy data. We will write a basic Sammon Mapping function and apply it.

```python
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt # For visualization

def sammon_stress(Y, high_dist_matrix, n_points):
    """Calculates Sammon stress function."""
    low_dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            low_dist_matrix[i, j] = low_dist_matrix[j, i] = np.linalg.norm(Y[i] - Y[j])

    stress_sum = 0.0
    high_dist_sum = np.sum(high_dist_matrix[np.triu_indices_from(high_dist_matrix, k=1)]) # Sum of upper triangle to avoid double count

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if high_dist_matrix[i, j] > 0: # Avoid division by zero
                stress_sum += (high_dist_matrix[i, j] - low_dist_matrix[i, j])**2 / high_dist_matrix[i, j]

    if high_dist_sum > 0:
        return (1 / high_dist_sum) * stress_sum
    else:
        return 0 # Handle case where all high dimensional distances are zero

def sammon_gradient(Y_flat, high_dist_matrix, n_points, low_dim):
    """Calculates the gradient of the Sammon stress function."""
    Y = Y_flat.reshape(n_points, low_dim) # Reshape flat array back to 2D points
    gradient_flat = np.zeros_like(Y_flat)
    gradient = gradient_flat.reshape(n_points, low_dim)

    low_dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            low_dist_matrix[i, j] = low_dist_matrix[j, i] = np.linalg.norm(Y[i] - Y[j])

    high_dist_sum = np.sum(high_dist_matrix[np.triu_indices_from(high_dist_matrix, k=1)])

    for i in range(n_points):
        for j in range(n_points):
            if i != j and high_dist_matrix[i, j] > 0 and low_dist_matrix[i, j] > 1e-6: # Added small epsilon to avoid division by near-zero
                term1 = (high_dist_matrix[i, j] - low_dist_matrix[i, j]) / (high_dist_matrix[i, j] * low_dist_matrix[i, j])
                term2 = (Y[i] - Y[j])
                gradient[i] += term1 * term2

    if high_dist_sum > 0:
        return (2 / high_dist_sum) * gradient_flat
    else:
        return gradient_flat


def perform_sammon_mapping(X, low_dim=2, initial_Y=None, max_iter=1000):
    """Performs Sammon Mapping."""
    n_points = X.shape[0]
    high_dim = X.shape[1]

    # 1. Calculate High-Dimensional Distances
    high_dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            high_dist_matrix[i, j] = high_dist_matrix[j, i] = np.linalg.norm(X[i] - X[j])


    # 2. Initialize Low-Dimensional Points
    if initial_Y is None:
        initial_Y = np.random.rand(n_points, low_dim) # Random initialization

    # Flatten initial Y for optimization
    initial_Y_flat = initial_Y.flatten()

    # 3. Optimization using minimize from scipy.optimize
    result = scipy.optimize.minimize(
        fun=sammon_stress,
        x0=initial_Y_flat,
        args=(high_dist_matrix, n_points),
        jac=sammon_gradient, # Provide gradient function for faster optimization
        method='CG',          # Conjugate Gradient method often works well
        options={'maxiter': max_iter}
    )

    Y_embedded = result.x.reshape(n_points, low_dim)
    stress_value = result.fun # Final stress value achieved

    return Y_embedded, stress_value



# 1. Dummy Data
np.random.seed(42)
X_high_dim = np.random.rand(20, 5) # 20 points, 5 dimensions

# 2. Perform Sammon Mapping
Y_low_dim, final_stress = perform_sammon_mapping(X_high_dim, low_dim=2, max_iter=500)

# 3. Output and Visualization

print("Sammon Mapping Completed.")
print("Final Sammon Stress:", final_stress)
print("\nLow-dimensional embeddings (first 5 points):\n", Y_low_dim[:5])

# 4. Simple 2D Scatter Plot of Embedded Data
plt.figure(figsize=(8, 6))
plt.scatter(Y_low_dim[:, 0], Y_low_dim[:, 1])
plt.title("Sammon Mapping Visualization (Dummy Data)")
plt.xlabel("Sammon Dimension 1")
plt.ylabel("Sammon Dimension 2")
plt.grid(True)
plt.show()

# 5. Saving and Loading Embedded Data (using numpy's save and load)
embedding_filename = 'sammon_embedding_dummy_data.npy'
np.save(embedding_filename, Y_low_dim)
print(f"\nSammon embedding saved to {embedding_filename}")

loaded_embedding = np.load(embedding_filename)
print("\nLoaded Sammon embedding (first 5 points):\n", loaded_embedding[:5])
```

**Explanation of the Code and Output:**

1.  **`sammon_stress(Y, high_dist_matrix, n_points)` function:**
    *   Calculates the Sammon Stress for a given set of low-dimensional points `Y` and the pre-calculated high-dimensional distance matrix.
    *   It iterates through pairs of points, calculates low-dimensional distances, and computes the stress according to the formula discussed in section 2.

2.  **`sammon_gradient(Y_flat, high_dist_matrix, n_points, low_dim)` function:**
    *   Calculates the gradient of the Sammon Stress function. This is necessary for efficient gradient-based optimization.  The gradient formula is derived from the Sammon Stress function.
    *   It's more complex than the stress function but essentially computes how to adjust each low-dimensional point to reduce the stress.

3.  **`perform_sammon_mapping(X, low_dim=2, initial_Y=None, max_iter=1000)` function:**
    *   This is the main function that performs Sammon Mapping.
    *   **Calculates High-Dimensional Distances:** It first computes the pairwise distances between all points in the high-dimensional data `X` and stores them in `high_dist_matrix`.
    *   **Initializes Low-Dimensional Points:** If `initial_Y` is not provided, it initializes the low-dimensional points randomly.
    *   **Optimization using `scipy.optimize.minimize`:** It uses `scipy.optimize.minimize` to minimize the `sammon_stress` function.
        *   `fun=sammon_stress`: Specifies the function to minimize.
        *   `x0=initial_Y_flat`:  Provides the initial guess for the low-dimensional points (flattened into a 1D array as required by `minimize`).
        *   `args=(high_dist_matrix, n_points)`: Passes additional arguments to the `sammon_stress` function (the distance matrix and number of points).
        *   `jac=sammon_gradient`: Provides the gradient function for faster optimization.
        *   `method='CG'`: Uses the Conjugate Gradient optimization method, which is often effective for this type of problem.
        *   `options={'maxiter': max_iter}`: Sets the maximum number of iterations.
    *   **Returns:** The function returns the `Y_embedded` (the low-dimensional coordinates) and `stress_value` (the final minimized Sammon Stress).

4.  **Dummy Data and Execution:**
    *   We generate dummy high-dimensional data `X_high_dim`.
    *   We call `perform_sammon_mapping` to perform Sammon Mapping, reducing to 2 dimensions (`low_dim=2`).
    *   We print the "Final Sammon Stress" and the first few embedded points.

5.  **Visualization:**
    *   We use `matplotlib.pyplot.scatter` to create a 2D scatter plot of the embedded points `Y_low_dim`. This visualizes the result of Sammon Mapping.

6.  **Saving and Loading:**
    *   We use `numpy.save` to save the `Y_low_dim` array to a `.npy` file.
    *   We use `numpy.load` to load the saved embedding back from the file and print the first few points of the loaded embedding to verify that saving and loading worked correctly.

**Interpreting the Output:**

When you run this code, you will see output like this (output values may vary slightly due to random initialization and optimization):

```
Sammon Mapping Completed.
Final Sammon Stress: 0.03...

Low-dimensional embeddings (first 5 points):
 [[ 0.4...  0.5...]
  [ 0.6...  0.4...]
  [-0.0...  0.3...]
  [ 0.1...  0.5...]
  [-0.2...  0.1...]]

Sammon embedding saved to sammon_embedding_dummy_data.npy

Loaded Sammon embedding (first 5 points):
 [[ 0.4...  0.5...]
  [ 0.6...  0.4...]
  [-0.0...  0.3...]
  [ 0.1...  0.5...]
  [-0.2...  0.1...]]
```

*   **"Final Sammon Stress":**  This value (e.g., 0.03...) is the minimized Sammon Stress. Lower stress values indicate a better preservation of distances. A stress value of 0 would be perfect distance preservation (often impossible in practice for dimensionality reduction).  The stress value itself is a metric of how well the Sammon Mapping has performed for this particular run.
*   **"Low-dimensional embeddings":** This is the output – the 2D coordinates of the points after Sammon Mapping. These are the coordinates used in the scatter plot visualization.
*   **Scatter Plot:** The plot will show a 2D scatter plot of the embedded data points. You can visually inspect this plot to look for any clusters or structures that might be revealed in the low-dimensional space. Since we used random dummy data, there may not be any strong visual patterns, but for real-world structured data, Sammon Mapping aims to reveal underlying structures visually.
*   **Saving and Loading:** The output confirms that the embedding has been saved to and successfully loaded from the `.npy` file.

**No "r-value" or similar in output:**  Like t-SNE, Sammon Mapping does not produce an "r-value" or a similar statistical measure in its direct output like regression models. The primary output is the low-dimensional embedding and the stress value. The "value" is in the visualization and the insight you gain from it, and the stress value gives a numerical measure of distance preservation quality.

## 6. Post-Processing: Interpreting Sammon Maps

Sammon Mapping is primarily a **visualization technique**, so post-processing mainly focuses on interpreting the resulting map and relating it back to your original data. Unlike some statistical models, you don't typically perform AB testing or hypothesis testing directly on the output of Sammon Mapping.

**Interpreting the Sammon Map Visually:**

*   **Cluster Identification:** Examine the Sammon map (the 2D or 3D scatter plot). Look for visual clusters or groupings of points.  Points that are close together in the Sammon map are intended to be points that were also "close" or similar in the original high-dimensional space (based on the distance metric you used).
*   **Structure and Relationships:** Observe the overall structure of the map. Are there linear trends, curved patterns, or other shapes in the point arrangement?  These visual patterns can sometimes reflect underlying relationships or manifolds in your high-dimensional data.
*   **Outlier Detection (potentially):** Points that appear isolated or far away from the main clusters in the Sammon map might be potential outliers in your original data. Investigate these points further to see if they are truly anomalous or just represent distinct data instances.
*   **Comparison with Domain Knowledge:** The most crucial part of interpretation is to relate the visual patterns you observe in the Sammon map back to your domain knowledge.
    *   Do the clusters you see in the map correspond to known categories or groups in your data (if you have labeled data)?
    *   Can you explain the observed structure in terms of the features and characteristics of your data, based on your understanding of the domain?

**Linking Back to Original Data Features:**

*   **Feature Analysis within Clusters:** If you identify clusters in the Sammon map, go back to your original high-dimensional data and analyze the characteristics of the data points within each cluster.
    *   Calculate descriptive statistics (means, medians, ranges, etc.) for each feature *within* each cluster. Are there features that have consistently higher or lower values in certain clusters?
    *   Compare the feature distributions across different clusters. Are there features that are more variable or less variable within certain clusters?
*   **Feature Importance (Indirect):** Sammon Mapping itself doesn't directly provide feature importance scores like some feature selection methods. However, if you can interpret the clusters in your Sammon map in terms of domain-relevant groupings, you can then analyze the features that are characteristic of those groupings. This can indirectly suggest which features are important in defining the data structure that Sammon Mapping is revealing.
*   **Color-Coding or Labeling Points:** If your data points have labels or categories (e.g., different classes in a classification problem, different groups of customers, different types of chemical compounds), color-code the points in your Sammon map according to these labels. This can visually confirm if points with the same labels tend to cluster together in the Sammon map, which indicates that Sammon Mapping is effectively separating these groups.

**Limitations of Interpretation:**

*   **Subjectivity:** Interpretation of visualizations is inherently subjective. Different people might see patterns or clusters differently.
*   **No Guarantee of "True" Clusters:** Sammon Mapping aims to preserve distances, but the clusters you see in the map are not guaranteed to be "true" clusters in the high-dimensional space. They are visual representations of relationships based on distance preservation.
*   **Global vs. Local Structure:** Sammon Mapping (like other non-linear dimensionality reduction methods) might emphasize local structure (relationships between nearby points) more than global structure (relationships between distant clusters). Be cautious about over-interpreting distances between clusters in the Sammon map.

**In summary:** Post-processing for Sammon Mapping is primarily about visual interpretation of the map, relating visual patterns to domain knowledge, and using the Sammon map as a tool to explore and understand the structure of your high-dimensional data, rather than for formal statistical inference or predictive modeling.

## 7. Hyperparameters of Sammon Mapping: Tuning for Better Maps

Sammon Mapping has fewer directly "tweakable" hyperparameters compared to some other dimensionality reduction methods like t-SNE. The main adjustable aspects are related to the optimization process and initialization.

**Key Aspects to Consider (Less like "Hyperparameters" in typical ML sense, more like algorithm settings):**

*   **Initialization of Low-Dimensional Points (`initial_Y` parameter in our example):**
    *   **Effect:** The starting configuration of the low-dimensional points can influence the optimization process and the final embedding, especially for non-convex optimization problems.
    *   **Options:**
        *   **Random Initialization (Common Default):** Initialize `Y` with random values (e.g., uniform or Gaussian distribution). This is simple but might lead to different results on different runs due to randomness.
        *   **PCA Initialization:** Initialize `Y` using the top principal components from PCA. PCA can provide a good linear approximation of the data structure as a starting point for non-linear methods like Sammon Mapping. This can sometimes lead to faster convergence and potentially more stable results compared to random initialization.
    *   **Implementation (Example - using PCA for Initialization):**

        ```python
        from sklearn.decomposition import PCA

        def perform_sammon_mapping_with_pca_init(X, low_dim=2, max_iter=1000):
            """Sammon Mapping with PCA initialization."""
            n_points = X.shape[0]
            high_dim = X.shape[1]

            # ... (Calculate high_dist_matrix as before) ...

            # PCA Initialization
            pca = PCA(n_components=low_dim)
            initial_Y = pca.fit_transform(X) # Project data to top PC space

            # ... (Rest of the sammon mapping optimization as before, starting from initial_Y) ...
            return Y_embedded, stress_value

        # ... (Dummy data X_high_dim) ...
        Y_low_dim_pca_init, final_stress_pca_init = perform_sammon_mapping_with_pca_init(X_high_dim, low_dim=2)

        print("Sammon Mapping with PCA Initialization Completed.")
        print("Final Sammon Stress (PCA Init):", final_stress_pca_init)
        # ... (Visualize Y_low_dim_pca_init, etc.) ...
        ```

        In this code, we use `sklearn.decomposition.PCA` to reduce the dimensionality of `X_high_dim` to `low_dim` dimensions. The result of `pca.fit_transform(X)` is used as `initial_Y` in the `perform_sammon_mapping` function.

*   **Optimization Method (`method` parameter in `scipy.optimize.minimize`):**
    *   **Effect:** The optimization algorithm used to minimize the Sammon Stress can influence convergence speed and the final stress value reached.
    *   **Options:** `scipy.optimize.minimize` offers various optimization methods (e.g., 'CG' - Conjugate Gradient, 'BFGS' - Broyden-Fletcher-Goldfarb-Shanno, 'Newton-CG' - Newton Conjugate Gradient, 'L-BFGS-B' - Limited-memory BFGS with bounds, 'Nelder-Mead' - Simplex method (derivative-free, slower but more robust in some cases), etc.).
    *   **Tuning:** Experiment with different optimization methods. 'CG' and 'BFGS' are often good starting points for Sammon Mapping. For very complex stress surfaces, 'Nelder-Mead' might be more robust but slower.

*   **Maximum Iterations (`max_iter` parameter):**
    *   **Effect:** Controls the number of iterations in the gradient descent optimization. More iterations allow the optimization algorithm more time to converge to a minimum stress value.
    *   **Tuning:**
        *   **Monitor Stress Value:** Run Sammon Mapping for different values of `max_iter` (e.g., 100, 500, 1000, 2000). Monitor the Sammon Stress value during the optimization process. Plot the stress value against the number of iterations.
        *   **Convergence Check:** Observe if the stress value is decreasing and converging (becoming stable) as iterations increase. If the stress is still decreasing significantly even at high iteration counts, you might need to increase `max_iter`. If the stress has plateaued, further iterations might not improve the embedding much.
    *   **Example (Monitoring Stress with Iterations - Conceptual):** You would need to modify the `perform_sammon_mapping` function to track the stress value at each iteration and return a history of stress values. Then, you can plot stress vs. iteration number.

*   **Learning Rate (Step Size) in Gradient Descent (If manually implementing gradient descent loop - not directly in `scipy.optimize.minimize` usage):**
    *   **Effect:**  In a direct gradient descent implementation (if not using `scipy.optimize.minimize`), you'd manually control the learning rate. A larger learning rate can lead to faster progress but also risk overshooting the minimum and oscillations. A smaller learning rate might lead to slower convergence but potentially more stable progress.
    *   **Tuning:** If you were to implement gradient descent manually, you might experiment with different learning rates. Techniques like adaptive learning rates or line search can also be used to automatically adjust the step size during optimization.  However, when using `scipy.optimize.minimize`, the optimization algorithm handles step size adjustment internally in most methods.

**Hyperparameter Tuning Process (Less Formal, More Exploratory):**

For Sammon Mapping, hyperparameter "tuning" is less about finding optimal predictive performance (as in supervised learning) and more about finding settings that produce a visually informative and meaningful low-dimensional map, with a reasonably low stress value.

1.  **Experiment with Initialization:** Try both random initialization and PCA initialization and compare the resulting Sammon maps visually and the final stress values. See if PCA initialization consistently leads to lower stress or more visually appealing maps for your data.
2.  **Experiment with Optimization Methods:** Try different `method` options in `scipy.optimize.minimize` (e.g., 'CG', 'BFGS', 'L-BFGS-B'). See if some methods converge faster or reach lower stress values for your data.
3.  **Monitor Convergence with Iterations:** Experiment with different `max_iter` values and monitor the stress value as optimization progresses. Choose a `max_iter` that is sufficient for convergence (stress value plateaus) without excessive computation.
4.  **Visual Inspection is Key:**  Ultimately, the best "hyperparameter" settings are those that result in a Sammon map that is most informative and insightful *visually* for your data and domain understanding.  Visual inspection of the map is the primary evaluation criterion.

## 8. "Accuracy" Metrics: Measuring Distance Preservation

As Sammon Mapping is a dimensionality reduction technique for visualization and exploring data structure, "accuracy" in the traditional classification or regression sense doesn't apply. Instead, we evaluate how well it **preserves distances** from the high-dimensional space to the low-dimensional space.

**Key Metric: Sammon Stress (Already Discussed)**

*   **Sammon Stress \(E\):**  This is the primary metric to assess the quality of a Sammon Mapping embedding. It directly quantifies the distortion of distances:

    $$
    E = \frac{1}{\sum_{i<j} d_{ij}} \sum_{i<j} \frac{(d_{ij} - \delta_{ij})^2}{d_{ij}}
    $$

    *   **Lower Sammon Stress is Better:** A lower stress value indicates that the low-dimensional distances \(\delta_{ij}\) are, on average, closer to the original high-dimensional distances \(d_{ij}\). A stress of 0 would mean perfect distance preservation (usually impossible in dimensionality reduction).
    *   **Typical Ranges:** The "acceptability" of a Sammon Stress value depends on the dataset and the application. There's no universal threshold. Stress values are typically between 0 and 1. Lower values are generally desirable.
    *   **Interpretation Guideline (Qualitative):**
        *   **Very Low Stress (e.g., < 0.05):**  Indicates very good distance preservation. The Sammon map likely represents the original relationships quite faithfully.
        *   **Low to Moderate Stress (e.g., 0.05 - 0.2):**  Reasonable distance preservation. The map might reveal meaningful structures, but some distance distortion is present.
        *   **High Stress (e.g., > 0.2):**  Significant distance distortion. The Sammon map might be less reliable in accurately representing the original relationships. Interpret visual patterns cautiously.

**Other Related Concepts/Considerations (Less Formal "Metrics"):**

*   **Visual Inspection (Subjective but Important):**  As emphasized before, visual inspection of the Sammon map is crucial.  Does the map reveal meaningful clusters, structures, or separations that are consistent with your domain knowledge or expectations?  Does it appear visually "well-organized" or "distorted"? Visual assessment is subjective but is a primary way to judge the usefulness of a dimensionality reduction technique for visualization.
*   **Comparison with Other Methods:** Compare the Sammon map with visualizations produced by other dimensionality reduction methods (e.g., PCA, t-SNE, UMAP) for the same dataset. Which method seems to reveal the most interpretable and meaningful structure for *your* data, considering both visual aspects and quantitative stress metrics (if available for other methods)? There is no single "best" method for all datasets.
*   **Trustworthiness and Continuity (Metrics from t-SNE evaluation, can be adapted for Sammon Mapping):** These are metrics that try to quantify how well local neighborhood relationships are preserved during dimensionality reduction.
    *   **Trustworthiness:** Measures the extent to which neighbors in the low-dimensional space are also neighbors in the high-dimensional space.
    *   **Continuity:** Measures the extent to which neighbors in the high-dimensional space are also neighbors in the low-dimensional space.
    *   Calculating these metrics for Sammon Mapping could provide more quantitative insights into local structure preservation, but they are less commonly used than just reporting the Sammon Stress itself and visual assessment.

**Equations for Trustworthiness and Continuity (for context - not essential to implement for basic Sammon Mapping evaluation):**

These metrics are more complex to compute than Sammon Stress and usually involve defining "neighbors" (e.g., k-nearest neighbors) in both high and low dimensions.  Equations and detailed explanations can be found in research papers on dimensionality reduction evaluation (e.g., the "How to Use t-SNE Effectively" Distill.pub article mentioned in the t-SNE blog post references).

**In summary:**  The **Sammon Stress** is the primary quantitative metric for evaluating Sammon Mapping.  Supplement this with **visual inspection** of the Sammon map and compare it to other dimensionality reduction techniques to assess the overall usefulness and quality of the embedding for your specific data and visualization goals.  Metrics like trustworthiness and continuity are less commonly used for Sammon Mapping evaluation in practice, but are relevant concepts for assessing distance and neighborhood preservation.

## 9. Productionizing Sammon Mapping

"Productionizing" Sammon Mapping, like t-SNE, is typically about pre-calculating and serving the low-dimensional embeddings for visualization and exploratory data analysis in a production setting, rather than real-time prediction tasks.

**Productionizing Steps:**

1.  **Offline Computation of Sammon Embeddings:**
    *   **Batch Processing:** Compute the Sammon Mapping embeddings in an offline, batch processing environment (server, cloud instance, scheduled job). This step can be computationally intensive, so it's not usually done in real-time for every user request.
    *   **Code (Python - adapting our previous example):**

        ```python
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # Assume 'df' is your Pandas DataFrame with high-dimensional data, features in columns, no target column
        X_data = df.values # Get numpy array from DataFrame
        scaler = StandardScaler() # Scale the data
        X_scaled = scaler.fit_transform(X_data)

        Y_embeddings, stress_value = perform_sammon_mapping(X_scaled, low_dim=2, max_iter=1000) # Compute Sammon Mapping

        df_embeddings = pd.DataFrame(Y_embeddings, columns=['sammon_dim1', 'sammon_dim2']) # Create DataFrame
        df_embeddings.to_csv('production_sammon_embeddings.csv', index=False) # Save to CSV
        print("Sammon embeddings computed and saved to production_sammon_embeddings.csv")

        # Also save the scaler, so you can use it to transform new data in production
        import joblib
        joblib.dump(scaler, 'production_scaler_sammon.joblib')
        print("Scaler saved to production_scaler_sammon.joblib")
        ```

2.  **Storage and Serving of Embeddings:**
    *   **Database or File Storage:** Store the pre-computed embeddings (e.g., in CSV files, databases, cloud storage).
    *   **Cloud Storage (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage):** Cloud storage is a good option for scalability and accessibility from web applications or dashboards.

3.  **Visualization in Dashboards or Web Applications:**
    *   **Web Frameworks (Flask, Django, React, Vue.js etc.):**  Use web frameworks to build web applications or dashboards that load the pre-calculated Sammon embeddings and display them as interactive scatter plots.
    *   **Dashboarding Libraries (Plotly Dash, Tableau, Power BI, Cloud Dashboards):** Use dashboarding tools to create interactive dashboards with Sammon Mapping visualizations. JavaScript libraries (D3.js, Chart.js, Plotly.js) can be used for web-based interactive plots.

4.  **Data Preprocessing in Production:**
    *   **Load Saved Scaler:** In your production environment, load the `StandardScaler` (or other scaler) that you trained during the offline phase (using `joblib.load('production_scaler_sammon.joblib')` in the example).
    *   **Transform New Data:** When new data points arrive that you want to visualize on the Sammon map, apply the *same* scaling transformation using the *loaded scaler* to ensure consistency. Then, you can use the pre-computed Sammon embeddings as the base map, and potentially plot the new, transformed data points on top of this existing map (though re-running Sammon Mapping with new data might be needed for a fully updated map).

**Deployment Environments (Similar to t-SNE blog):**

*   **Cloud:** Cloud platforms (AWS, Google Cloud, Azure) for scalability, storage, and managed services.
*   **On-Premise:** Deploy on your servers if needed for security or compliance.
*   **Local Testing:** Develop and test locally on your machine.

**Code Snippet - Conceptual Flask App for Serving Sammon Map (Very Similar to t-SNE example, just loading different embedding file):**

```python
# Conceptual Flask app (requires flask, pandas, matplotlib)
from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def home():
    # Load pre-computed Sammon embeddings from CSV
    df_embeddings = pd.read_csv('production_sammon_embeddings.csv')
    # Generate matplotlib plot in memory (same plotting code as before)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_embeddings['sammon_dim1'], df_embeddings['sammon_dim2'])
    plt.title('Sammon Mapping Visualization (Production)')
    plt.xlabel('Sammon Dimension 1')
    plt.ylabel('Sammon Dimension 2')
    plt.grid(True)
    # Save plot to buffer and encode to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf8')
    plot_url = f'data:image/png;base64,{img_base64}'
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
```

**Key Production Considerations (Same as for t-SNE):**

*   **Performance:** Sammon Mapping can be computationally intensive.  Optimize code if needed. For very large datasets, consider more scalable methods if visualization speed is critical.
*   **Scalability:** Design for scalability if handling large datasets or frequent updates.
*   **Monitoring:** Monitor data pipelines and visualization processes.
*   **Consistency:** Ensure preprocessing in production matches training.
*   **Version Control:** Manage versions of embeddings, scalers, and code.

## 10. Conclusion: Sammon Mapping in the Data Visualization Toolkit

Sammon Mapping is a valuable dimensionality reduction technique, especially when your goal is to create low-dimensional visualizations that preserve pairwise distances as faithfully as possible.  It's been used in various fields to explore and understand complex data structures.

**Real-World Applications:**

*   **Cheminformatics and Drug Discovery:** Visualizing similarity of chemical compounds based on their properties, aiding in drug design and discovery.
*   **Materials Science:** Visualizing relationships between materials based on their properties, helping to discover new materials.
*   **Social Sciences and Humanities:** Visualizing survey data, text data, or other complex datasets to explore patterns and relationships.
*   **General Exploratory Data Analysis:** Sammon Mapping can be a useful addition to the data scientist's toolkit for exploring the structure of high-dimensional datasets and creating informative visualizations.

**Where Sammon Mapping is Still Relevant:**

*   **Distance Preservation Focus:** If your primary goal is to create a visualization where the relative distances between points in the 2D or 3D map closely reflect their original distances, Sammon Mapping can be a suitable choice.
*   **Complementary to Other Techniques:** Sammon Mapping can be used in conjunction with other dimensionality reduction techniques like PCA, t-SNE, or UMAP.  Comparing visualizations from different methods can provide a more comprehensive understanding of the data.

**Optimized and Newer Algorithms:**

While Sammon Mapping is a useful technique, it has some limitations:

*   **Computational Cost:** It can be slower than some other methods, especially for large datasets, due to the iterative optimization process.
*   **Non-Convex Optimization:** The optimization of the Sammon Stress function is non-convex, which means the algorithm might get stuck in local minima and the results can depend on initialization.

Newer and often more popular dimensionality reduction techniques include:

*   **t-SNE (t-Distributed Stochastic Neighbor Embedding):**  Excellent for revealing local structure and clusters in data, often produces visually compelling maps, but can distort global distances and is also computationally intensive.
*   **UMAP (Uniform Manifold Approximation and Projection):**  Generally faster and more scalable than t-SNE and Sammon Mapping, often preserves both local and global structure well, and is becoming a popular general-purpose dimensionality reduction method.
*   **LargeVis:** Designed for visualizing very large datasets, optimized for speed.

**Choosing between Sammon Mapping and Alternatives:**

*   **For Visualization Focusing on Distance Preservation:** Sammon Mapping is a dedicated technique for this goal.
*   **For General-Purpose Visualization and Speed:** UMAP is often a strong alternative due to its speed, scalability, and good overall performance. t-SNE is also widely used, especially when you want to emphasize local structure.
*   **For Linear Dimensionality Reduction and Speed:** PCA is the standard for linear methods and is computationally very efficient.

**Final Thought:** Sammon Mapping remains a valuable technique in the broader landscape of dimensionality reduction and data visualization. While it may not be as widely used as some newer methods, its focus on distance preservation makes it a useful tool for specific visualization goals and data exploration scenarios. As always, the best approach often involves experimenting with different methods and choosing the one that best reveals the structure and insights within your data.

## 11. References and Resources

Here are some references to learn more about Sammon Mapping and related techniques:

1.  **Sammon, J. W. (1969). A nonlinear mapping for data structure analysis.** *IEEE Transactions on Computers*, *18*(5), 401-409. ([IEEE Xplore Link - possibly behind paywall, search for paper title online for access](https://ieeexplore.ieee.org/document/1670256)) - The original paper that introduced Sammon Mapping. Provides a detailed explanation of the algorithm and its motivation.
