---
title: "Decision Stump"
excerpt: "Decision Stump Algorithm"
# permalink: /courses/classification/decision-stump/
last_modified_at: 2025-01-22T23:45:00-00:00
classes: narrow
hidden: false
strip_title: true
categories:
  - Machine Learning
tags: 
  - Machine Learning
  - Classification Model
  - Tree Model
  - Supervised Learning
---


---
title: "Decision Stump 101: A Simple Yet Powerful Tool for Anomaly Detection"
date: 2025-01-25
author: ML Explorer
categories: [Machine Learning, Anomaly Detection]
tags: [decision-stump, jekyll, tutorial]
---

![Decision Stump Visualization](https://via.placeholder.com/800x400?text=Single+Threshold+Decision+Split)  
*Fig 1. How a decision stump makes binary splits :cite[2]:cite[5]*

## 1. Introduction: The "One-Question Detective"
A **decision stump** is like a detective that asks just **one yes/no question** to solve cases. Imagine a security system that checks:
- "Is the transaction amount > $10,000?" (fraud detection)
- "Is body temperature > 100.4°F?" (health monitoring) :cite[2]

While simple, these "one-question classifiers" form the building blocks of powerful ensemble methods like AdaBoost. They're ideal for:
- Real-time anomaly alerts
- IoT sensor monitoring
- Medical triage systems

## 2. The Math Made Simple
### Core Equation: The Binary Split
For feature \( x \) and threshold \( t \):
\[
\text{Prediction} = 
\begin{cases} 
\text{Anomaly (-1)} & \text{if } x \leq t \\
\text{Normal (1)} & \text{otherwise}
\end{cases}
\]

**Real Example**:  
*Credit Card Fraud Detection*  
Threshold \( t = \$500 \):
- Transaction 1: \$300 → **Normal**
- Transaction 2: \$2,000 → **Flagged** :cite[5]

![Math Breakdown](https://via.placeholder.com/600x200?text=Threshold+Comparison+Diagram)  
*How thresholds create decision boundaries :cite[8]*

## 3. Preprocessing Essentials
| Requirement          | Why Matters?                          | Tools                 |
|----------------------|---------------------------------------|-----------------------|
| Binary Labels         | Works with yes/no outcomes            | LabelEncoder         |
| Feature Selection     | Finds most impactful feature          | Correlation Matrix   |
| Missing Values        | Prevents skewed thresholds            | SimpleImputer        |
| Python Setup          | Essential libraries                   | numpy, scikit-learn  |

```python
# Installation
!pip install numpy scikit-learn pandas