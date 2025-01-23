---
title: Decision Tree
excerpt: "Decision Tree Algorithm"
# permalink: /courses/classification/decision-tree/
last_modified_at: 2025-01-22T23:45:00-00:00
hidden: false
categories:
  - Machine Learning
tags: 
  - Machine Learning
  - Classification Model
  - Tree Model
  - Supervised Learning
---

## Introduction
Decision Trees (DTs) are a non-parametric **supervised learning** method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
The internal nodes represent the features of a dataset and branches represents the decision rules. Each leaf node represents the outcome.

{% capture fig_img %}
![Decision Tree]({{ '/assets/images/courses/Decision_tree_for_playing_outside.webp' | relative_url }}){:width="50%"}
{% endcapture %}

<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Decision to play outside</figcaption>
</figure>

