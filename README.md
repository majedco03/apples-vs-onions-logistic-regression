# Apples vs Onions – Logistic Regression

This project explores **binary image classification** using **Logistic Regression** as a deliberate baseline model, with the goal of building strong intuition about how linear models behave on visual data.

Rather than focusing on high accuracy, the project emphasizes **understanding model limitations** and representation choices before moving to more complex deep learning architectures.

---

## Project Motivation
In many image classification problems, deep learning models are applied immediately without first understanding simpler approaches.

This project intentionally restricts the model choice to Logistic Regression in order to:
- Study how images are represented as numerical feature vectors
- Observe the impact of flattening images on spatial information
- Analyze why linear decision boundaries are insufficient for most visual tasks
- Build intuition that justifies the use of neural networks and CNNs later on

---

## Problem Description
The task is to classify images into two categories:
- **Apples**
- **Onions**

This binary setup keeps the problem focused and allows clear analysis of what the model can and cannot learn.

---

## Planned Approach
The project will follow a simple and transparent pipeline:
1. Image preprocessing and normalization
2. Conversion of images into 1D feature vectors
3. Training a Logistic Regression model using gradient descent
4. Evaluating performance and generalization behavior
5. Analyzing failure cases and model limitations

---

## Expected Insights
This project is expected to demonstrate that:
- Model expressiveness is more important than hyperparameter tuning
- Linear models struggle with complex visual patterns
- Improving performance requires better representations, not just more training

---

## Scope and Constraints
To keep the learning objective clear, the project intentionally avoids:
- Deep Neural Networks or CNNs
- Advanced feature engineering
- Heavy data augmentation

---

## Future Work
After completing this baseline model, the project can be extended by:
- Replacing Logistic Regression with a shallow Neural Network
- Preserving spatial information using Convolutional Neural Networks (CNNs)
- Comparing results against this baseline

---

## Key Goal
The primary goal of this project is not performance, but **understanding** — specifically, understanding *why* Logistic Regression fails on image data and **when** more complex models become necessary.