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

## Key Goal
The primary goal of this project is not performance, but **understanding** — specifically, understanding *why* Logistic Regression fails on image data and *when* more complex models become necessary.

## How to run
follow these steps to set up the project and classify your own images

1. Prerequisites
Ensure you have Python installed. You will need to install the following libraries:
NumPy: For vectorized mathematical operations.
Pillow (PIL): For image loading and preprocessing (resizing and RGB conversion).

You can install them via pip: 
pip install numpy pillow

2. Data Setup
/train data
    /apple  <-- Place apple training images here
    /onion  <-- Place onion training images here
/test data
    /apple  <-- Place apple test images here
    /onion  <-- Place onion test images here

Note: The code expects images to be in .jpg, .jpeg, or .png format.
Preprocessing: The code will automatically resize your images to 64 X 64 pixels and normalize them.

3. Execution
Run the main script to open the interactive menu:
python Main.py

4. Usage Flow
  1. Select Option 1: This loads the data from your folders and trains the Logistic Regression model using gradient descent.

  2. Select Option 2: View the accuracy of the model on both your training and test sets.

  3. Select Option 3: Provide a path to a specific image file to see if the model classifies it as an Apple or an Onion.

  4. Select Option 4: Exit the program



Note: The learning rate is set to 0.5 and the model runs for 2000 iterations by defoult, you can change them if you want in Data.py

# Note
this Project is implemented after finishing week 2 of DeepLearning.AI course of Deep learning specialization