# Hand Gesture Recognition using SVM

## Project Overview

This project implements a Support Vector Machine (SVM) based hand gesture recognition system. The model learns from hand gesture images and classifies them into different gesture categories.

## Objective

To develop a machine learning model that can identify and classify different hand gestures from image data, supporting intuitive human-computer interaction and gesture-based control systems.

## Dataset

The project uses the **LeapGestRecog** hand gesture image dataset.

The dataset contains images belonging to multiple hand gesture categories.

The dataset is stored locally and is not included in this GitHub repository because of its large size.

## Methodology

1. Load hand gesture images
2. Resize the images
3. Convert images to grayscale
4. Flatten images into numerical features
5. Normalize pixel values
6. Split the dataset into training and testing sets
7. Train the SVM classifier
8. Predict the test images
9. Evaluate model performance
10. Display the confusion matrix

## Machine Learning Algorithm

**Support Vector Machine (SVM)**

SVM is a supervised machine learning algorithm used for classification. It finds a decision boundary that separates different classes of data.

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

## Model Evaluation

The model is evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix

## Project Structure

```text
SCT_ML_4/
├── hand_recognization.py
├── .gitignore
└── README.md
```

The `leapGestRecog` dataset is kept locally and excluded from GitHub using `.gitignore`.

## How to Run

Install the required libraries:

```text
pip install opencv-python scikit-learn numpy matplotlib
```

Place the dataset locally:

```text
leapGestRecog/
```

Then run:

```text
python hand_recognization.py
```

The program loads the gesture images, trains the SVM model, evaluates its performance, and displays the classification results.

## Internship Task

**SkillCraft Technology — Machine Learning Internship**

Task 4: Develop a hand gesture recognition model that can identify and classify different hand gestures from image data.
