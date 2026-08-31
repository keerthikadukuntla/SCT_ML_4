import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------
# 1. Dataset path
# ---------------------------------------

DATASET_PATH = "LeapGestRecog"

IMAGE_SIZE = 64

# Number of images used from each gesture
MAX_IMAGES_PER_GESTURE = 500


# ---------------------------------------
# 2. Load images
# ---------------------------------------

images = []
labels = []
gesture_names = []

print("Loading hand gesture images...")

# Find gesture folders
for person_folder in sorted(os.listdir(DATASET_PATH)):

    person_path = os.path.join(DATASET_PATH, person_folder)

    if not os.path.isdir(person_path):
        continue

    for gesture_folder in sorted(os.listdir(person_path)):

        gesture_path = os.path.join(person_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        # Create label for each gesture
        if gesture_folder not in gesture_names:
            gesture_names.append(gesture_folder)

        label = gesture_names.index(gesture_folder)

        count = 0

        for filename in os.listdir(gesture_path):

            if count >= MAX_IMAGES_PER_GESTURE:
                break

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                gesture_path,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convert to grayscale
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # Flatten image
            image = image.flatten()

            images.append(image)
            labels.append(label)

            count += 1


# ---------------------------------------
# 3. Convert to NumPy arrays
# ---------------------------------------

X = np.array(images)
y = np.array(labels)

print("\nImages loaded:", len(X))
print("Number of features:", X.shape[1])
print("Number of gestures:", len(gesture_names))

print("\nGesture classes:")

for i, name in enumerate(gesture_names):
    print(i, ":", name)


# ---------------------------------------
# 4. Normalize pixel values
# ---------------------------------------

X = X / 255.0


# ---------------------------------------
# 5. Split dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ---------------------------------------
# 6. Create SVM model
# ---------------------------------------

print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


# ---------------------------------------
# 7. Train model
# ---------------------------------------

model.fit(X_train, y_train)

print("Training completed!")


# ---------------------------------------
# 8. Prediction
# ---------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------
# 9. Accuracy
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("HAND GESTURE RECOGNITION")
print("==============================")

print(
    "Accuracy:",
    accuracy
)


# ---------------------------------------
# 10. Classification report
# ---------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=gesture_names
    )
)


# ---------------------------------------
# 11. Confusion matrix
# ---------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ---------------------------------------
# 12. Display confusion matrix
# ---------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title(
    "Hand Gesture Recognition - Confusion Matrix"
)

plt.xlabel("Predicted Gesture")
plt.ylabel("Actual Gesture")

plt.xticks(
    range(len(gesture_names)),
    gesture_names,
    rotation=90
)

plt.yticks(
    range(len(gesture_names)),
    gesture_names
)

plt.colorbar()

plt.tight_layout()

plt.show()


# ---------------------------------------
# 13. Sample predictions
# ---------------------------------------

print("\nSample Predictions:")

for i in range(min(10, len(X_test))):

    actual = gesture_names[y_test[i]]
    predicted = gesture_names[y_pred[i]]

    print(
        f"Image {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )


print("\nTask completed successfully!")