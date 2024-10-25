# Dashiell Brooks
# ITP 259 Fall 2024
# HW5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Read the dataset into a dataframe.
df = pd.read_csv("A_Z Handwritten Data.csv")
# print(df)

# label is the target variable.
# Separate the dataframe into feature set and target variable.
X = df.drop(columns=["label"])
y = df["label"]

# Print the shape of the feature set and target variable.
# print(X.shape)
# print(y.shape)

# The target variable values are numbers.
# A data dictionary could be used to map the numbers to the letters of the alphabet.
# Mapping numbers to letters.
word_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
}
df['label'] = df['label'].map(word_dict)

# Show a histogram of the letters.
# plt.figure(1)
# sb.countplot(x="label", data=df)
# plt.show()

# Sample 64 random rows.
# sampled_df = df.sample(n=64)

# Separate the images and labels.
# images = sampled_df.drop(columns=["label"]).values.reshape(-1, 28, 28)
# labels = sampled_df["label"].values

# Create an 8 x 8 grid of subplots.
# fig, axs = plt.subplots(8, 8, figsize=(10, 10))

# Flatten the axes array.
# axs = axs.flatten()

# Loop over the images, labels, and axes.
# for img, label, ax in zip(images, labels, axs):
#     ax.imshow(img, cmap="gray")
#     ax.set_title(label)
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# Partition the data into train and test sets.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.30, random_state=2023, stratify=y)

# Scale the data.
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Create an MLPClassifier.
# Runs very slow with (50,50,50) layers. Using simpler model for loading purposes.
model = MLPClassifier(hidden_layer_sizes=(5,5), activation="relu",
                    max_iter=1000, alpha=1e-3, solver="adam",
                    random_state=2023, learning_rate_init=0.01, verbose=False)

model.fit(X_train_scaled, y_train)

# Plot the loss curve.
# plt.plot(model.loss_curve_)
# plt.show()

# Print test and training accuracy. Using (5,5) model for loading purposes.
# print("The training accuracy is", model.score(X_train,y_train))
# print("The test accuracy is", model.score(X_test,y_test))

# Plot the confusion matrix with letters.
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

