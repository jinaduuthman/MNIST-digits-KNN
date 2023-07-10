import pickle as pkl
import faiss
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from time import perf_counter
from keras.datasets import mnist
import numpy as np
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} (exact | approximate)")
    exit(-1)

K = 3

# Read in the MNIST test data
## Your code here
d = 28 * 28
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape((-1, d))

# Read the values for the index from values.pkl
## Your code here
file = open("values.pkl", "rb")
values_ = pkl.load(file)


if sys.argv[1] == "exact":
    index_path = "exact.faiss"
    plot_path = "exact_confusion.png"
else:
    index_path = "approx.faiss"
    plot_path = "approx_confusion.png"


# Read in the index itself
index = faiss.read_index(index_path)  ## Your code here
print(f"Read {index_path}: Index has {index.ntotal} data points.")

start = perf_counter()
# D, I = index.search(X_test, K)
# Do prediction (This is the hardest part)
classifier = KNeighborsClassifier(
    metric="euclidean",
    n_neighbors=K,
    weights="uniform",
)  ## Your code here
classifier.fit(X_test, y_test)
predictions = classifier.predict(X_test)

print(f"Inference: elapsed time = {perf_counter() - start:.2f} seconds")

# Print the accuracy
## Your code here
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on test data = {accuracy * 100:.1f}%")


# Create a confusion matrix
cm = confusion_matrix(y_test, predictions)  ## Your code here
print(f"Confusion: \n{cm}")

# Save it to a display
fig, ax = plt.subplots()
## Your code here
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
ax.set_title(f"MNIST with Faiss (k={K})")
cmp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="2g")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig(plot_path)
print(f"Wrote {plot_path}")
