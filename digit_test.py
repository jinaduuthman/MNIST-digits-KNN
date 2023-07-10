import pickle as pkl
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from time import perf_counter
from keras.datasets import mnist

# Read in the MNIST data
d = 28 * 28
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape((-1, d))

# Read in the classifier
with open("knn_model.pkl", "rb") as f:
    classifier = pkl.load(f)

start = perf_counter()
# Make predictions for the test data
predictions = classifier.predict(X_test)
## Your code here
print(f"Inference: elapsed time = {perf_counter() - start:.2f} seconds")

# Show accuracy
## Your code here
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on training data: {(accuracy * 100):.1f}%")


# Create a confusion matrix
## Your code here
cm = confusion_matrix(y_test, predictions)
print(f"Confusion: \n {cm}")

# Make a display
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots()
ax.set_title("MNIST Confusion Matrix")
cmp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="2g")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("sk_confusion.png")
print("Wrote sk_confusion.png")
