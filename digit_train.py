import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import perf_counter
import numpy as np
from keras.datasets import mnist


def show_stats(cv_results):
    # Getting the Mean_Test_Score
    means = cv_results["mean_test_score"]
    # Zip the mean_test_score and the params and then sort it.
    zipped = zip(means, cv_results["params"])
    sorted_zipped = sorted(zipped, key=lambda t: t[0], reverse=True)
    for mean, params in sorted_zipped:
        print(
            f'metric={params["metric"]} n_neighbors={params["n_neighbors"]} weights={params["weights"]} : Mean accuracy={mean * 100:.2f}%  '
        )


# Read in the MNIST Training DAta
d = 28 * 28
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape((-1, d))
n = X_train.shape[0]

# Make a dictionary with all the parameter values you want to test
## Your code here

parameters = {
    "metric": ["euclidean", "manhattan"],
    "n_neighbors": [1, 3, 5, 7],
    "weights": ["uniform", "distance"],
}


# Create a KNN classifier
## Your code here
clf = KNeighborsClassifier()

# Create a GridSearchCV to determine the best values for the parameters
## Your code here
grid_searcher = GridSearchCV(clf, parameters, cv=4, verbose=3)

# Run it
start = perf_counter()
## Your code here
grid_searcher.fit(X_train, y_train)
print(f"Took {perf_counter() - start:.1f} seconds")

# List out the combinations and their scores
show_stats(grid_searcher.cv_results_)

# Get the best values
best_combo_index = grid_searcher.best_index_  ## Your code here
best_params = grid_searcher.best_params_  ## Your code here

print(f"Best parameters: {best_params}")


# Create a new classifier with the best hyperparameters
## Your code here
Params = {
    "algorithm": "auto",
    "leaf_size": 30,
    "metric": "euclidean",
    "metric_params": None,
    "n_jobs": None,
    "n_neighbors": 3,
    "p": 2,
    "weights": "distance",
}
classifier = KNeighborsClassifier(
    algorithm="auto",
    leaf_size=30,
    metric=best_params["metric"],
    metric_params=None,
    n_jobs=None,
    n_neighbors=best_params["n_neighbors"],
    p=2,
    weights=best_params["weights"],
)  ## Your code here


# Fit to training data
## Your code here
classifier.fit(X_train, y_train)

# Do predictions for the training data, and print the accuracy
## Your code here
predictions = classifier.predict(X_train)
accuracy = accuracy_score(y_train, predictions)
print(f"Params: {Params}")
print(f"Accuracy on training data: {(accuracy * 100):.1f}%")

# Write the classifier out to a pickle file
with open("knn_model.pkl", "wb") as f:
    pkl.dump(classifier, f)
print("Wrote knn_model.pkl")
