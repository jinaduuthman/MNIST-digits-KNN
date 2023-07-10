import pickle as pkl
import faiss
import numpy as np
from keras.datasets import mnist
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} (exact | approximate)")
    exit(-1)

# Read in the MNIST training data
## Your code here
d = 28 * 28
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape((-1, d))
n = X_train.shape[0]


if sys.argv[1] == "exact":
    index = faiss.IndexFlatL2(d) ## Your code here
    print("Using IndexFlatL2 for true KNN")
    path_prefix = "exact"
else:
    quantizer = faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFPQ(quantizer, d, 1024, 16, 8) ## Your code here
    print("Using HNSW/IVFPQ for approximate KNN")
    path_prefix = "approx"

# Train and add with the training data
## Your code here
X_train = X_train.astype("float32")
## Your code here
index.train(X_train)
index.add(X_train)


print(f"Index has {index.ntotal} data points.")

path = f"{path_prefix}.faiss"
# Write out the index and the classes it indexes
faiss.write_index(index, path)
## Your code here
with open("values.pkl", "wb") as f:
    pkl.dump(y_train, f)
print(f"Wrote index to values.pkl and {path}")
