from sklearn.feature_extraction.text import CountVectorizer
import re
import string
from scipy import sparse
import numpy as np

documents = []

with open("WOS46985/X.txt", "r") as f: # make sure the web of science file is in the same directory, or change the path
    for line in f:
        documents.append(line.strip())

print(documents[0])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

X.eliminate_zeros()

# convert X to CSC format
X = X.tocsc()

print(X.shape)
print(X.getformat())

#print the number of non zeros
print(X.nnz)

def save_sparse_csc(filename, array):
    np.savetxt(filename + "_data.txt", array.data, delimiter=",", fmt="%d")
    np.savetxt(filename + "_indices.txt", array.indices, delimiter=",", fmt="%d")
    np.savetxt(filename + "_indptr.txt", array.indptr, delimiter=",", fmt="%d")

save_sparse_csc("X", X)

# After this, convert the CSC file(s) created by this to COO with CSC_to_COO.py