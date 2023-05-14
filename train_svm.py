from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

file = np.load("time2vec_udsjpy.npz")
files, vectors, labels = file["arr_0"], file["arr_1"], file["arr_2"]
clf = svm.SVC(verbose=True)
clf.fit(vectors, labels)

with open('svm.pkl','wb') as f:
    pickle.dump(clf,f)