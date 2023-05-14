from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

file = np.load("time2vec_udsjpy.npz")
files, vectors, labels = file["arr_0"], file["arr_1"], file["arr_2"]
clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=1.0, max_depth=1, random_state=0, verbose=True)
clf.fit(vectors, labels)
clf.score(vectors[-1000:], labels[-1000:])

with open('gb.pkl','wb') as f:
    pickle.dump(clf,f)