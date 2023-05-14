import xgboost as xgb
import numpy as np
import pickle

file = np.load("time2vec_udsjpy.npz")
files, vectors, labels = file["arr_0"], file["arr_1"], file["arr_2"]
clf  = xgb.XGBClassifier(n_estimators=3000)
clf.fit(vectors, labels)
clf.save_model('xgb.model')
