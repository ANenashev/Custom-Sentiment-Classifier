# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:07:21 2017

@author: ANTON
"""

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import cPickle as pickle


data_train = pd.read_csv("Custom sentiment train.csv")

labels = data_train.label.values
texts = data_train.text

SGD_pipeline = Pipeline([("vect", CountVectorizer(ngram_range=(1, 2), stop_words = 'english', max_features = 10000)),
                        ("tfidf", TfidfTransformer(use_idf=True)),
                         ("clf", SGDClassifier(random_state=10))])

SGD_clf = SGD_pipeline.fit(texts, labels)

output = open('SGDClassifierTfidfVect.pkl', 'wb')
pickle.dump(SGD_clf, output, 2)
output.close()
