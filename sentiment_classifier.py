# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:23:41 2017

@author: ANTON
"""

from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./SGDClassifierTfidfVect.pkl")
        #self.vectorizer = joblib.load("./BigramUnprocessedVectorizer.pkl")
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            #vectorized = self.vectorizer.transform([text])
            return self.model.predict([text])[0]#,\
                   #self.model.predict_proba([text])[0].max()
                   
        except:
            print "prediction error"
            return -1#, 0.8

    def predict_list(self, list_of_texts):
        try:
            #vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(list_of_texts)#,\
                   #self.model.predict_proba(list_of_texts)
        except:
            print 'prediction error'
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction
        #prediction_probability = prediction[1]
        #return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]
        return self.classes_dict[class_prediction]
