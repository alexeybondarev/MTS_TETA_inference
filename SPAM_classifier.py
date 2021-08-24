import joblib
import numpy as np
import re

class SpamClassifier(object):
    def __init__(self):
        self.transformation = joblib.load('text_transformation.pkl')
        self.model = joblib.load('text_classification_model.pkl')
        self.target_names = ['HAM' , "SPAM"]
        self.stop = joblib.load('stop_words.pkl')
    
    def word_preprocessing(self, text_to_clasify):
        output = np.zeros(4, dtype=np.int)
        output[0] = len([w for w in text_to_clasify.split() if w in self.stop])
        output[1] = len(re.findall('[^\w\s]', text_to_clasify))
        output[2] = len(re.findall(r'\w+', text_to_clasify))
        output[3] = len(text_to_clasify)

        return [output]
    
    def predict_text(self, text):
        try:
            vectorized_text = self.transformation.transform([re.sub('[^\w\s]', '', text)]).toarray()
            input_cls = np.concatenate((self.word_preprocessing(text), vectorized_text), axis=1)
            return self.model.predict(input_cls)[0]
        except:
            print('Failed to predict')
            return None
