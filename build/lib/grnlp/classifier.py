

import fasttext as fs
from pathlib import Path
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from typing import List, Any, Tuple
from keras.models import model_from_json
import pickle
import os

TOKENIZER = WordPunctTokenizer()

class TextClassifier:

    __instance = None

    @staticmethod
    def getInstance():
        if TextClassifier.__instance == None:
            TextClassifier()
        return TextClassifier.__instance

    def __init__(self, model_path: {}):

        if TextClassifier.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TextClassifier.__instance = self

        self.__w_v_s      = 100
        self.__sen_s      = 20
        self.__fil        = 128
        self.__wv         = {}
        self.__model_path = model_path

        self.__load_model()

    def __load_fs_model(self):

        if "fasttext" in self.__model_path:
            self.__fs_model = fs.load_model(self.__model_path["fasttext"])

    def __load_ml_model(self):

        if "ml_model_json" in self.__model_path and "ml_model_h5" in self.__model_path:
            json_file = open(self.__model_path["ml_model_json"], 'r')
            model_json = json_file.read()
            json_file.close()
            self.__ml_model = model_from_json(model_json)
            self.__ml_model.load_weights(self.__model_path["ml_model_h5"])
            print(self.__ml_model.summary())

    def __load_ml_classes(self):

        if "classes" in self.__model_path:
            self.__CLASSES = pickle.load(open(self.__model_path["classes"], 'rb'))
            print(self.__CLASSES)


    def __fs_wv(self, word: str, size: int)->List[float]:
        if word in self.__wv:
            return self.__wv[word]
        else:
            try:
                self.__wv[word] = self.__fs_model[word]
            except:
                self.__wv[word] = np.zeros(size)

            return self.__wv[word]


    def __generate_features(self, sentences: List[str])->Any:
        features = []
        for sent in sentences:
            wvectors = []
            new_sentence = sent
            tokens = TOKENIZER.tokenize(new_sentence)
            if len(tokens) > self.__sen_s:
                tokens = tokens[:self.__sen_s]

            for tok in tokens:
                wvectors.append(self.__fs_wv(tok, self.__w_v_s))

            if len(tokens) < self.__sen_s:
                pads = [np.zeros(self.__w_v_s)
                        for _ in range(self.__sen_s - len(tokens))]

                wvectors = pads + wvectors

            wvectors = np.asarray(wvectors)

            features.append(wvectors)
        return np.array(features)


    def predict(self, sentence: str)-> Tuple[str, float]:
        if sentence == "":
            raise "Empty Text"

        test_sentence = [sentence]
        test_features = self.__generate_features(test_sentence)
        predictions = self.__ml_model.predict(test_features)
        if max(predictions[0]) >= 0.50:
            return (self.__CLASSES[np.argmax(predictions)], predictions[0][np.argmax(predictions)])
        else:
            return ("unknown", 0.00)

    def __load_model(self):
        self.__load_fs_model()
        self.__load_ml_model()
        self.__load_ml_classes()

