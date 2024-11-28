import json
import numpy as np


class Embeddings:
    def __init__(self):
        self.embeddings = self.__load_json("embeddings.json")

    def __load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def get_embedding(self, word):
        return np.array(self.embeddings[word])
    
