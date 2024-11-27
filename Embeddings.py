import json

class Embeddings:
    def __init__(self, json_file):
        self.embeddings = self.__load_json(json_file)

    def __load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def get_embedding(self, word):
        return self.embeddings.get("embeddings", {}).get(word, None)
    
    def get_sentence(self):
      return self.embeddings.get("sentence", "")
