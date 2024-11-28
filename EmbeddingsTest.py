# -*- coding: utf-8 -*-

from Embeddings import Embeddings

if __name__ == "__main__":
    embeddings = Embeddings()
    word = "dog"
    vector = embeddings.get_embedding(word)
    print("Vector for 'dog':", vector)  # Wyświetli: [0.45, 0.01, -0.56, 0.55]
