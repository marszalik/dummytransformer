# -*- coding: utf-8 -*-

from Embeddings import Embeddings

if __name__ == "__main__":
    embeddings = Embeddings("embeddings.json")
    print("Embedding for 'Ala':", embeddings.get_embedding("Ala"))  # Wyświetli: [0.12, 0.85, -0.34]
