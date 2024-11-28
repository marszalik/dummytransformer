# -*- coding: utf-8 -*-
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding

if __name__ == "__main__":
    # Definiowanie zdania w main.py
    sentence = "Ala ma kota"  # Przykładowe zdanie
    embedding_dim = 3  # Wymiar embeddingu (powinien pasować do wymiarów w embeddings.json)
    
    # Ładowanie embeddingów z pliku JSON
    embeddings = Embeddings("embeddings.json")
    
    # Tworzenie obiektu PositionalEncoding dla zdania
    pos_enc = PositionalEncoding(sentence.split(), embedding_dim)
    
    # Wyświetlenie wynikowego Positional Encoding
    print("Positional Encoding:")
    print(pos_enc.get_positional_encoding())
    
    # Przykładowe wyświetlenie embeddingu dla słowa 'Ala'
    print("\nEmbedding for 'Ala':")
    print(embeddings.get_embedding("Ala"))
