# -*- coding: utf-8 -*-
from pprint import pprint

from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding

#  Tablica wektorów
vectors_array =[[ 0.12,  0.85, -0.34,  0.45],[ 0.45,  0.01, -0.56,  0.55],[-0.78,  0.23,  0.67,  0.69]]
    
# Obliczenie positional encoding
positionalEncoding = PositionalEncoding()
vectors_with_positions = positionalEncoding.add_position(vectors_array)
    
# Wyświetlenie wynikowego Positional Encoding
pprint(vectors_with_positions)
    
# Przykładowe wyświetlenie embeddingu dla słowa 'what dog drinks'
#[[0.11971220728891936, 0.7512804051402927, -0.3334870921408144, 0.43496553411123023],
# [0.9004471023526769, 0.9999500004166653, 0.8472551110134161, 0.8525245220595057],
# [-0.0016804582673049814, 0.0004955199584290004, 0.0014434707410497481, 0.0014865593886070607]]