import numpy as np

class PositionalEncoding:
    def __init__(self, sentence, embedding_dim):
        self.sentence = sentence
        self.sentence_length = len(sentence)  # Używamy długości listy jako liczby
        self.embedding_dim = embedding_dim
        self.positional_encoding = self._compute_positional_encoding()

    def _compute_positional_encoding(self):
        # Macierz o rozmiarze (sentence_length, embedding_dim)
        positional_encoding = np.zeros((self.sentence_length, self.embedding_dim))
        
        # Wypełniamy macierz wartościami opartymi na funkcjach trygonometrycznych
        # Obliczamy poszczególne wartości dla sinusoidy
        for pos in range(self.sentence_length):
            for i in range(self.embedding_dim):
                if i % 2 == 0:
                    positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.embedding_dim)))
                else:
                    positional_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / self.embedding_dim)))
        
        return positional_encoding

    def get_positional_encoding(self):
        return self.positional_encoding