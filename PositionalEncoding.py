import numpy as np

class PositionalEncoding:

    def add_position(self, vectors):
        
        sentence_length = len(vectors)
        positional_encoding = []
        # Obliczamy poszczególne wartości dla sinusoidy
        for i in range(len(vectors)):
            positional_encoding.append([])
            for j in range(len(vectors[0])):
                if i % 2 == 0:
                    positional_encoding[i].append(float(np.sin(vectors[i][j] / (10000 ** (i / sentence_length)))))
                else:
                    positional_encoding[i].append(float(np.cos(vectors[i][j] / (10000 ** ((i - 1) / sentence_length)))))
        
        return positional_encoding
 