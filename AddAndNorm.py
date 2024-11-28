import numpy as np


class AddAndNorm:

    def __init__(self):
       pass
        

    def layer_norm(self, x):
        if(x.ndim == 2):
            self.d_model = x[0].shape[0] #wymiarowość embedingu lub sumy headow dla wielu
        else:
            self.d_model = x.shape[0] #wymiarowość embedingu lub sumy headow dla jednego
        self.gamma = np.ones(self.d_model)  # Skalar dla LayerNorm
        self.beta = np.zeros(self.d_model)  # Przesunięcie dla LayerNorm
        """
        Warstwa normalizacji (LayerNorm).

        Args:
            x (np.ndarray): Wejściowy wektor lub macierz.

        Returns:
            np.ndarray: Znormalizowany wektor lub macierz.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + 1e-6)
        return self.gamma * normalized + self.beta

    def add_and_norm(self, input_tensor, residual_tensor):
        """
        Operacja Add & Norm.

        Args:
            input_tensor (np.ndarray): Wejściowy tensor z warstwy uwagi (attention output).
            residual_tensor (np.ndarray): Tensor wejściowy oryginalnego słowa (residual connection).

        Returns:
            np.ndarray: Wynik Add & Norm.
        """
        # Dodawanie residuala
        added = input_tensor + residual_tensor
        # Normalizacja
        normalized = self.layer_norm(added)
        return normalized

