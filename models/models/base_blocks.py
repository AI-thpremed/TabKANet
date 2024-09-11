from typing import Dict

import torch 
import torch.nn as nn

class Activation(nn.Module):
    def __init__(self, activation: str) -> None:
        """
        Activation function wrapper via string input.

        Parameters:
        - activation (str): Activation function to use. 
        """
        super(Activation, self).__init__()
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() in {'gelu', 'elu', 'selu', 'celu'}:
            self.activation = getattr(nn, activation.upper())()
        elif activation.lower() in {'tanh', 'sigmoid', 'hardshrink', 'hardsigmoid', 'tanhshrink', 'softshrink', 'softsign', 'softplus'}:
            self.activation = getattr(nn, activation.capitalize())()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f'Invalid activation function: {activation}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)
    
class ColumnEmbedding(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        """
        Column embedding layer for categorical features.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features 
                      (e.g. {
                                'column_name_1': 
                                {
                                    'category_1_1': index_1, 
                                    'category_1_2': index_2,
                                }, 
                                'column_name_2': 
                                {
                                    'category_2_1': index_1, 
                                    'category_2_2': index_2,
                                    'category_2_3': index_3,
                                }
                            }).
        - embedding_dim (int): Embedding dimension.
        """
        super(ColumnEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict({
            column: nn.Embedding(len(vocab), embedding_dim)
            for column, vocab in vocabulary.items()
        })
        
    def forward(self, x: torch.Tensor, column: str) -> torch.Tensor:
        return self.embeddings[column](x)