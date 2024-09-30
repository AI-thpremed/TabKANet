from typing import Dict, List
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from tabkanet.models.base_blocks import Activation, ColumnEmbedding

class CatEncoder(nn.Module):
    def __init__(self, 
                 vocabulary: Dict[str, Dict[str, int]],
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, 
                 dropout_rate: float):
        """
        Categorical feature encoder.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features
        - embedding_dim (int): Embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - dropout_rate (float): Dropout rate.
        """
        super(CatEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.model = nn.ModuleDict({
            'column_embedding_layer': ColumnEmbedding(vocabulary, embedding_dim),
            'transformer_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=nhead,
                    dim_feedforward=dim_feedforward, 
                    dropout=dropout_rate,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True),
                    num_layers=num_layers,
                    norm=nn.LayerNorm([embedding_dim]))})

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = [self.model['column_embedding_layer'](x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = self.model['transformer_encoder'](x).view(batch_size, -1)
        # x = self.model['transformer_encoder'](x)
        return x

class NumEncoder(nn.Module):
    def __init__(self, num_features: int):
        """
        Continuous feature encoder.

        Parameters:
        - num_features (int): Number of continuous features.
        """
        super(NumEncoder, self).__init__()
        self.norm = nn.LayerNorm([num_features])
        
    def forward(self, x: torch.Tensor):
        return self.norm(x)





class NumEncoderTransformer(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, dropout_rate: float):

        super(NumEncoderTransformer, self).__init__()
        self.norm = nn.LayerNorm([num_continuous_features])
        self.kanclassifier=KAN([num_continuous_features,64,embedding_dim * num_continuous_features])
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        x=self.norm(x)
        x = self.kanclassifier(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str, dropout_rate: float):
        """
        MLP block.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm([output_dim]),
            Activation(activation),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int], activation: str, 
                 dropout_rate: float):
        """
        MLP model.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - hidden_dims (List[int]): List of hidden layer dimensions.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims
        self.model = nn.Sequential(*(
            [
                MLPBlock(
                    dims[i], dims[i + 1], 
                    activation, dropout_rate) 
                for i in range(len(dims) - 1)] + \
                [nn.Linear(dims[-1], output_dim)]))
        
    def forward(self, x):
        return self.model(x)

class TabularTransformer(nn.Module):
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float,
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float):
        """
        Tabular transformer model.

        Parameters:
        - output_dim (int): Output dimension.
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features.
        - num_continuous_features (int): Number of continuous features.
        - embedding_dim (int): Embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - attn_dropout_rate (float): Dropout rate.
        - mlp_hidden_dims (List[int]): List of hidden layer dimensions.
        - activation (str): Activation function.
        - ffn_dropout_rate (float): Dropout rate.
        """
        super(TabularTransformer, self).__init__()
        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate),
            'continuous_feature_encoder': NumEncoder(num_continuous_features),
        })
        self.classifier = MLP(embedding_dim * len(vocabulary) + num_continuous_features, output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)


    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        categorical_x = self.encoders['categorical_feature_encoder'](categorical_x)
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        x = torch.cat([categorical_x, continuous_x], dim=-1)
        
        x = self.classifier(x)
        return x
    