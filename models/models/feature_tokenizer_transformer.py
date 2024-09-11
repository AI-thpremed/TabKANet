from typing import Dict, List

import torch
import torch.nn as nn

from models.models.base_blocks import Activation, ColumnEmbedding

class CatEncoder(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        super(CatEncoder, self).__init__()
        """
        Categorical feature encoder.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features
        - embedding_dim (int): Embedding dimension.
        """
        self.vocabulary = vocabulary
        self.column_embedding = ColumnEmbedding(vocabulary, embedding_dim)
    
    def forward(self, x):
        x = [self.column_embedding(x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        return x

class NumEncoder(nn.Module):
    def __init__(self, num_features: int, embedding_dim:int):
        """
        Continuous feature encoder.

        Parameters:
        - num_features (int): Number of continuous features.
        - embedding_dim (int): Embedding dimension.
        """
        super(NumEncoder, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(num_features)])
        
    def forward(self, x):
        x = [linear(x[:, i].unsqueeze(1)) for i, linear in enumerate(self.linears)]
        x = torch.stack(x, dim=1)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout_rate: float):
        """
        Transformer encoder.

        Parameters:
        - d_model (int): Dimension of the model.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - dropout_rate (float): Dropout rate.
        """
        super(Transformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward, 
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ), 
            num_layers=num_layers,
            norm=nn.LayerNorm([d_model])
        )

    def forward(self, x):
        return self.transformer(x)

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str, dropout_rate: float) -> None:
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
            nn.LayerNorm(output_dim),
            Activation(activation),
            nn.Dropout(dropout_rate))
    
    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim: int, 
                 hidden_dims: List[int], activation: str, dropout_rate: float) -> None:
        """
        MLP model.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - hidden_dims (List[int]): List of hidden dimensions.
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
                for i in range(len(dims) - 1)] \
                + [nn.Linear(dims[-1], output_dim)]))
    
    def forward(self, x):
        return self.model(x)

class FeatureTokenizerTransformer(nn.Module):
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float, 
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float):
        """
        Feature Tokenizer Transformer model.

        Parameters:
        - output_dim (int): Output dimension.
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features.
        - num_continuous_features (int): Number of continuous features.
        - embedding_dim (int): Embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - attn_dropout_rate (float): Dropout rate.
        - mlp_hidden_dims (List[int]): List of hidden dimensions for MLP.
        - activation (str): Activation function.
        - ffn_dropout_rate (float): Dropout rate for feedforward network.
        """
        super(FeatureTokenizerTransformer, self).__init__()
        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim),
            'continuous_feature_encoder': NumEncoder(num_continuous_features, embedding_dim)})
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.transformer = Transformer(embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate)
        self.classifier = MLP(embedding_dim, output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)

    def forward(self, categorical_x, continuous_x):
        batch_size = categorical_x.size(0)
        categorical_x = self.encoders['categorical_feature_encoder'](categorical_x)
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        x = torch.cat([categorical_x, continuous_x], dim=1)
        # Add CLS token
        cls_tokens = self.cls_token_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        # Get CLS token output
        cls_token_output = x[:, 0, :].view(batch_size, -1)
        # Get final output
        x = self.classifier(cls_token_output)
        return x