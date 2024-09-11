from typing import Dict, List
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.models.base_blocks import Activation, ColumnEmbedding

class CatEncoder(nn.Module):
    def __init__(self, 
                 vocabulary: Dict[str, Dict[str, int]],
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, 
                 dropout_rate: float,total_fea:int):
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
        self.total_fea=total_fea
        self.vocabulary = vocabulary

        self.embedding_dim=embedding_dim
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

    def forward(self, x: torch.Tensor,  continuous_x_res:torch.Tensor):
        batch_size = x.size(0)
        x = [self.model['column_embedding_layer'](x[:, i], col) for i, col in enumerate(self.vocabulary)]
        # x = torch.stack(x, dim=1)
        x = torch.stack(x, dim=1).view(batch_size, -1)

        x = torch.cat([x, continuous_x_res], dim=-1)
        x = x.view(batch_size, self.total_fea, self.embedding_dim)


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
    













class TabKANet(nn.Module):
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float,
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float):

        super(TabKANet, self).__init__()
 
        self.embedding_dim=embedding_dim
        self.len_vocabulary=len(vocabulary)
        self.num_continuous_features = num_continuous_features


        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate,self.num_continuous_features+self.len_vocabulary),
            'continuous_feature_encoder': NumEncoderTransformer(num_continuous_features, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate),
        })


        self.classifier = MLP(embedding_dim * (len(vocabulary) + num_continuous_features), output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)


    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):

        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        categorical_x = self.encoders['categorical_feature_encoder'](categorical_x,continuous_x)
        x = torch.cat([categorical_x, continuous_x], dim=-1)
        x = self.classifier(categorical_x)
        return x
    













class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        Initialize the KAN model.

        Parameters:
            layers_hidden (list): List containing the number of input features for each hidden layer.
            grid_size (int): Grid size, default is 5.
            spline_order (int): Order of the piecewise polynomial, default is 3.
            scale_noise (float): Scaling noise, default is 0.1.
            scale_base (float): Base scaling, default is 1.0.
            scale_spline (float): Scaling for the piecewise polynomial, default is 1.0.
            base_activation (torch.nn.Module): Base activation function, default is SiLU.
            grid_eps (float): Grid adjustment parameter, default is 0.02.
            grid_range (list): Grid range, default is [-1, 1].
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):

        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )





class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  
        spline_order=3, 
        scale_noise=0.1,  
        scale_base=1.0,   
        scale_spline=1.0,  
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  
        grid_eps=0.02,
        grid_range=[-1, 1],  
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size 
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   
        grid = ( 
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) 
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise 
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( 
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """

        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
    
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(
            0, 1 
        ) 
        B = y.transpose(0, 1)  
        solution = torch.linalg.lstsq(   
            A, B
        ).solution 
        result = solution.permute( 
            2, 0, 1
        ) 

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):

        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): 

        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) 
        spline_output = F.linear( 
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01): 
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out) 
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] 
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
   
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

