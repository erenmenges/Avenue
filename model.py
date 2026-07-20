import torch
from torch import nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, D_h: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        positions = torch.arange(0, max_seq_len)
        theta_tensor = base ** (-torch.arange(0, D_h, 2).float() / D_h)
        angles = positions[:, None] * theta_tensor[None, :]

        C = torch.cos(angles)
        S = torch.sin(angles)

        self.register_buffer("C", C)
        self.register_buffer("S", S)
    
    def apply_rope(self, X: torch.Tensor):
        x_values = X[..., 0::2]  ### (B, H, N, D_h/2)
        y_values = X[..., 1::2]  ### (B, H, N, D_h/2)

        new_x_values = x_values * self.C[:X.shape[-2]] - y_values * self.S[:X.shape[-2]]  ### (B, H, N, D_h/2)
        new_y_values = x_values * self.S[:X.shape[-2]] + y_values * self.C[:X.shape[-2]]  ### (B, H, N, D_h/2)

        result = torch.zeros_like(X)
        result[..., 0::2] = new_x_values
        result[..., 1::2] = new_y_values

        return result

class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int, RoPE: RoPE):
        super().__init__()

        self.D_h = D//H
        self.H = H
        self.RoPE = RoPE

        self.Q_layer = nn.Linear(in_features=D, out_features=D)
        self.K_layer = nn.Linear(in_features=D, out_features=D)
        self.V_layer = nn.Linear(in_features=D, out_features=D)
        self.O_layer = nn.Linear(in_features=D, out_features=D)

        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.MLP = nn.Sequential(nn.Linear(in_features=D, out_features=4*D), nn.ReLU(), nn.Linear(in_features=4*D, out_features=D))

    def compute_qkv(self, X: torch.Tensor) -> tuple:
        Q = self.Q_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        K = self.K_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        V = self.V_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        return (self.RoPE.apply_rope(Q), self.RoPE.apply_rope(K), V)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N = X.shape[0], X.shape[-2]
        Q, K, V = self.compute_qkv(self.ln1(X))  ### pre-norm layernorm 1 before attention, this keeps softmax healthy
        sdpa_output = F.scaled_dot_product_attention(query=Q, key=K, value=V, is_causal=True)  ### (B, H, N, D_h)
        sdpa_output = sdpa_output.permute(0,2,1,3).reshape(B, N, -1)  ### combine heads to make (B, N, D)
        output = self.O_layer(sdpa_output)  ### (B, N, D) = (B, N, D) @ (1, D, D)
        output = X + output  ### residual 1
        output = output + self.MLP(self.ln2(output)) ### layernorm 2 before MLP, and residual 2
        return output



class Transformer(nn.Module):
    def __init__(self, K: int, D: int, H: int, V: int):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=V, embedding_dim=D)

        assert D % H == 0, f"D must be divisible by H"
        self.RoPE = RoPE(D//H)

        layers = []
        for _ in range(K):
            layers.append(TransformerBlock(D, H, self.RoPE))

        self.main = nn.Sequential(*layers)

        self.ln_final = nn.LayerNorm(D)
        self.output_head = nn.Linear(in_features=D, out_features=V)

    def forward(self, X: torch.Tensor):
        embedded_X = self.embeddings(X)  ### (B,N) --> (B,N,D)
        intermediate = self.main(embedded_X)
        intermediate = self.ln_final(intermediate)
        return self.output_head(intermediate)
    