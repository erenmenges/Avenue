import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int):
        super().__init__()

        assert D % H == 0, f"D must be divisible by H"
        self.D_h = D//H
        self.H = H
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
        return (Q, K, V)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N = X.shape[0], X.shape[-2]
        Q, K, V = self.compute_qkv(X)
        sdpa_output = F.scaled_dot_product_attention(query=Q, key=K, value=V, is_causal=True)  ### (B, H, N, D_h)
        sdpa_output = sdpa_output.permute(0,2,1,3).reshape(B, N, -1)  ### combine heads to make (B, N, D)
        output = self.O_layer(sdpa_output)  ### (B, N, D) = (B, N, D) @ (1, D, D)
        output = X + output  ### residual 1
        output = self.ln1(output)  ### layernorm 1
        output = output + self.MLP(output) ### MLP and residual 2
        output = self.ln2(output)  ### layernorm 2
        return output
    
class Transformer(nn.Module):
    def __init__(self, K: int, D: int, H: int, V: int):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=V, embedding_dim=D)

        layers = []
        for _ in range(K):
            layers.append(TransformerBlock(D, H))

        self.main = nn.Sequential(*layers)

        self.output_head = nn.Linear(in_features=D, out_features=V)

    def forward(self, X: torch.Tensor):
        embedded_X = self.embeddings(X)  ### (B,N) --> (B,N,D)
        intermediate = self.main(embedded_X)
        return self.output_head(intermediate)
    