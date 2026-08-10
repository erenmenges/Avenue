import config
import torch
from torch import nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    def __init__(self, D_h: int, max_seq_len: int = config.SEQ_LEN, base: float = 10000.0):
        super().__init__()
        positions = torch.arange(0, max_seq_len)
        theta_tensor = base ** (-torch.arange(0, D_h, 2).float() / D_h)
        angles = positions[:, None] * theta_tensor[None, :]

        C = torch.cos(angles)
        S = torch.sin(angles)

        self.register_buffer("C", C)
        self.register_buffer("S", S)
    
    def apply_rope(self, X: torch.Tensor):
        assert X.shape[-2] <= self.C.shape[0], "sequence longer than RoPE positions table"
        x_values = X[..., 0::2]  ### (B, H, N, D_h/2)
        y_values = X[..., 1::2]  ### (B, H, N, D_h/2)

        new_x_values = x_values * self.C[:X.shape[-2]] - y_values * self.S[:X.shape[-2]]  ### (B, H, N, D_h/2)
        new_y_values = x_values * self.S[:X.shape[-2]] + y_values * self.C[:X.shape[-2]]  ### (B, H, N, D_h/2)

        result = torch.stack((new_x_values, new_y_values), dim=-1).flatten(-2)  ### interleave naturally

        return result

# ternary functions and class

def quantize_weights(W: torch.Tensor):
    quantized_W = W.float()  ### cast to fp32
    abs_mu = quantized_W.abs().mean(dim=-1, keepdim=True)  ### shape: (..., 1)
    abs_mu = torch.clamp(abs_mu, min=1e-5).detach()
    quantized_W = quantized_W / abs_mu  ### drop the scale. each param is now "how many avg weights is this?"
    quantized_W = torch.round(quantized_W)
    quantized_W = torch.clamp(quantized_W, min=-1, max=1)
    return quantized_W.to(W.dtype) * abs_mu

def quantize_activations(x: torch.Tensor):
    quantized_x = x.float()
    abs_max = quantized_x.abs().amax(dim=-1, keepdim=True)  ### shape: (..., 1)
    abs_max = torch.clamp(abs_max, min=1e-5).detach()
    scale = abs_max/127  ### scale: size of an integer step
    quantized_x = quantized_x / scale
    quantized_x = torch.round(quantized_x)
    quantized_x = torch.clamp(quantized_x, min=-128, max=127)
    return quantized_x.to(x.dtype) * scale

class Bitlinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.ln = nn.RMSNorm((in_features,))
        W_tensor = 0.02 * torch.randn((out_features, in_features))
        self.weight = nn.Parameter(W_tensor)

    def forward(self, x: torch.Tensor):
        forward_x = self.ln(x)
        x_quantized = quantize_activations(forward_x)  ### x is quantized to int8, W to ternary
        W_quantized = quantize_weights(self.weight)
        W_quantized = self.weight + (W_quantized - self.weight).detach()   ### forward uses quantized, backward uses latent weights
        x_quantized = forward_x + (x_quantized - forward_x).detach()
        y = x_quantized @ W_quantized.T
        return y.to(x.dtype)



class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int, RoPE: RoPE, ternary: bool = False):
        super().__init__()
        self.ternary = ternary

        self.D_h = D//H
        assert self.D_h % 2 == 0
        self.H = H
        self.rope = RoPE
        if not ternary:
            self.Q_layer = nn.Linear(in_features=D, out_features=D, bias=False)  ### layernorm already acts like bias
            self.K_layer = nn.Linear(in_features=D, out_features=D, bias=False)  ### layernorm already acts like bias
            self.V_layer = nn.Linear(in_features=D, out_features=D, bias=False)  ### layernorm already acts like bias
            self.O_layer = nn.Linear(in_features=D, out_features=D, bias=False)
        else:
            self.Q_layer = Bitlinear(in_features=D, out_features=D)
            self.K_layer = Bitlinear(in_features=D, out_features=D)
            self.V_layer = Bitlinear(in_features=D, out_features=D)
            self.O_layer = Bitlinear(in_features=D, out_features=D)

        if not ternary:
            self.ln1 = nn.RMSNorm(D)
            self.ln2 = nn.RMSNorm(D)
            self.MLP = nn.Sequential(nn.Linear(in_features=D, out_features=4*D, bias=False), nn.GELU(), nn.Linear(in_features=4*D, out_features=D, bias=False))
        else:
            self.MLP = nn.Sequential(Bitlinear(in_features=D, out_features=4*D), nn.GELU(), Bitlinear(in_features=4*D, out_features=D))


    def compute_qkv(self, X: torch.Tensor) -> tuple:
        """
        computes qkv values to pass to f.sdpa.
        """
        Q = self.Q_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        K = self.K_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        V = self.V_layer(X).reshape(X.shape[0], X.shape[1], -1, self.D_h).permute(0, 2, 1, 3)  ### (B, N, D)-->(B, N, D)-->(B, N, H, D_h)-->(B, H, N, D_h)
        return (self.rope.apply_rope(Q), self.rope.apply_rope(K), V)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        one transformer block. pre-norm layernorm and residuals.
        """
        B, N = X.shape[0], X.shape[-2]
        Q, K, V = self.compute_qkv(self.ln1(X)) if not self.ternary else self.compute_qkv(X)  ### pre-norm layernorm 1 before attention, this keeps softmax healthy
        sdpa_output = F.scaled_dot_product_attention(query=Q, key=K, value=V, is_causal=True)  ### (B, H, N, D_h)
        sdpa_output = sdpa_output.permute(0,2,1,3).reshape(B, N, -1)  ### combine heads to make (B, N, D)
        output = self.O_layer(sdpa_output)  ### (B, N, D) = (B, N, D) @ (1, D, D)
        output = X + output  ### residual 1
        output = output + self.MLP(self.ln2(output)) if not self.ternary else output + self.MLP(output) ### layernorm 2 before MLP, and residual 2
        return output



class Transformer(nn.Module):
    def __init__(self, K: int, D: int, H: int, V: int, ternary: bool = False):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=V, embedding_dim=D)

        assert D % H == 0, f"D must be divisible by H"
        self.RoPE = RoPE(D//H)

        layers = []
        for _ in range(K):
            if not ternary:
                layers.append(TransformerBlock(D, H, self.RoPE))
            else:
                layers.append(TransformerBlock(D, H, self.RoPE, True))

        self.main = nn.Sequential(*layers)

        self.ln_final = nn.RMSNorm(D)
        self.output_head = nn.Linear(in_features=D, out_features=V, bias=False)  ### bias=False like gpt-2. 

        self.output_head.weight = self.embeddings.weight  ### weight tying - embeddings and final head serve the same purpose but opposite directions

        # init scaling for everything except layernorm
        self.apply(self._init_weights)

        # init scaling for residuals, so variance doesn't explode
        for name, p in self.named_parameters():
            if name.endswith("O_layer.weight") or name.endswith("MLP.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * K))

    def _init_weights(self, module):
        """
        scale every parameter except layernorm so that std=0.02 
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, X: torch.Tensor, return_hidden: bool = False):
        embedded_X = self.embeddings(X)  ### (B,N) --> (B,N,D)
        intermediate = self.main(embedded_X)
        intermediate = self.ln_final(intermediate)
        if return_hidden:
            return intermediate
        return self.output_head(intermediate)
    