"""
The fused observation of a step: one linear token per input (the query: clinical features + mask flags; each view: its embedding, + a type
embedding), cross-attention of the query over {a learned null token, the present views} (a missing view is masked; the null token is
always a key), residual + layer norm, then an MLP of [result, presence flags] -> (mean, logvar). With 3 tokens the attention is a
patient-specific gate on the masked sum of the views (the H update of ALDSL). No product of experts: in runs/hs_fused_sde_20261026-155353
the clinical expert took 97-99% of the fused precision.
"""
import torch
import torch.nn as nn

from experiments.latent.parts.layers import gauss, mlp
from experiments.latent.parts.masking import key_padding


class FusionEncoder(nn.Module):
    def __init__(self, d_query, views, d_tok, heads, d_out):
        """views: {name: embedding width}, in the order of the keys"""
        super().__init__()
        self.query = nn.Linear(d_query, d_tok)
        self.tok = nn.ModuleDict({k: nn.Linear(d, d_tok) for k, d in views.items()})
        self.null = nn.Parameter(0.02 * torch.randn(d_tok))
        self.type_emb = nn.ParameterDict({k: nn.Parameter(0.02 * torch.randn(d_tok)) for k in views})
        self.att = nn.MultiheadAttention(d_tok, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_tok)
        self.out = mlp(d_tok + len(views), 2 * d_out)
        self.tokens = self.last_att = None

    def forward(self, query_in, views):
        """query_in (B, N, d_query); views {name: (embedding (B, N, d), presence (B, N, 1))} -> (mean, logvar) (B, N, d_out); kept:
        tokens (the view tokens), last_att (B, N, 1 + views: the attention weights, mean over the heads)"""
        B, N = query_in.shape[:2]
        q = self.query(query_in)
        self.tokens = tuple(self.tok[k](e) for k, (e, _) in views.items())
        kv = torch.stack([self.null.expand(B, N, -1)] + [t + self.type_emb[k] for t, k in zip(self.tokens, views)], 2)
        pres = [p for _, p in views.values()]
        d, V = q.shape[-1], kv.shape[2]
        a, w = self.att(q.reshape(B * N, 1, d), kv.reshape(B * N, V, d), kv.reshape(B * N, V, d),
                        key_padding_mask=key_padding(pres).reshape(B * N, V))
        self.last_att = w.reshape(B, N, V)
        h = self.norm(q + a.reshape(B, N, d))
        return gauss(self.out(torch.cat([h] + pres, -1)))
