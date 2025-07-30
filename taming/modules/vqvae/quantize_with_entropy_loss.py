import torch
from einops import rearrange, reduce
from taming.modules.vqvae.quantize import VectorQuantizer2
import torch.nn.functional as F

import torch.nn as nn
import numpy as np
from torch import einsum
from einops import rearrange

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

class VectorQuantizer2WithEntropyLoss(VectorQuantizer2):

    def __init__(self, n_e, e_dim, beta, normalize_embedding, remap=None, unknown_index="random", sane_index_shape=False, legacy=True,
                 diversity_gamma = 1., frac_per_sample_entropy=1.):
        self.diversity_gamma = diversity_gamma
        self.frac_per_sample_entropy = frac_per_sample_entropy
        super().__init__(n_e, e_dim, beta, normalize_embedding, remap, unknown_index, sane_index_shape, legacy)
        self.quantizer_normalize_embedding = normalize_embedding

        if self.quantizer_normalize_embedding:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

    def entropy_loss(self, distances, inv_temperature=100.):
        prob = (-distances * inv_temperature).softmax(dim = -1)
 
        if self.frac_per_sample_entropy < 1.:
            num_tokens = prob.shape[0]
            num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
            rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens
            per_sample_probs = prob[rand_mask]
        else:
            per_sample_probs = prob
        
        per_sample_entropy = entropy(per_sample_probs).mean()

        avg_prob = reduce(per_sample_probs, '... d -> d', 'mean')
        codebook_entropy = entropy(avg_prob).mean()

        entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        return entropy_aux_loss
    

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        if self.quantizer_normalize_embedding:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

 
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        perplexity = None
        min_encodings = None

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        if self.training:
            entropy_aux_loss = self.entropy_loss(d, inv_temperature=100.)
        else:
            entropy_aux_loss = None

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z = rearrange(z, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) 
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, (loss, entropy_aux_loss), (perplexity, min_encodings, min_encoding_indices) 