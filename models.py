# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

from config import (
    ENCODER_TOP_CONFIG, QUANTIZER_TOP_CONFIG,
    ENCODER_BOTTOM_CONFIG, QUANTIZER_BOTTOM_CONFIG,
    DECODER_CONFIG
)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(), # CORRECTED: Was nn.ReLU(True)
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        return x + self._block(x)


class EfficientCrossAttention(nn.Module):
    """
    Efficient cross-attention for hierarchical models.
    Bottom (fine) queries Top (coarse) using pooled keys/values.
    """

    def __init__(self, top_dim, bottom_dim, num_heads=4):
        super().__init__()
        hidden_dim = max(top_dim, bottom_dim)
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Conv2d(bottom_dim, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(top_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(top_dim, hidden_dim, kernel_size=1)

        self.out_proj = nn.Conv2d(hidden_dim, bottom_dim, kernel_size=1)
        self.norm = nn.GroupNorm(min(32, bottom_dim), bottom_dim)

    def forward(self, z_top, z_bottom):
        """
        z_top:    [B, Ct, Ht, Wt]  (coarse latent)
        z_bottom: [B, Cb, Hb, Wb]  (fine latent)
        """
        B, _, Hb, Wb = z_bottom.shape

        # Queries from bottom
        Q = self.q_proj(z_bottom)
        Q = Q.view(B, self.num_heads, self.head_dim, Hb * Wb)
        Q = Q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]

        # Keys & Values from top (global pooling)
        K = self.k_proj(z_top)
        V = self.v_proj(z_top)

        K = F.adaptive_avg_pool2d(K, 1).view(B, self.num_heads, self.head_dim, 1)
        V = F.adaptive_avg_pool2d(V, 1).view(B, self.num_heads, self.head_dim, 1)

        K = K.permute(0, 1, 3, 2)  # [B, heads, 1, head_dim]
        V = V.permute(0, 1, 3, 2)

        # Attention: [B, heads, HW, 1]
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention
        out = (attn @ V)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, -1, Hb, Wb)

        out = self.out_proj(out)
        return self.norm(out + z_bottom)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        
        # Improved orthogonal initialization using QR decomposition
        self._orthogonal_init()
        self._commitment_cost = commitment_cost

        # Federated Codebook Vitalization
        self.register_buffer("code_usage", torch.zeros(num_embeddings))
        self.register_buffer("batch_counter", torch.zeros(1))

        # Unlearning: inactive rows are never selected; vitalization can be switched off.
        # With the defaults (all rows active, vitalize on) behaviour is unchanged.
        self.register_buffer("active_mask", torch.ones(num_embeddings, dtype=torch.bool))
        self.vitalize = True

    @staticmethod
    def squared_distances(flat_input, codebook):
        """Squared distances [N, K] between flat latents [N, D] and codebook rows [K, D]."""
        return (torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(flat_input, codebook.t()))

    def nearest(self, z, codebook=None, mask=None, return_distances=False):
        """
        Nearest-row index map [B, H, W] for latents z of shape [B, D, H, W].
        Updates no counters and never vitalizes. Ties go to the lowest index.
        """
        codebook = self._embedding.weight if codebook is None else codebook
        mask = self.active_mask if mask is None else mask
        B, D, H, W = z.shape
        flat_input = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        distances = self.squared_distances(flat_input, codebook)
        if not bool(mask.all()):
            distances = distances.masked_fill(~mask.to(distances.device), float("inf"))
        indices = torch.argmin(distances, dim=1).view(B, H, W)
        if return_distances:
            return indices, distances
        return indices

    def _orthogonal_init(self):
        """Initialize codebook with orthogonal vectors for maximum diversity."""
        # Generate random matrix and orthogonalize using QR decomposition
        init_weight = torch.randn(self._num_embeddings, self._embedding_dim)
        if self._num_embeddings <= self._embedding_dim:
            q, _ = torch.linalg.qr(init_weight.t())
            self._embedding.weight.data = q.t()[:self._num_embeddings]
        else:
            # For overcomplete case, do it in batches
            q, _ = torch.linalg.qr(init_weight)
            self._embedding.weight.data = q
        # Scale to reasonable range
        self._embedding.weight.data *= 0.1

    def get_vitality_stats(self):
        usage = self.code_usage.clone().detach().cpu()
        active = (usage > 0).sum().item()
        dead = usage.numel() - active

        if usage.sum() > 0:
            p = usage / usage.sum()
            entropy = -(p[p > 0] * torch.log(p[p > 0])).sum().item()
        else:
            entropy = 0.0

        return {
            "active": active,
            "dead": dead,
            "entropy": entropy
        }


    def forward(self, inputs):
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self._embedding_dim)
        
        distances = self.squared_distances(flat_input, self._embedding.weight)
        if not bool(self.active_mask.all()):
            distances = distances.masked_fill(~self.active_mask, float("inf"))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        with torch.no_grad():
            flat_idx = encoding_indices.view(-1)
            self.code_usage.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
            self.batch_counter += 1

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()

        if self.training and self.vitalize:
            if self.batch_counter.item() % config.CODEBOOK_RESET_INTERVAL == 0:
                dead_codes = self.code_usage < config.CODEBOOK_DEAD_THRESHOLD
                if dead_codes.any():
                    with torch.no_grad():
                        z_flat = flat_input.detach()
                        rand_idx = torch.randint(0, z_flat.size(0), (dead_codes.sum(),), device=z_flat.device)
                        self._embedding.weight[dead_codes] = z_flat[rand_idx] + 0.01 * torch.randn_like(
                            self._embedding.weight[dead_codes])
                self.code_usage.zero_()  # Always reset usage counter at interval

            
        return loss, quantized.permute(0, 3, 1, 2).contiguous()




class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_residual_layers, num_residual_hiddens, downsample_factor):
        super().__init__()
        
        if downsample_factor == 1:
            layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        else:
            num_downsample_layers = int(torch.log2(torch.tensor(downsample_factor)).item())
            layers = [nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            for _ in range(num_downsample_layers - 1):
                layers.append(nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1))
        
        self.main = nn.Sequential(*layers)
        
        self.res_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, num_residual_hiddens) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        x = self.main(x)
        for layer in self.res_stack:
            x = layer(x)
        return x


# --- SUPER-RESOLUTION : CONDITIONAL DECODER WITH EFFICIENT ATTENTION ---
class ConditionalHierarchicalDecoder(nn.Module):
    def __init__(self, top_embedding_dim, bottom_embedding_dim, hidden_channels, num_residual_layers, num_residual_hiddens, out_channels):
        super().__init__()
        self.lr_condition_upsampler = nn.Sequential(
            nn.ConvTranspose2d(config.IMAGE_CHANNELS, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=hidden_channels, num_hiddens=hidden_channels, num_residual_hiddens=num_residual_hiddens),
            nn.ReLU()
        )
        
        self.pre_conv_top = nn.Conv2d(top_embedding_dim, hidden_channels, kernel_size=3, padding=1)
        self.pre_conv_bottom = nn.Conv2d(bottom_embedding_dim, hidden_channels, kernel_size=3, padding=1)
        
        # Efficient Cross-Level Attention (memory-friendly for H200)
        self.cross_attention = EfficientCrossAttention(
            top_dim=hidden_channels,
            bottom_dim=hidden_channels,
            num_heads=4
        )

        # Separate upsampling for top and bottom
        self.upsample_top_1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.upsample_top_2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.upsample_bottom = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fusion_conv = nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.res_stack = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, num_residual_hiddens) 
            for _ in range(num_residual_layers)
        ])
        
        self.upsample_final = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z_top, z_bottom, lr_condition):
        # Process latents
        z_top_proc = self.pre_conv_top(z_top)    
        z_bottom_proc = self.pre_conv_bottom(z_bottom)
        
        # Apply efficient cross-attention (bottom enhanced by top)
        z_bottom_enhanced = self.cross_attention(z_top_proc, z_bottom_proc)
        
        # Upsample to target resolution (256x256)
        z_top_upsampled = self.upsample_top_1(z_top_proc)       # 64->128
        z_top_upsampled = self.upsample_top_2(z_top_upsampled)  # 128->256
        
        z_bottom_upsampled = self.upsample_bottom(z_bottom_enhanced)  # 128->256
        
        lr_features = self.lr_condition_upsampler(lr_condition) 
        
        # Fuse all three streams (all at 256x256)
        x = torch.cat([z_top_upsampled, z_bottom_upsampled, lr_features], dim=1)  
        x = self.fusion_conv(x)  
        
        # Residual refinement
        for layer in self.res_stack:
            x = layer(x)
        x = self.upsample_final(x)
        x = torch.tanh(x)  # Constrain output to [-1, 1]
        return x


# --- SUPER-RESOLUTION : The Main VQ-VAE-2 Model ---
class VQVAE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_top = HierarchicalEncoder(**ENCODER_TOP_CONFIG)
        self.quantizer_top = VectorQuantizer(**QUANTIZER_TOP_CONFIG)
        self.pre_quant_conv_top = nn.Conv2d(ENCODER_TOP_CONFIG['hidden_channels'], QUANTIZER_TOP_CONFIG['embedding_dim'], kernel_size=1)
        
        self.encoder_bottom = HierarchicalEncoder(**ENCODER_BOTTOM_CONFIG)
        self.quantizer_bottom = VectorQuantizer(**QUANTIZER_BOTTOM_CONFIG)
        self.pre_quant_conv_bottom = nn.Conv2d(ENCODER_BOTTOM_CONFIG['hidden_channels'], QUANTIZER_BOTTOM_CONFIG['embedding_dim'], kernel_size=1)
        self.decoder = ConditionalHierarchicalDecoder(**DECODER_CONFIG)

    def forward(self, lr_image):
        z_top_unquantized = self.encoder_top(lr_image)
        z_top_unquantized = self.pre_quant_conv_top(z_top_unquantized)
        vq_loss_top, z_top_quantized = self.quantizer_top(z_top_unquantized)
        
        z_bottom_unquantized = self.encoder_bottom(lr_image)
        z_bottom_unquantized = self.pre_quant_conv_bottom(z_bottom_unquantized)
        vq_loss_bottom, z_bottom_quantized = self.quantizer_bottom(z_bottom_unquantized)
        reconstructed_image = self.decoder(z_top_quantized, z_bottom_quantized, lr_image)
        total_vq_loss = vq_loss_top + vq_loss_bottom
        return reconstructed_image, total_vq_loss

    def encode(self, lr_image):
        """Pre-quant latents (z_top, z_bottom), before the quantizer. No quantizer side effects."""
        z_top = self.pre_quant_conv_top(self.encoder_top(lr_image))
        z_bottom = self.pre_quant_conv_bottom(self.encoder_bottom(lr_image))
        return z_top, z_bottom

    def get_shared_state_dict(self):
        shared_dict = {}
        for name, param in self.named_parameters():
            if not name.startswith('decoder.'):
                shared_dict[name] = param.clone().detach().cpu()
        return shared_dict

    def load_shared_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)
    
    def get_codebook_vitality(self):
        return {
            "top": self.quantizer_top.get_vitality_stats(),
            "bottom": self.quantizer_bottom.get_vitality_stats()
        }


    def get_codebook_vectors(self):
        """Extract codebook vectors for similarity comparison"""
        codebooks = {
            'top_codebook': self.quantizer_top._embedding.weight.clone().detach().cpu(),
            'bottom_codebook': self.quantizer_bottom._embedding.weight.clone().detach().cpu()
        }
        return codebooks