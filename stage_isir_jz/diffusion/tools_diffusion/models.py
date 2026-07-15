import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# NOUVEAU : FONCTION DE CALCUL DU CRPS (PyTorch)
# ============================================================
def compute_crps(ensemble, observation):
    """
    Calcule le CRPS (Continuous Ranked Probability Score) de manière vectorisée.
    ensemble: Tensor de dimension (batch_size, n_members, latent_dim)
    observation: Tensor de dimension (batch_size, latent_dim)
    """
    n_members = ensemble.size(1)
    
    # On trie l'ensemble sur la dimension des membres
    ensemble, _ = torch.sort(ensemble, dim=1)
    
    # Terme 1: Erreur Absolue Moyenne (MAE) par rapport à la distribution
    mae = torch.mean(torch.abs(ensemble - observation.unsqueeze(1)), dim=1) # (batch, latent_dim)
    
    # Terme 2: Écartement (Spread) intra-ensemble
    # Formule astucieuse pour calculer la somme des différences absolues de paires triées
    i = torch.arange(1, n_members + 1, device=ensemble.device).view(1, n_members, 1)
    spread = torch.sum((2 * i - n_members - 1) * ensemble, dim=1) / (n_members * n_members) # (batch, latent_dim)
    
    crps = mae - spread
    return torch.mean(crps) # Moyenne sur le batch et les dimensions latentes

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalDenoiserMLP(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=256, time_dim=128, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        # Projection de l'entrée bruitée, du temps et de la condition
        self.proj_in = nn.Linear(latent_dim + time_dim + cond_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, t, condition):
        t_emb = self.time_mlp(t)
        # Concaténation classique (on pourrait utiliser du Cross-Attention ou AdaLN pour plus de raffinement)
        x = torch.cat([z_t, t_emb, condition], dim=-1)
        x = self.proj_in(x)
        return self.net(x)

class LatentDiffusionModel(nn.Module):
    def __init__(self, condition_encoder, denoiser, num_timesteps=1000):
        super().__init__()
        self.condition_encoder = condition_encoder
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        
        # Scheduler linéaire standard (DDPM)
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x_sst, x_slp, z_true):
        # 1. Extraction de la condition via ton CNN
        condition = self.condition_encoder(x_sst, x_slp)
        
        # 2. Échantillonnage du temps et ajout du bruit
        batch_size = z_true.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=z_true.device).long()
        noise = torch.randn_like(z_true)
        
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar_t) * z_true + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 3. Prédiction du bruit
        noise_pred = self.denoiser(z_t, t, condition)
        
        return noise, noise_pred
    
    @torch.no_grad()
    def sample(self, x_sst, x_slp, shape):
        device = x_sst.device
        condition = self.condition_encoder(x_sst, x_slp)
        z = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = self.denoiser(z, t, condition)
            
            alpha_t = self.alpha[t].unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
            beta_t = self.beta[t].unsqueeze(-1)
            
            if i > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
                
            # Équation de mise à jour DDPM standard
            z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise
            
        return z
    def sample_ddim(self, x_sst, x_slp, shape, ddim_steps=50, eta=0.0):
        """
        Échantillonneur DDIM rapide.
        ddim_steps : Nombre de pas d'inférence (ex: 50 au lieu de 1000).
        eta : 0.0 pour un échantillonnage déterministe, 1.0 pour se rapprocher de DDPM.
        """
        device = x_sst.device
        condition = self.condition_encoder(x_sst, x_slp)
        z = torch.randn(shape, device=device)
        
        # Création de la sous-séquence temporelle (ex: sauts de 20 en 20)
        step_ratio = self.num_timesteps // ddim_steps
        timesteps = (np.arange(0, ddim_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        
        for i, t_val in enumerate(timesteps):
            # Création du tenseur de temps pour le batch
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            prev_t_val = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            
            # Prédire le bruit avec le denoiser
            pred_noise = self.denoiser(z, t, condition)
            
            # Récupérer les alphas cumulés (alpha_bar) actuel et précédent
            alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
            if prev_t_val >= 0:
                prev_t = torch.full((shape[0],), prev_t_val, device=device, dtype=torch.long)
                alpha_bar_prev = self.alpha_bar[prev_t].unsqueeze(-1)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            
            # Calcul de la variance stochastique (sigma_t)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            
            # 1. Estimation de l'état d'origine pur (z_0)
            pred_x0 = (z - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            
            # 2. Direction pointant vers x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
            
            # 3. Ajout du bruit aléatoire (uniquement si eta > 0)
            noise = torch.randn_like(z) if t_val > 0 else torch.zeros_like(z)
            
            # Mise à jour de l'état vers le temps précédent
            z = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
            
        return z

class ResidualBlock1D(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Projection de la condition et du temps pour moduler les features (AdaLN)
        self.cond_proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, cond_emb):
        residual = x
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.fc1(x)
        
        # Modulation via la condition
        scale, shift = self.cond_proj(cond_emb).chunk(2, dim=-1)
        x = x * (1 + scale) + shift
        
        x = self.norm2(x)
        x = self.act2(x)
        x = self.fc2(x)
        
        return x + residual

class ConditionalResNetDenoiser1D(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=128, time_dim=128, hidden_dim=512, num_blocks=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        # On projette le latent bruité
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        
        # Les blocs résiduels
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, time_dim + cond_dim) for _ in range(num_blocks)
        ])
        
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, t, condition):
        t_emb = self.time_mlp(t)
        # L'embedding de condition est temps + contexte
        cond_emb = torch.cat([t_emb, condition], dim=-1)
        
        x = self.proj_in(z_t)
        for block in self.blocks:
            x = block(x, cond_emb)
            
        return self.proj_out(x)