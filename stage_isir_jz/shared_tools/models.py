import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def spatial_penalty_tikhonov(weights, in_chans, h, w):
    """Pénalise la différence entre les pixels voisins (Lissage 1er ordre). Attention pas circulaire, donc les bords sont moins pénalisés."""
    if weights.numel() == 0: return 0.0
    w_spatial = weights.view(-1, in_chans, h, w)
    
    # Différences finies verticales et horizontales
    diff_h = torch.sum((w_spatial[:, :, 1:, :] - w_spatial[:, :, :-1, :]) ** 2)
    diff_w = torch.sum((w_spatial[:, :, :, 1:] - w_spatial[:, :, :, :-1]) ** 2)
    return diff_h + diff_w

def spatial_penalty_laplacian(weights, in_chans, h, w):
    """Pénalise la courbure (Lissage 2ème ordre, préserve mieux les pics). Attention pas circulaire, donc les bords sont moins pénalisés."""
    if weights.numel() == 0: return 0.0
    w_spatial = weights.view(-1, in_chans, h, w)
    
    # Opérateur Laplacien (pixels haut, bas, gauche, droite - 4 * centre)
    laplacian = (
        w_spatial[:, :, 2:, 1:-1] + w_spatial[:, :, :-2, 1:-1] + 
        w_spatial[:, :, 1:-1, 2:] + w_spatial[:, :, 1:-1, :-2] - 
        4 * w_spatial[:, :, 1:-1, 1:-1]
    )
    return torch.sum(laplacian ** 2)


def compute_loss(preds, target, loss_type, quantiles=None, reduction='mean'):
    """
    Calcule la loss (MSE, L1 ou Quantile).
    Si reduction='none', retourne la loss par échantillon (batch_size,).
    """
    if loss_type == 'mse':
        mse = F.mse_loss(preds, target, reduction='none')
        if reduction == 'mean': return mse.mean()
        return mse.mean(dim=1)
        
    elif loss_type == 'l1':
        l1 = F.l1_loss(preds, target, reduction='none')
        if reduction == 'mean': return l1.mean()
        return l1.mean(dim=1)
        
    elif loss_type == 'quantile':
        # preds: (bs, latent_dim * n_quantiles)
        # target: (bs, latent_dim)
        bs, _ = target.shape
        n_q = len(quantiles)
        
        # Reshape preds -> (bs, latent_dim, n_quantiles)
        preds_reshaped = preds.view(bs, -1, n_q)
        # Reshape target -> (bs, latent_dim, 1) pour le broadcast
        target_expanded = target.unsqueeze(-1)
        
        # Tenseur des quantiles -> (1, 1, n_quantiles)
        q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=preds.device).view(1, 1, n_q)
        
        errors = target_expanded - preds_reshaped
        # Pinball loss : max(q * e, (q - 1) * e)
        loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        
        if reduction == 'mean': 
            return loss.mean()
        # Moyenne sur la dimension latente et les quantiles pour avoir une loss par sample
        return loss.mean(dim=(1, 2)) 
    
    elif loss_type == 'correlation':
         corr_val = correlation_loss(preds, target)
         if reduction == 'mean':
             return corr_val
         else:
             # Broadcast manuel : crée un vecteur de taille batch_size rempli de la valeur globale
             return torch.full((preds.shape[0],), corr_val, device=preds.device)

def get_median_prediction(preds, loss_type, quantiles, latent_dim):
    """Extrait la prédiction médiane (q=0.5) pour la reconstruction et l'évaluation visuelle."""
    if loss_type != 'quantile':
        return preds
        
    median_idx = quantiles.index(0.5)
    bs = preds.size(0)
    preds_reshaped = preds.view(bs, latent_dim, len(quantiles))
    return preds_reshaped[:, :, median_idx]

def get_median_prediction_full_slp(preds, loss_type, quantiles, out_dim=None):
    """Extrait la prédiction médiane (q=0.5), compatible vecteurs latents et cartes spatiales 2D."""
    if loss_type != "quantile":
        return preds

    median_idx = quantiles.index(0.5)

    # CAS 1 : Sortie Spatiale 4D -> (Batch, Quantiles, Height, Width)
    if preds.dim() == 4:
        # On extrait simplement le canal correspondant au quantile médian
        return preds[:, median_idx : median_idx + 1, :, :]
        # Note: on garde le :median_idx+1 pour conserver la dimension du canal (B, 1, H, W)

    # CAS 2 : Sortie Latente 2D/3D -> (Batch, latent_dim * n_quantiles)
    else:
        bs = preds.size(0)
        n_q = len(quantiles)
        # Utilisation de .reshape par sécurité au lieu de .view
        preds_reshaped = preds.reshape(bs, -1, n_q)
        return preds_reshaped[:, :, median_idx]
    
def decode_latent_to_map(predicted_latent, args, latent_dim, pca_model=None, vae_model=None, safe_wgts=None):
    """Décode un tenseur latent vers l'espace physique SLP (B, 1, 53, 113)."""
    if args.loss_type == 'quantile':
        med_latent = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, latent_dim)
    else:
        med_latent = predicted_latent
        
    if args.embed_method == 'pca':
        pred_np = med_latent.detach().cpu().numpy()
        padded_pred = np.zeros((pred_np.shape[0], pca_model.n_components_))
        padded_pred[:, :latent_dim] = pred_np
        
        slp_flat_polluted = pca_model.inverse_transform(padded_pred)
        if args.lat_weight and safe_wgts is not None:
            slp_flat = slp_flat_polluted / safe_wgts
        else:
            slp_flat = slp_flat_polluted
        return torch.tensor(slp_flat.reshape(-1, 1, 53, 113), dtype=torch.float32, device=predicted_latent.device)
        
    elif args.embed_method == 'vae':
        with torch.no_grad():
            decoded_slp = vae_model.decode(med_latent)
        return decoded_slp

# ============================================================
# La loss custom de Clara pour la PC1, à comparer avec une MSE classique
# ============================================================

def quantile_loss(preds, target, quantiles):
    """
    Computes the pinball (quantile) loss function 
    
    Args:
        preds : Tensor, Predicted quantile values with shape (batch_size, n_quantiles).
        target : Tensor, Targeted values with shape (batch_size).
        quantiles : Tensor, Quantile levels

    Returns:
        Tensor, Mean quantile loss over all samples and quantiles.
    """
    
    losses = []
    for i, q in enumerate(quantiles):
        errors = target.unsqueeze(1) - preds[:, i].unsqueeze(1)
        losses.append(torch.max((q - 1) * errors, q * errors))
    
    return torch.mean(torch.cat(losses, dim=1))

# ============================================================
# VAE pour essayer de embed 
# classe un peu redondante avec le dossier data_analysis ou essaie d'embed à part 
# il me semble qu'il y a aussi une version avec crop plutôt que du padding dans les convolutions? 
# ============================================================

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Attention : la taille 6720 dépend de la taille d'entrée SLP (ici assumée ~53x113)
        self.flatten_size = 64 * 7 * 15 
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 7, 15)
        x = self.decoder(x)
        # Interpolation pour retomber exactement sur (53, 113)
        x = F.interpolate(x, size=(53, 113), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, wgts_tensor=None):
    if wgts_tensor is not None:
        # NOUVEAU : wgts_tensor est déjà cos(lat), plus besoin de le mettre au carré
        sq_error = (recon_x - x) ** 2
        weighted_sq_error = sq_error * wgts_tensor 
        MSE = torch.sum(weighted_sq_error) / x.shape[0]
    else:
        MSE = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return MSE + beta * KLD, MSE, KLD

# ============================================================
# PATCH EMBEDDING LAYER  
# ============================================================
class PatchEmbedding(nn.Module):
    """Découpe la carte spatiale en patches.

    Pour la SST de taille (85, 360) et un patch de (5, 10), on obtient une grille de 17x36 = 612 patches.
    Chaque patch de 5x10 pixels est projeté en un vecteur d'embedding de dimension `embed_dim` (ex: 128).
    
    Pour la SLP de taille (53, 113) et un patch de (5, 10), on obtient une grille de 11x12 = 132 patches à prédire,
    c'est-à-dire 6600 = 132 patches * 50 pixels par patch (5*10) à reconstruire = 55 * 120 qu'on recroppera ensuite à 53 * 113.
    """
    def __init__(self, img_size=(85, 360), patch_size=(5, 10), in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # ce qui est suit est inutile pour certains modèles donc éventuellement à uniformiser mais voilà
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        x = self.proj(x)  # -> (Batch, embed_dim, grid_H, grid_W)
        x = x.flatten(2)  # -> (Batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # -> (Batch, num_patches, embed_dim)
        return x


    
def get_fast_labels(slp_batch, ref_dict, metric='mse', projector=None, device='cpu'):
    """
    Convertit un batch de cartes SLP (B, 53, 113) en un tenseur de labels (B,)
    Les métriques mse et correlation sont générales et devraient marcher dans tous les cas, même sans projecteur. 
    """
    B = slp_batch.shape[0]
    
    # -------------------------------------------------------------
    # 1. MÉTRIQUE POUR LES QUANTILES DE LA PC1
    # -------------------------------------------------------------
    if metric == 'pc1_quantiles':
        if projector is None:
            raise ValueError("Le modèle PCA (projector) doit être fourni.")
        if 'pc1_bins' not in ref_dict:
            raise ValueError("Le dictionnaire ne contient pas 'pc1_bins'.")

        bins = ref_dict['pc1_bins']
        slp_flat = slp_batch.reshape(B, -1)
        
        # Projection PCA et extraction de la PC1
        latent_batch = projector.transform(slp_flat)
        pc1 = latent_batch[:, 0]
        
        # np.digitize renvoie l'index de l'intervalle (ex: 1 pour le premier intervalle)
        # On soustrait 1 pour avoir des labels qui commencent à 0 (0, 1, 2, 3...)
        labels = np.digitize(pc1, bins) - 1
        
        # Sécurité pour les valeurs extrêmes qui dépasseraient légèrement les bins
        labels = np.clip(labels, 0, len(bins) - 2)
        
        return torch.tensor(labels, dtype=torch.long)

# -------------------------------------------------------------
    # 2 & 3. MÉTRIQUES DANS L'ESPACE PHYSIQUE (Pixels)
    # -------------------------------------------------------------
    if metric in ['correlation', 'mse']:
        # On récupère directement les clés triées qui nous intéressent
        regime_keys = sorted([k for k in ref_dict.keys() if k.endswith("_slp_0_mean") and not k.startswith("GLOBAL")])
        
        # On extrait les matrices
        ref_slp = np.array([ref_dict[k] for k in regime_keys])
        
        slp_flat = slp_batch.reshape(B, -1)
        ref_slp_flat = ref_slp.reshape(len(regime_keys), -1)
        
        if metric == 'correlation':
            s_c = slp_flat - slp_flat.mean(axis=1, keepdims=True)
            r_c = ref_slp_flat - ref_slp_flat.mean(axis=1, keepdims=True)
            s_n = np.linalg.norm(s_c, axis=1, keepdims=True)
            r_n = np.linalg.norm(r_c, axis=1, keepdims=True)
            s_n[s_n == 0], r_n[r_n == 0] = 1e-10, 1e-10
            
            corr_matrix = np.dot(s_c, r_c.T) / (s_n * r_n.T) 
            labels = np.argmax(corr_matrix, axis=1) 
            
        elif metric == 'mse':
            diff = slp_flat[:, np.newaxis, :] - ref_slp_flat[np.newaxis, :, :]
            mse_matrix = np.mean(diff**2, axis=2) 
            labels = np.argmin(mse_matrix, axis=1) 
            
    # -------------------------------------------------------------
    # 4. MÉTRIQUE DANS L'ESPACE LATENT K-MEANS # je ne sais pas si cette option marche
    # -------------------------------------------------------------
    elif metric == 'mse_latent':
        if projector is None or 'ref_centroids_latent' not in ref_dict:
            raise ValueError("Projector ou 'ref_centroids_latent' manquant.")

        ref_latent = ref_dict['ref_centroids_latent']
        slp_flat = slp_batch.reshape(B, -1)
        
        # Pour ton VAE ou PCA
        if str(ref_dict.get('embedding_method', 'pca')) == 'pca':
            latent_batch = projector.transform(slp_flat)
        else:
            slp_tensor = torch.tensor(slp_batch).unsqueeze(1).float().to(device)
            with torch.no_grad():
                latent_batch, _ = projector.encode(slp_tensor)
                latent_batch = latent_batch.cpu().numpy()

        diff = latent_batch[:, np.newaxis, :] - ref_latent[np.newaxis, :, :]
        mse_matrix = np.mean(diff**2, axis=2)
        labels = np.argmin(mse_matrix, axis=1)
        
    else:
        raise ValueError(f"Métrique inconnue : {metric}")
        
    return torch.tensor(labels, dtype=torch.long)
    

def old_get_fast_labels(slp_batch, ref_dict, metric='correlation'):
    """
    Convertit un batch de cartes SLP (B, 53, 113) en un tenseur de labels (B,)
    """
    B = slp_batch.shape[0]
    
    # 1. Extraction des 4 cartes de référence SLP (dans l'ordre 0, 1, 2, 3)
    regime_prefixes = []
    for i in range(1, 5): 
        for key in ref_dict.keys():
            if key.startswith(f"regime_{i}_") and key.endswith("_slp_0"):
                regime_prefixes.append(key.replace("_slp_0", ""))
                break
    
    ref_slp = np.array([ref_dict[f"{prefix}_slp_0"] for prefix in regime_prefixes])
    
    # 2. Aplatissement
    slp_flat = slp_batch.reshape(B, -1) # (B, 5989)
    ref_slp_flat = ref_slp.reshape(4, -1) # (4, 5989)
    
    # 3. Calcul
    if metric == 'correlation':
        s_c = slp_flat - slp_flat.mean(axis=1, keepdims=True)
        r_c = ref_slp_flat - ref_slp_flat.mean(axis=1, keepdims=True)
        s_n = np.linalg.norm(s_c, axis=1, keepdims=True)
        r_n = np.linalg.norm(r_c, axis=1, keepdims=True)
        s_n[s_n == 0], r_n[r_n == 0] = 1e-10, 1e-10
        
        corr_matrix = np.dot(s_c, r_c.T) / (s_n * r_n.T) # (B, 4)
        labels = np.argmax(corr_matrix, axis=1) # On prend la corrélation MAX
        
    elif metric == 'mse':
        diff = slp_flat[:, np.newaxis, :] - ref_slp_flat[np.newaxis, :, :]
        mse_matrix = np.mean(diff**2, axis=2) # (B, 4)
        labels = np.argmin(mse_matrix, axis=1) # On prend l'erreur MIN
        
    # Retourne un tenseur PyTorch formaté pour la CrossEntropy
    return torch.tensor(labels, dtype=torch.long)

def pearson_correlation(y_pred, y_true, dim=0):
    """Calcule le coefficient de Pearson sur une dimension donnée (par défaut le batch)."""
    # Centrage
    y_pred_centered = y_pred - y_pred.mean(dim=dim, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=dim, keepdim=True)
    
    # Covariance et écarts-types
    cov = (y_pred_centered * y_true_centered).sum(dim=dim)
    std_pred = torch.sqrt((y_pred_centered ** 2).sum(dim=dim) + 1e-8)
    std_true = torch.sqrt((y_true_centered ** 2).sum(dim=dim) + 1e-8)
    
    corr = cov / (std_pred * std_true)
    return corr

def correlation_loss(y_pred, y_true, mode='global'):
    """
    Loss de corrélation (1 - Pearson r).
    
    Args:
        mode (str): 
            - 'component' : Corrélation par composante sur le batch, puis moyenne.
            - 'global' (default): Corrélation sur toutes les composantes et tout le batch mélangés. Pondère par naturellement selon la variance ds composantes et comme les composantes sont centrées ce n'est pas genant
    """
    if mode == 'component':
        # Centrage le long de la dimension du batch (dim=0)
        pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
        true_centered = y_true - y_true.mean(dim=0, keepdim=True)
        
        # Similarité cosinus sur les vecteurs centrés = Pearson r par composante
        corr = F.cosine_similarity(pred_centered, true_centered, dim=0, eps=1e-8)
        return 1.0 - corr.mean()
        
    elif mode == 'global':
        # Centrage sur l'ensemble de la matrice (un seul scalaire moyen par tenseur)
        pred_centered = y_pred.flatten() - y_pred.mean()
        true_centered = y_true.flatten() - y_true.mean()
        
        corr = F.cosine_similarity(
            pred_centered.unsqueeze(0), 
            true_centered.unsqueeze(0), 
            eps=1e-8
        )
        return 1.0 - corr.squeeze()
        
    else:
        raise ValueError(f"Mode inconnu : {mode}. Utilisez 'component' ou 'global'.")

# ============================================================
# CLASSE WRAPPER POUR SHAP
# ============================================================
class SHAP_Embedding_Wrapper(nn.Module):
    def __init__(self, base_model, loss_type, quantiles, latent_dim):
        super().__init__()
        self.base_model = base_model
        self.loss_type = loss_type
        self.quantiles = quantiles
        self.latent_dim = latent_dim

    def forward(self, *inputs):
        x_sst = inputs[0]
        x_slp = inputs[1] if len(inputs) > 1 else None
        
        out = self.base_model(x_sst, x_slp)
        
        if self.loss_type == 'quantile':
            return get_median_prediction_full_slp(out, self.loss_type, self.quantiles, self.latent_dim)
        return out