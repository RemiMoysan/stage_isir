import torch

def encode_to_latent_gpu(true_maps, embed_method, current_latent_dim, pca_components_gpu=None, pca_mean_gpu=None, wgts_gpu=None, vae_model=None):
    """Encodage rapide sur GPU gérant la tronquature (slicing) PCA et VAE. Le pca_components_gpu doit avoir la bonne taille pour donner le latent_dim souhaité."""
    B = true_maps.size(0)
    if embed_method == 'pca':
        slp_flat = true_maps.view(B, -1)
        if wgts_gpu is not None:
            slp_flat = slp_flat * wgts_gpu
        target_embed = (slp_flat - pca_mean_gpu) @ pca_components_gpu.T
    elif embed_method == 'vae':
        with torch.no_grad():
            target_embed, _ = vae_model.encode(true_maps)
    return target_embed

def decode_to_spatial_map_gpu(latent_preds, embed_method, pca_components_gpu=None, pca_mean_gpu=None, wgts_gpu=None, vae_model=None):
    """Décodage rapide sur GPU gérant la tronquature PCA et VAE."""
    B = latent_preds.size(0)
    if embed_method == 'pca':
        pred_maps_flat = (latent_preds @ pca_components_gpu) + pca_mean_gpu
        if wgts_gpu is not None:
            pred_maps_flat = pred_maps_flat / wgts_gpu
        pred_maps = pred_maps_flat.view(B, 1, 53, 113)
    elif embed_method == 'vae':
        with torch.no_grad():
            pred_maps = vae_model.decode(latent_preds)
    return pred_maps

def compute_targeted_spatial_metrics(pred_maps, true_maps, wgts_gpu=None):
    """Calcule R2, L1 et Corrélation Spatio-Temporelle Globale (Pondérés)."""
    B, C, H, W = pred_maps.shape
    p_flat = pred_maps.view(B, -1)
    t_flat = true_maps.view(B, -1)

    if wgts_gpu is not None:
        area_weights = wgts_gpu**2
        w_norm = area_weights / torch.sum(area_weights)
    else:
        w_norm = torch.ones(H*W, dtype=torch.float32, device=pred_maps.device) / (H*W)

    # 1. R^2 GLOBAL
    global_mse = torch.sum(torch.mean((p_flat - t_flat)**2, dim=0) * w_norm)
    t_time_mean = torch.mean(t_flat, dim=0, keepdim=True)
    global_var = torch.sum(torch.mean((t_flat - t_time_mean)**2, dim=0) * w_norm)
    global_r2 = 1.0 - (global_mse / global_var) if global_var > 0 else torch.tensor(0.0)

    # 2. L1 SKILL SCORE GLOBAL
    global_l1 = torch.sum(torch.mean(torch.abs(p_flat - t_flat), dim=0) * w_norm)
    global_mad = torch.sum(torch.mean(torch.abs(t_flat), dim=0) * w_norm)
    global_l1_score = 1.0 - (global_l1 / global_mad) if global_mad > 0 else torch.tensor(0.0)

    # 3. CORRÉLATION GLOBALE SPATIO-TEMPORELLE
    w_tensor = w_norm.unsqueeze(0)
    p_glob_mean = torch.mean(torch.sum(p_flat * w_tensor, dim=1))
    t_glob_mean = torch.mean(torch.sum(t_flat * w_tensor, dim=1))
    
    cov_glob = torch.mean(torch.sum((p_flat - p_glob_mean) * (t_flat - t_glob_mean) * w_tensor, dim=1))
    p_var_glob = torch.mean(torch.sum(((p_flat - p_glob_mean)**2) * w_tensor, dim=1))
    t_var_glob = torch.mean(torch.sum(((t_flat - t_glob_mean)**2) * w_tensor, dim=1))
    
    global_corr = torch.where((p_var_glob * t_var_glob) > 1e-6, cov_glob / torch.sqrt(p_var_glob * t_var_glob), torch.tensor(0.0, device=pred_maps.device))

    return global_r2.item(), global_l1_score.item(), global_corr.item()

def compute_targeted_embedding_metrics(pred_embeds, true_embeds):
    """
    Calcule R2, L1 et Corrélation globale directement sur l'espace latent.
    Les tenseurs sont de forme [B, latent_dim]. 
    Tout est aplati pour un calcul global sur l'ensemble du batch et des composantes.
    """
    p_flat = pred_embeds.flatten()
    t_flat = true_embeds.flatten()
    
    # 1. R2 Global
    mse = torch.mean((p_flat - t_flat)**2)
    var = torch.var(t_flat, unbiased=False)
    r2 = 1.0 - (mse / var) if var > 0 else torch.tensor(0.0, device=p_flat.device)
    
    # 2. L1 Skill Score Global (Baseline = médiane théorique = 0)
    mae = torch.mean(torch.abs(p_flat - t_flat))
    mad = torch.mean(torch.abs(t_flat)) 
    l1_score = 1.0 - (mae / mad) if mad > 0 else torch.tensor(0.0, device=p_flat.device)
    
    # 3. Corrélation Globale
    p_mean = torch.mean(p_flat)
    t_mean = torch.mean(t_flat)
    cov = torch.mean((p_flat - p_mean) * (t_flat - t_mean))
    p_std = torch.sqrt(torch.mean((p_flat - p_mean)**2))
    t_std = torch.sqrt(torch.mean((t_flat - t_mean)**2))
    
    corr = cov / (p_std * t_std) if (p_std * t_std) > 1e-8 else torch.tensor(0.0, device=p_flat.device)
    
    return r2.item(), l1_score.item(), corr.item()