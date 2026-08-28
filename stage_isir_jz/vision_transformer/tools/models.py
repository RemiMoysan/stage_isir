import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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


# ============================================================
# ViT utilisé jusqu'à la fin du stage : 
# ============================================================
    
class ViT_Latent_SLP_Multimodal_tunable(nn.Module):
    def __init__(self, 
                 # -- Dimensions et Inputs --
                 sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=3,
                 slp_size=(53, 113), patch_size_slp=(5, 5), in_chans_slp=2,
                 nb_out=10, 
                 # -- Hyperparamètres du Transformer --
                 embed_dim=128, depth=4, num_heads=4, 
                 mlp_ratio=4.0,              # <-- NOUVEAU: Multiplicateur du FeedForward
                 transformer_act="gelu",     # <-- NOUVEAU: Activation interne au Transformer
                 dr=0.1, 
                 # -- Stratégies architecturales --
                 use_lags_attention=False,
                 pool_strategy='cls',        # <-- NOUVEAU: 'cls' (Class Token) ou 'gap' (Global Average Pooling)
                 head_hidden_dim=None,       # <-- NOUVEAU: Flexibilité sur la taille de la couche cachée finale
                 head_act='tanh',# <-- NOUVEAU: Activation de la tête: tanh ou relu (si pas tanh)
                 norm_first=False):  # historiquement on mettait false mais recherche actuel suggère que true est mieux...    
        super().__init__()

        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) doit être divisible par num_heads ({num_heads})"
        self.use_sst_in = in_chans_sst > 0
        self.use_slp_in = in_chans_slp > 0
        self.use_lags_attention = use_lags_attention
        self.pool_strategy = pool_strategy

        if not self.use_sst_in and not self.use_slp_in:
            raise ValueError("Le modèle doit recevoir au moins une source de données (SST ou SLP).")

        # 1. Création des extracteurs de patches
        if self.use_sst_in:
            c_in_sst = 1 if use_lags_attention else in_chans_sst
            # On suppose que PatchEmbedding est défini ailleurs dans ton code
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, c_in_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
            
            if use_lags_attention:
                self.sst_time_embed = nn.Parameter(torch.zeros(1, in_chans_sst, 1, embed_dim))
                
        if self.use_slp_in:
            c_in_slp = 1 if use_lags_attention else in_chans_slp
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, c_in_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))
            
            if use_lags_attention:
                self.slp_time_embed = nn.Parameter(torch.zeros(1, in_chans_slp, 1, embed_dim))

        # 2. Le "Class Token" (uniquement si pool_strategy == 'cls')
        if self.pool_strategy == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dr)

        # 3. L'encodeur Transformer commun
        dim_feedforward = int(embed_dim * mlp_ratio) # Rendu dynamique via mlp_ratio
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dr, 
            dim_feedforward=dim_feedforward, 
            activation=transformer_act, 
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Tête de régression paramétrable
        self.norm = nn.LayerNorm(embed_dim)
        
        h_dim = head_hidden_dim if head_hidden_dim is not None else embed_dim
        final_act = nn.Tanh() if head_act == 'tanh' else nn.ReLU()
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, h_dim), 
            final_act,
            nn.Dropout(p=dr), # Optionnel, souvent utile avant la prédiction finale
            nn.Linear(h_dim, nb_out)
        )

        # 5. Initialisation des poids
        if self.use_sst_in:
            nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.sst_time_embed, std=0.02)
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.slp_time_embed, std=0.02)
                
        if self.pool_strategy == 'cls':
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_sst=None, x_slp=None):
        B = x_sst.shape[0] if self.use_sst_in else x_slp.shape[0]
        
        # --- ROUTE A : ATTENTION SPATIO-TEMPORELLE (ViViT) ---
        if self.use_lags_attention:
            if self.use_sst_in:
                _, T_sst, H_sst, W_sst = x_sst.shape
                x_sst_re = x_sst.reshape(B * T_sst, 1, H_sst, W_sst)
                tokens_sst = self.sst_embed(x_sst_re) 
                tokens_sst = tokens_sst.reshape(B, T_sst, -1, tokens_sst.shape[-1])
                tokens_sst = tokens_sst + self.sst_pos_embed.unsqueeze(1) + self.sst_time_embed
                tokens_sst = tokens_sst.reshape(B, -1, tokens_sst.shape[-1])

            if self.use_slp_in:
                _, T_slp, H_slp, W_slp = x_slp.shape
                x_slp_re = x_slp.reshape(B * T_slp, 1, H_slp, W_slp)
                tokens_slp = self.slp_in_embed(x_slp_re)
                tokens_slp = tokens_slp.reshape(B, T_slp, -1, tokens_slp.shape[-1])
                tokens_slp = tokens_slp + self.slp_in_pos_embed.unsqueeze(1) + self.slp_time_embed
                tokens_slp = tokens_slp.reshape(B, -1, tokens_slp.shape[-1])
                
        # --- ROUTE B : ARCHITECTURE CLASSIQUE (EARLY FUSION) ---
        else:
            if self.use_sst_in:
                tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed
            if self.use_slp_in:
                tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed

        # --- FUSION FINALE DES MODALITÉS ---
        if self.use_sst_in and self.use_slp_in:
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1)
        elif self.use_sst_in:
            tokens = tokens_sst
        elif self.use_slp_in:
            tokens = tokens_slp

        # --- AJOUT CONDITIONNEL DU CLS TOKEN ---
        if self.pool_strategy == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, tokens), dim=1)
        else:
            x = tokens # Si 'gap', on ne rajoute pas de token artificiel

        x = self.pos_drop(x)
        
        # --- PASSAGE DANS LE TRANSFORMER ---
        x = self.transformer(x)
        
        # --- EXTRACTION DU RÉSUMÉ (READOUT) ---
        if self.pool_strategy == 'cls':
            # On prend la représentation du CLS token (position 0)
            representation = x[:, 0]
        else: # 'gap'
            # On moyenne tous les patchs
            representation = x.mean(dim=1) 
            
        cls_out = self.norm(representation)
        
        # --- RÉGRESSION FINALE ---
        output = self.head(cls_out)
        
        return output
    
# ============================================================
# Version classifier (abandonné)
    

class ViT_Classifier_Multimodal(nn.Module):
    def __init__(self, 
                 sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=3,
                 slp_size=(53, 113), patch_size_slp=(5, 10), in_chans_slp=1,
                 num_classes=4, embed_dim=128, depth=4, num_heads=4, dr=0.1,use_lags_attention=False): # <-- nb_out devient num_classes
        super().__init__()
        
        self.use_sst_in = in_chans_sst > 0
        self.use_slp_in = in_chans_slp > 0
        self.use_lags_attention = use_lags_attention
        
        if self.use_sst_in:
            c_in_sst = 1 if use_lags_attention else in_chans_sst
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, c_in_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
            if use_lags_attention:
                self.sst_time_embed = nn.Parameter(torch.zeros(1, in_chans_sst, 1, embed_dim))
            
        if self.use_slp_in:
            c_in_slp = 1 if use_lags_attention else in_chans_slp
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, c_in_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))
            if use_lags_attention:
                self.slp_time_embed = nn.Parameter(torch.zeros(1, in_chans_slp, 1, embed_dim))

        if not self.use_sst_in and not self.use_slp_in:
            raise ValueError("Le modèle doit recevoir au moins une source de données.")
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dr)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- TÊTE DE CLASSIFICATION ---
        self.norm = nn.LayerNorm(embed_dim)
        # Un simple Linear vers num_classes (pas de Tanh, pas de Softmax ici !)
        self.head = nn.Linear(embed_dim, num_classes) 

        # Init
        if self.use_sst_in:
            nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.sst_time_embed, std=0.02)
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.slp_time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_sst = None, x_slp = None):
        B = x_sst.shape[0] if self.use_sst_in else x_slp.shape[0]
        
        if self.use_lags_attention:
            if self.use_sst_in:
                _, T_sst, H_sst, W_sst = x_sst.shape
                x_sst_re = x_sst.reshape(B * T_sst, 1, H_sst, W_sst)
                tokens_sst = self.sst_embed(x_sst_re) 
                tokens_sst = tokens_sst.reshape(B, T_sst, -1, tokens_sst.shape[-1])
                tokens_sst = tokens_sst + self.sst_pos_embed.unsqueeze(1) + self.sst_time_embed
                tokens_sst = tokens_sst.reshape(B, -1, tokens_sst.shape[-1])

            if self.use_slp_in:
                _, T_slp, H_slp, W_slp = x_slp.shape
                x_slp_re = x_slp.reshape(B * T_slp, 1, H_slp, W_slp)
                tokens_slp = self.slp_in_embed(x_slp_re)
                tokens_slp = tokens_slp.reshape(B, T_slp, -1, tokens_slp.shape[-1])
                tokens_slp = tokens_slp + self.slp_in_pos_embed.unsqueeze(1) + self.slp_time_embed
                tokens_slp = tokens_slp.reshape(B, -1, tokens_slp.shape[-1])
        else:
            if self.use_sst_in:
                tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed
            if self.use_slp_in:
                tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed

        if self.use_sst_in and self.use_slp_in:
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1)
        elif self.use_sst_in:
            tokens = tokens_sst
        elif self.use_slp_in:
            tokens = tokens_slp
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1) 

        x = self.pos_drop(x)
        x = self.transformer(x)
        
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out) # Renvoie [Batch, 4]
        
        return logits

# ===============================================================
# Autre ViT pour des codes abandonnées qui ne sont pas sur le Git
# ===============================================================
    

class ViT_Decoded_SLP_Multimodal(nn.Module):
    def __init__(self, sst_size=(85, 360), slp_size=(53, 113), 
                 patch_size_sst=(5, 10), patch_size_slp=(5, 10), 
                 in_chans_sst=3, in_chans_slp=0, out_chans=1, # <-- out_chans gère les quantiles !
                 embed_dim=128, enc_depth=4, dec_depth=4, num_heads=4, dr=0.1, 
                 use_lags_attention=False):
        super().__init__()
        
        self.patch_size_slp = patch_size_slp
        self.target_slp_size = slp_size
        self.out_chans = out_chans
        self.use_lags_attention = use_lags_attention

        # --- ENCODEUR : Branche SST ---
        self.use_sst_in = in_chans_sst > 0
        if self.use_sst_in:
            c_in_sst = 1 if use_lags_attention else in_chans_sst
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, c_in_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
            
            if use_lags_attention:
                self.sst_time_embed = nn.Parameter(torch.zeros(1, in_chans_sst, 1, embed_dim))
        
        # --- ENCODEUR : Branche SLP (Optionnelle) ---
        self.use_slp_in = in_chans_slp > 0
        if self.use_slp_in:
            c_in_slp = 1 if use_lags_attention else in_chans_slp
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, c_in_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))
            
            if use_lags_attention:
                self.slp_time_embed = nn.Parameter(torch.zeros(1, in_chans_slp, 1, embed_dim))

        if not self.use_sst_in and not self.use_slp_in:
            raise ValueError("Le modèle doit recevoir au moins une source de données (SST ou SLP).") 
    
        self.pos_drop = nn.Dropout(p=dr)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        # --- DECODEUR (Génération de la SLP cible) ---
        self.slp_grid_H = math.ceil(slp_size[0] / patch_size_slp[0]) 
        self.slp_grid_W = math.ceil(slp_size[1] / patch_size_slp[1]) 
        num_slp_patches_out = self.slp_grid_H * self.slp_grid_W      
        
        self.slp_queries = nn.Parameter(torch.zeros(1, num_slp_patches_out, embed_dim))
        self.slp_pos_embed = nn.Parameter(torch.zeros(1, num_slp_patches_out, embed_dim))
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # --- RECONSTRUCTION ---
        # Le nombre de valeurs à prédire par patch est multiplié par le nombre de quantiles (out_chans)
        pixels_per_patch = patch_size_slp[0] * patch_size_slp[1] * out_chans
        self.head = nn.Linear(embed_dim, pixels_per_patch)

        # Init
        if self.use_sst_in:
            nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.sst_time_embed, std=0.02)
        nn.init.trunc_normal_(self.slp_queries, std=0.02)
        nn.init.trunc_normal_(self.slp_pos_embed, std=0.02)
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
            if use_lags_attention:
                nn.init.trunc_normal_(self.slp_time_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_sst, x_slp):
        B = x_sst.shape[0] if self.use_sst_in else x_slp.shape[0] 
        
        # --- 1. PHASE ENCODEUR ---
        if self.use_lags_attention:
            if self.use_sst_in:
                _, T_sst, H_sst, W_sst = x_sst.shape
                x_sst_re = x_sst.reshape(B * T_sst, 1, H_sst, W_sst)
                tokens_sst = self.sst_embed(x_sst_re) 
                tokens_sst = tokens_sst.reshape(B, T_sst, -1, tokens_sst.shape[-1])
                tokens_sst = tokens_sst + self.sst_pos_embed.unsqueeze(1) + self.sst_time_embed
                tokens_sst = tokens_sst.reshape(B, -1, tokens_sst.shape[-1])

            if self.use_slp_in:
                _, T_slp, H_slp, W_slp = x_slp.shape
                x_slp_re = x_slp.reshape(B * T_slp, 1, H_slp, W_slp)
                tokens_slp = self.slp_in_embed(x_slp_re)
                tokens_slp = tokens_slp.reshape(B, T_slp, -1, tokens_slp.shape[-1])
                tokens_slp = tokens_slp + self.slp_in_pos_embed.unsqueeze(1) + self.slp_time_embed
                tokens_slp = tokens_slp.reshape(B, -1, tokens_slp.shape[-1])
        else:
            if self.use_sst_in:
                tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed
            if self.use_slp_in:
                tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed

        if self.use_sst_in and self.use_slp_in:
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1)
        elif self.use_sst_in:
            tokens = tokens_sst
        elif self.use_slp_in:
            tokens = tokens_slp
            
        tokens = self.pos_drop(tokens)
        memory = self.encoder(tokens) 

        # --- 2. PHASE DECODEUR ---
        queries = self.slp_queries.expand(B, -1, -1) + self.slp_pos_embed
        dec_out = self.decoder(tgt=queries, memory=memory) 

        # --- 3. RECONSTRUCTION ---
        out_pixels = self.head(dec_out) 
        
        H_grid, W_grid = self.slp_grid_H, self.slp_grid_W
        pH, pW = self.patch_size_slp
        
        # Réarrangement pour séparer les canaux de quantiles et les dimensions spatiales
        out_map = out_pixels.view(B, H_grid, W_grid, self.out_chans, pH, pW)
        out_map = out_map.permute(0, 3, 1, 4, 2, 5).contiguous()
        padded_img = out_map.view(B, self.out_chans, H_grid * pH, W_grid * pW)
        
        # Découpage final aux dimensions cibles exactes -> shape: (B, out_chans, H, W)
        final_slp = padded_img[:, :, :self.target_slp_size[0], :self.target_slp_size[1]]
        return final_slp.contiguous()


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

def correlation_loss(y_pred, y_true):
    """Loss de corrélation (1 - corrélation moyenne). En particulier marche même si on est pas dans le cas 1D."""
    corr = pearson_correlation(y_pred, y_true, dim=0)
    return 1.0 - corr.mean()

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

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD