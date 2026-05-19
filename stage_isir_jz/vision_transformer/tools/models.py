import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# A termes il faudrait tous les rendres lag flexibles, et avoir juste l'encoder (token cls taille 128) et le transformer encoder decoder 

# Remarque : à l'heure actuelle, pas de self attention entre les différents lags, car on mélange tout avant 
# Alors que pour slp et sst, on concatène dans la liste de patch : clairement de la self-attention (mais est ce que c'est plus logique que de faire de la cross attention), cela veut dire que première compo patch slp a la même info que première compo patch sst. 

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

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


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
# 1. Encoder-Decoder ViT pour la traduction SST -> SLP : on essaie de prédire toute la carte SLP d'un coup
# ============================================================
class ViT_SST_to_SLP(nn.Module):
    def __init__(self, sst_size=(85, 360), slp_size=(53, 113), patch_size=(5, 10), 
                 in_chans=3, embed_dim=128, enc_depth=4, dec_depth=4, num_heads=4, dr=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.target_slp_size = slp_size # (53, 113)
        
        # --- ENCODEUR (Lecture de la SST) ---
        self.sst_embed = PatchEmbedding(sst_size, patch_size, in_chans, embed_dim)
        num_sst_patches = (sst_size[0] // patch_size[0]) * (sst_size[1] // patch_size[1]) # 17 * 36 = 612
        
        # Positions SST (Plus de CLS token, on fait du pixel-to-pixel)
        self.sst_pos_embed = nn.Parameter(torch.zeros(1, num_sst_patches, embed_dim))
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        # --- DECODEUR (Génération de la SLP) ---
        # Astuce de dimension : On arrondit au multiple supérieur du patch_size
        # 53 -> 55 (11 patches) | 113 -> 120 (12 patches)
        self.slp_grid_H = math.ceil(slp_size[0] / patch_size[0]) # 11
        self.slp_grid_W = math.ceil(slp_size[1] / patch_size[1]) # 12
        num_slp_patches = self.slp_grid_H * self.slp_grid_W      # 132 patches à prédire
        
        # Les "Learnable Queries" (Les dessinateurs)
        self.slp_queries = nn.Parameter(torch.zeros(1, num_slp_patches, embed_dim))
        # Les positions des requêtes (Pour qu'elles sachent où elles dessinent sur la carte)
        self.slp_pos_embed = nn.Parameter(torch.zeros(1, num_slp_patches, embed_dim))
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # --- RECONSTRUCTION (Unpatchify) ---
        pixels_per_patch = patch_size[0] * patch_size[1] # 5 * 10 = 50 pixels
        self.head = nn.Linear(embed_dim, pixels_per_patch)

        # Initialisation propre
        nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.slp_queries, std=0.02)
        nn.init.trunc_normal_(self.slp_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # ==========================================
        # 1. PHASE ENCODEUR (SST -> Mémoire)
        # ==========================================
        # On découpe et on ajoute les coordonnées géographiques
        sst_tokens = self.sst_embed(x) + self.sst_pos_embed 
        # L'encodeur réfléchit aux corrélations au sein de l'océan
        memory = self.encoder(sst_tokens) 
        # memory shape: (Batch, 612, 128)

        # ==========================================
        # 2. PHASE DECODEUR (Mémoire + Requêtes -> SLP Latente)
        # ==========================================
        # On prépare nos 132 requêtes vides et on leur donne leurs coordonnées spatiales
        queries = self.slp_queries.expand(B, -1, -1) + self.slp_pos_embed
        
        # Le dialogue : Les requêtes (tgt) interrogent la SST (memory)
        dec_out = self.decoder(tgt=queries, memory=memory) 
        # dec_out shape: (Batch, 132, 128)

        # ==========================================
        # 3. RECONSTRUCTION ET CROPPING
        # ==========================================
        # On transforme l'embedding abstrait en 50 pixels de pression atmosphérique
        out_pixels = self.head(dec_out) # -> (Batch, 132, 50)
        
        # Magie du reshape pour reformer l'image 2D à partir de la liste de patches
        # 1. On sépare la grille de patchs et les pixels à l'intérieur
        H_grid, W_grid = self.slp_grid_H, self.slp_grid_W
        pH, pW = self.patch_size
        out_map = out_pixels.view(B, H_grid, W_grid, 1, pH, pW)
        
        # 2. On réordonne les dimensions : (Batch, Channel, H_grid, pH, W_grid, pW)
        out_map = out_map.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # 3. On fusionne pour avoir l'image finale paddée : (Batch, 1, 55, 120)
        padded_img = out_map.view(B, 1, H_grid * pH, W_grid * pW)
        
        # 4. Le Crop : On coupe la bordure droite et basse pour retomber sur (53, 113)
        final_slp = padded_img[:, :, :self.target_slp_size[0], :self.target_slp_size[1]]
        
        return final_slp
    

# 1.bis version multimodale du précédent; peut etre que ça vaut le coup d'ajouter un type embedding entre slp et sst mais les positional embeddins pour les deux sont déjà indépendant donc je pense que c'est okay. 

class ViT_Decoded_SLP_Multimodal(nn.Module):
    def __init__(self, sst_size=(85, 360), slp_size=(53, 113), 
                 patch_size_sst=(5, 10), patch_size_slp=(5, 10), 
                 in_chans_sst=3, in_chans_slp=0, # <-- Accepte maintenant les canaux SLP
                 embed_dim=128, enc_depth=4, dec_depth=4, num_heads=4, dr=0.1):
        super().__init__()
        
        self.patch_size_slp = patch_size_slp
        self.target_slp_size = slp_size

    
        # --- ENCODEUR : Branche SST ---
        self.use_sst_in = in_chans_sst > 0
        if self.use_sst_in:
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, in_chans_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
        
        # --- ENCODEUR : Branche SLP (Optionnelle) ---
        self.use_slp_in = in_chans_slp > 0
        if self.use_slp_in:
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, in_chans_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))

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
        # plutôt que le même positional embedding pour les deux...
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # --- RECONSTRUCTION ---
        pixels_per_patch = patch_size_slp[0] * patch_size_slp[1] 
        self.head = nn.Linear(embed_dim, pixels_per_patch)

        # Init
        if self.use_sst_in:
            nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.slp_queries, std=0.02)
        nn.init.trunc_normal_(self.slp_pos_embed, std=0.02)
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
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
        
        tokens = None

        if self.use_sst_in and self.use_slp_in:
            tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed # -> (B, num_patches_sst, embed_dim)
            tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed # -> (B, num_patches_slp, embed_dim)
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1) # -> (B, num_patches_total, embed_dim)
        elif self.use_sst_in:
            tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed
            tokens = tokens_sst
        elif self.use_slp_in:
            tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed
            tokens = tokens_slp
            
        tokens = self.pos_drop(tokens)
        memory = self.encoder(tokens) 

        # 2. PHASE DECODEUR
        queries = self.slp_queries.expand(B, -1, -1) + self.slp_pos_embed
        dec_out = self.decoder(tgt=queries, memory=memory) 

        # 3. RECONSTRUCTION
        out_pixels = self.head(dec_out) 
        
        H_grid, W_grid = self.slp_grid_H, self.slp_grid_W
        pH, pW = self.patch_size_slp
        out_map = out_pixels.view(B, H_grid, W_grid, 1, pH, pW)
        out_map = out_map.permute(0, 3, 1, 4, 2, 5).contiguous()
        padded_img = out_map.view(B, 1, H_grid * pH, W_grid * pW)
        
        final_slp = padded_img[:, :, :self.target_slp_size[0], :self.target_slp_size[1]]
        return final_slp

# ============================================================
# 2. ViT basé sur le code de Clara pour prédire PC1
# Redondant avec le code suivant mais des visualisations différentes : calcul de corrélation mensuelle sur les membres de tests.
# Elagué pour l'instant. 
# ============================================================

# ============================================================
# 3. ViT Custom lags SST, SLP vers PC1
# C'est le VIT encoder le plus flexible, je pense qu'il englobe tous les autres ViT encoder.
# ============================================================
class ViT_Multimodal(nn.Module):
    """
    Vision Transformer adapté à la prédiction à partir de DEUX sources 
    (SST et SLP) avec des tailles et des lags différents.
    On choisit nb_out = 10 si on veut prédire les quantiles de la PC1 auquel cas on utilise la quantile_loss
    """
    def __init__(self, 
                 sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=3,
                 slp_size=(53, 113), patch_size_slp=(5, 5), in_chans_slp=2,  # <-- Tu peux changer les lags ici
                 nb_out=10, embed_dim=128, depth=4, num_heads=4, dr=0.1):
        super().__init__()
        
        # 1. Création des DEUX extracteurs de patches
        self.patch_embed_sst = PatchEmbedding(sst_size, patch_size_sst, in_chans_sst, embed_dim)
        self.patch_embed_slp = PatchEmbedding(slp_size, patch_size_slp, in_chans_slp, embed_dim)
        
        # Le nombre total de "mots" dans notre phrase est l'addition des deux
        num_patches = self.patch_embed_sst.num_patches + self.patch_embed_slp.num_patches

        # 2. Le "Class Token" et les positions
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dr)

        # 3. L'encodeur Transformer commun (Le cœur qui va mélanger Océan et Atmosphère)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dr, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Tête de régression (Output vers tes quantiles ou ton espace latent)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), # JE NE VOIS PAS TROP L'INTERET de faire en deux fois surtout qu'on pourrait avoir nb_out \approx 128...
            nn.Tanh(),
            nn.Linear(embed_dim // 2, nb_out)
        )

        # Initialisation propre des poids
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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

    def forward(self, x_sst, x_slp):
        """
        Le modèle prend maintenant deux tenseurs en entrée.
        """
        B = x_sst.shape[0] # Taille du Batch
        
        # 1. Découpage en patches de chaque côté
        tokens_sst = self.patch_embed_sst(x_sst) # -> (B, num_patches_sst, embed_dim)
        tokens_slp = self.patch_embed_slp(x_slp) # -> (B, num_patches_slp, embed_dim)
        
        # 2. FUSION (Early Fusion) : on colle les patchs SLP à la suite des patchs SST
        x = torch.cat((tokens_sst, tokens_slp), dim=1) # -> (B, num_patches_total, embed_dim)
        
        # 3. Ajout du token de classification
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (B, 1 + num_patches_total, embed_dim)
        
        # 4. Ajout de l'information de position globale
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 5. Passage dans le Transformer (qui analyse tout ensemble)
        x = self.transformer(x)
        
        # 6. Extraction du résumé global et Régression
        cls_out = self.norm(x[:, 0])
        output = self.head(cls_out)
        
        return output
    
# ============================================================
# 4. ViT de SST vers embedding de SLP, redondant en gros il n'y a pas l'étape de division par 2 de la dimension à la toute fin

class ViT_Latent(nn.Module):
    def __init__(self, img_size=(85, 360), patch_size=(5, 10), in_chans=3, 
                 latent_dim=128, embed_dim=64, depth=4, num_heads=4, dr=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dr)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dr, 
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Tête de régression vers l'espace latent
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        
        cls_out = self.norm(x[:, 0])
        latent_pred = self.head(cls_out)
        return latent_pred
    
class ViT_Latent_SLP_Multimodal(nn.Module):

    def __init__(self, 
                 sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=3,
                 slp_size=(53, 113), patch_size_slp=(5, 5), in_chans_slp=2,  # <-- Tu peux changer les lags ici
                 nb_out=10, embed_dim=128, depth=4, num_heads=4, dr=0.1):
        super().__init__()
        
        self.use_sst_in = in_chans_sst > 0
        self.use_slp_in = in_chans_slp > 0
        # 1. Création des DEUX extracteurs de patches
        if self.use_sst_in:
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, in_chans_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
        if self.use_slp_in:
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, in_chans_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))

        if not self.use_sst_in and not self.use_slp_in:
            raise ValueError("Le modèle doit recevoir au moins une source de données (SST ou SLP).")
        
        # 2. Le "Class Token" et les positions
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dr)

        # 3. L'encodeur Transformer commun (Le cœur qui va mélanger Océan et Atmosphère)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dr, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Tête de régression (Output vers tes quantiles ou ton espace latent)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.Tanh(),
            nn.Linear(embed_dim, nb_out)
        )

        # Init
        if self.use_sst_in:
            nn.init.trunc_normal_(self.sst_pos_embed, std=0.02)
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
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

    def forward(self, x_sst, x_slp):
        """
        Le modèle prend maintenant deux tenseurs en entrée.
        """
        B = x_sst.shape[0] if self.use_sst_in else x_slp.shape[0]
        
        # 2. FUSION (Early Fusion) : on colle les patchs SLP à la suite des patchs SST
        if self.use_sst_in and self.use_slp_in:
            tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed # -> (B, num_patches_sst, embed_dim)
            tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed # -> (B, num_patches_slp, embed_dim)
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1) # -> (B, num_patches_total, embed_dim)
        elif self.use_sst_in:
            tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed
            tokens = tokens_sst
        elif self.use_slp_in:
            tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed
            tokens = tokens_slp
        # 3. Ajout du token de classification
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1) # -> (B, 1 + num_patches_total, embed_dim)

        x = self.pos_drop(x)
        
        # 5. Passage dans le Transformer (qui analyse tout ensemble)
        x = self.transformer(x)
        
        # 6. Extraction du résumé global et Régression
        cls_out = self.norm(x[:, 0])
        output = self.head(cls_out)
        
        return output
    
# ============================================================
# La version classification du ViT multimodal, pour comparer avec les modèles de classification plus classiques.
    

class ViT_Classifier_Multimodal(nn.Module):
    def __init__(self, 
                 sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=3,
                 slp_size=(53, 113), patch_size_slp=(5, 10), in_chans_slp=1,
                 num_classes=4, embed_dim=128, depth=4, num_heads=4, dr=0.1): # <-- nb_out devient num_classes
        super().__init__()
        
        self.use_sst_in = in_chans_sst > 0
        self.use_slp_in = in_chans_slp > 0
        
        if self.use_sst_in:
            self.sst_embed = PatchEmbedding(sst_size, patch_size_sst, in_chans_sst, embed_dim)
            self.sst_pos_embed = nn.Parameter(torch.zeros(1, self.sst_embed.num_patches, embed_dim))
        if self.use_slp_in:
            self.slp_in_embed = PatchEmbedding(slp_size, patch_size_slp, in_chans_slp, embed_dim)
            self.slp_in_pos_embed = nn.Parameter(torch.zeros(1, self.slp_in_embed.num_patches, embed_dim))

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
        if self.use_slp_in:
            nn.init.trunc_normal_(self.slp_in_pos_embed, std=0.02)
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

    def forward(self, x_sst, x_slp):
        B = x_sst.shape[0] if self.use_sst_in else x_slp.shape[0]
        
        if self.use_sst_in and self.use_slp_in:
            tokens_sst = self.sst_embed(x_sst) + self.sst_pos_embed 
            tokens_slp = self.slp_in_embed(x_slp) + self.slp_in_pos_embed 
            tokens = torch.cat((tokens_sst, tokens_slp), dim=1) 
        elif self.use_sst_in:
            tokens = self.sst_embed(x_sst) + self.sst_pos_embed
        elif self.use_slp_in:
            tokens = self.slp_in_embed(x_slp) + self.slp_in_pos_embed
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1) 

        x = self.pos_drop(x)
        x = self.transformer(x)
        
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out) # Renvoie [Batch, 4]
        
        return logits
    
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