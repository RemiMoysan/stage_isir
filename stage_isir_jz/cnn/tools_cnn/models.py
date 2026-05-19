import torch
import torch.nn as nn
import torch.nn.functional as F

# version de gemini : augmentation du nombre de channel jusqu'à un certain embed dim avec moyennage sur la carte, qui en une couche finale devient nb_out.

class LonCircularConv2d(nn.Module):
    """
    Convolution 2D avec un padding circulaire uniquement sur la longitude (largeur/axe W).
    La latitude (hauteur/axe H) conserve un zero-padding standard pour accepter les effets de bord.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding_lat=1):
        super().__init__()
        self.pad_lon = kernel_size // 2  # Fonctionne pour les kernels impairs (ex: 3, 5)
        # La Conv2d classique ne gère que le padding de la latitude (H). 
        # La longitude sera gérée manuellement dans le forward.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=(padding_lat, 0))

    def forward(self, x):
        # F.pad sur un tenseur 4D (B, C, H, W) : (pad_left, pad_right, pad_top, pad_bottom)
        # On pad circulairement la dimension W (longitudes)
        if self.pad_lon > 0:
            x = F.pad(x, (self.pad_lon, self.pad_lon, 0, 0), mode='circular')
        return self.conv(x)


class CNN_Latent_SLP_Multimodal0(nn.Module):
    def __init__(self, 
                 sst_size=(85, 360), in_chans_sst=3,
                 slp_size=(53, 113), in_chans_slp=2,  # <-- Lags
                 nb_out=10, n_feat=128, depth=4, dr=0.1):
        """
        Adaptation CNN. Les paramètres patch_size et num_heads sont conservés dans la signature 
        pour ne pas casser l'instanciation, mais ne sont plus utilisés. 
        'depth' dicte maintenant le nombre de couches de convolution.

        Version un peu bizarre qui augmente le nombre de channel jusqu'à un embed_dim (n_feat) jusqu'à 1 pixel puis concatène pour les lags / entre sst et slp pour obtenir en gros l'embedding final.
        PAS UTIlISE. 
        """
        super().__init__()
        
        embed_dim = n_feat

        self.use_sst_in = in_chans_sst > 0
        self.use_slp_in = in_chans_slp > 0

        if not self.use_sst_in and not self.use_slp_in:
            raise ValueError("Le modèle doit recevoir au moins une source de données (SST ou SLP).")
        
        # Fonction utilitaire pour générer un encodeur CNN profond
        def make_cnn_branch(in_channels):
            layers = []
            current_channels = in_channels
            
            # Augmentation progressive des canaux jusqu'à embed_dim
            for i in range(depth):
                # On s'assure d'atteindre embed_dim sur la dernière couche de conv
                out_channels = max(16, embed_dim // (2 ** (depth - i - 1)))
                if i == depth - 1:
                    out_channels = embed_dim
                    
                layers.append(LonCircularConv2d(current_channels, out_channels, kernel_size=3, padding_lat=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.GELU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # Réduction spatiale
                
                current_channels = out_channels
                
            # Global Average Pooling pour obtenir un vecteur 1D indépendant de la taille d'entrée
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        # 1. Création des DEUX extracteurs CNN
        if self.use_sst_in:
            self.sst_encoder = make_cnn_branch(in_chans_sst)
        if self.use_slp_in:
            self.slp_encoder = make_cnn_branch(in_chans_slp)

        # 2. Préparation de la dimension pour la tête de réseau (Late Fusion)
        head_in_features = 0
        if self.use_sst_in: head_in_features += embed_dim
        if self.use_slp_in: head_in_features += embed_dim

        self.norm = nn.LayerNorm(head_in_features)
        self.dropout = nn.Dropout(p=dr)

        # 3. Tête de régression (Mapping latent / quantiles)
        self.head = nn.Sequential(
            nn.Linear(head_in_features, embed_dim), 
            nn.Tanh(),
            nn.Linear(embed_dim, nb_out)
        )

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Initialisation recommandée pour les CNN (He/Kaiming)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_sst, x_slp):
        features = []
        
        # 1. Extraction spatiale indépendante pour chaque modalité
        if self.use_sst_in:
            # (B, C, 85, 360) -> CNN -> AdaptivePool -> (B, embed_dim)
            sst_feat = self.sst_encoder(x_sst)
            features.append(sst_feat)
            
        if self.use_slp_in:
            # (B, C, 53, 113) -> CNN -> AdaptivePool -> (B, embed_dim)
            slp_feat = self.slp_encoder(x_slp)
            features.append(slp_feat)
            
        # 2. FUSION (Late Fusion) : Concaténation des vecteurs latents
        x = torch.cat(features, dim=1) # -> (B, head_in_features)
        
        # 3. Normalisation et Régression
        x = self.norm(x)
        x = self.dropout(x)
        output = self.head(x)
        
        return output
    

# version plus proche de clara, plus simple et qui celle qu'on utilise. 

class conv_geo(nn.Module):
    def __init__(self, in_ch, out_ch, kx, ky):
        super(conv_geo, self).__init__()
        self.conv_nopad = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, [kx, ky], padding=[0, 0], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Tanh()
        )
        self.kx = kx

    def forward(self, x):
        kx = self.kx
        x = F.pad(x, [kx // 2, kx // 2, 0, 0], mode='circular')
        x = self.conv_nopad(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch, drop, kx, ky, mpx, mpy):
        super(down_layer, self).__init__()
        self.convg = conv_geo(in_ch, out_ch, kx, ky)
        self.drop = nn.Dropout(p=drop)
        self.pool = nn.MaxPool2d([mpx, mpy])

    def forward(self, x):
        x = self.convg(x)
        x = self.drop(x)
        x = self.pool(x)
        return x

class CNN_Latent_SLP_Multimodal1(nn.Module):
    def __init__(self, dr=0.1, nb_out=10, 
                 in_chans_sst=3, in_chans_slp=2, 
                 n_feat=8):
        super().__init__()
        
        self.use_sst = in_chans_sst > 0
        self.use_slp = in_chans_slp > 0
        
        if not self.use_sst and not self.use_slp:
            raise ValueError("Il faut au moins SST ou SLP.")

        # -- Branche SST --
        if self.use_sst:
            self.sst_branch = nn.Sequential(
                down_layer(in_chans_sst, n_feat, dr, 3, 5, 2, 3),
                down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
            )
            
        # -- Branche SLP --
        if self.use_slp:
            self.slp_branch = nn.Sequential(
                # On utilise des kernels plus petits (3,3 partout) 
                # et on pool moins agressivement sur la longitude
                down_layer(in_chans_slp, n_feat, dr, 3, 3, 2, 2), 
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1), # Plus de pooling ici
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
            )

        # On a besoin de savoir la taille exacte après le flatten.
        # Plutôt que de coder "1003" en dur (qui pourrait planter si SLP n'a pas la même taille d'entrée),
        # on utilise un "LazyLinear" qui calculera tout seul la bonne taille lors du premier forward.
        self.regr = nn.Sequential(
            nn.Dropout(p=dr), 
            nn.LazyLinear(n_feat), # S'adapte automatiquement à la taille du tenseur aplati
            nn.Tanh(),
            nn.Linear(n_feat, nb_out)
        )

    def forward(self, x_sst=None, x_slp=None):
        features = []
        
        if self.use_sst and x_sst is not None:
            out_sst = self.sst_branch(x_sst)
            out_sst = torch.flatten(out_sst, 1) # Aplatit toutes les dimensions spatiales
            features.append(out_sst)
            
        if self.use_slp and x_slp is not None:
            out_slp = self.slp_branch(x_slp)
            out_slp = torch.flatten(out_slp, 1)
            features.append(out_slp)
            
        # Concaténation des deux vecteurs aplatis (si les deux sont utilisés)
        # Ex: si SST sort un vecteur de taille 32000 et SLP de taille 16000 -> x fera 48000
        x = torch.cat(features, dim=1)
        
        output = self.regr(x)
        return output
    

# ============================================================
# MODÈLE CNN POUR LA CLASSIFICATION MULTIMODALE
# ============================================================
class CNN_Classifier_Multimodal(nn.Module):
    def __init__(self, dr=0.1, num_classes=4, 
                 in_chans_sst=3, in_chans_slp=1, 
                 n_feat=8):
        super().__init__()
        
        self.use_sst = in_chans_sst > 0
        self.use_slp = in_chans_slp > 0
        
        if not self.use_sst and not self.use_slp:
            raise ValueError("Il faut au moins SST ou SLP.")

        # -- Branche SST --
        if self.use_sst:
            self.sst_branch = nn.Sequential(
                down_layer(in_chans_sst, n_feat, dr, 3, 5, 2, 3),
                down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
            )
            
        # -- Branche SLP --
        if self.use_slp:
            self.slp_branch = nn.Sequential(
                # Kernels et poolings adaptés à la plus petite résolution de la SLP
                down_layer(in_chans_slp, n_feat, dr, 3, 3, 2, 2), 
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1),
                down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
            )

        # --- TÊTE DE CLASSIFICATION ---
        # J'ai ajouté une couche cachée (Linear -> ReLU) avant la projection finale
        # pour permettre au réseau de bien mélanger/croiser les features SST et SLP
        # avant de prendre sa décision de classe finale.
        hidden_dim = n_feat * 4 
        
        self.head = nn.Sequential(
            nn.Dropout(p=dr), 
            nn.LazyLinear(hidden_dim), # S'adapte à la taille des vecteurs concaténés
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes) # Sortie brute (logits)
        )

    def forward(self, x_sst=None, x_slp=None):
        features = []
        
        if self.use_sst and x_sst is not None:
            out_sst = self.sst_branch(x_sst)
            out_sst = torch.flatten(out_sst, 1)
            features.append(out_sst)
            
        if self.use_slp and x_slp is not None:
            out_slp = self.slp_branch(x_slp)
            out_slp = torch.flatten(out_slp, 1)
            features.append(out_slp)
            
        # Concaténation des vecteurs (Late Fusion)
        x = torch.cat(features, dim=1)
        
        # Décision de classification
        logits = self.head(x) # Renvoie un tenseur de taille [Batch, num_classes]
        
        return logits