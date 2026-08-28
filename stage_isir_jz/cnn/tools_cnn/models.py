import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_geo(nn.Module):
    def __init__(self, in_ch, out_ch, kx, ky,activation='tanh'):
        super(conv_geo, self).__init__()
        act_layer = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.conv_nopad = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, [kx, ky], padding=[0, 0], bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer
        )
        # un biais serait inutile puisque la BN va le supprimer.
        self.kx = kx
        self.ky = ky

    def forward(self, x):
        ky = self.ky
        x = F.pad(x, [ky // 2, ky // 2, 0, 0], mode='circular') # ky et pas kx car c'est sur la longitude ie deuxième compo 
        x = self.conv_nopad(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch, drop, kx, ky, mpx, mpy, pool_type='max', activation='tanh'):
        super(down_layer, self).__init__()
        self.convg = conv_geo(in_ch, out_ch, kx, ky, activation=activation)
        self.drop = nn.Dropout2d(p=drop) # drop out 2d (ie par feature) plutôt que dropout classique. 
        if pool_type == 'max':
            self.pool = nn.MaxPool2d([mpx, mpy])
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d([mpx, mpy])
    def forward(self, x):
        x = self.convg(x)
        x = self.drop(x)
        x = self.pool(x)
        return x

        
class CNN_Latent_SLP_Multimodal1_tunable(nn.Module):
    """Input SST : n_lags x 85 x 360, Input SLP : n_lags x 53 x 113
    Standart qu'on ne remet pas en question :
    Couche conv stride de 1, pas besoin de padding ici puisque circulaire. 3 fois 3 une fois qu'on s'est ramené à quelque chose d'à peu près carré (comme Clara).
    Pools de 2 fois 2 SOTA de Vision après avoir fait un pool typiquement 3 fois 2 qui rend carré ou éventuellement 2x2 ou 1x1

    Choix : type de kernel et de pool pour la première couche non carré, lazylinear ou gap à la toute fin, augmentation éventuelle géo du nombre de feature, diminition progressive éventuelle des pools
    """
    def __init__(self, dr_conv=0.1, dr_fc=0.5,fc_dim=128,nb_out=10, 
                 in_chans_sst=3, in_chans_slp=2, 
                 n_feat=8, early_fusion_sst=True,
                 depth=3, filter_mult=1, 
                 sst_kx=3, sst_ky=5,pool_type='max',activation='tanh',sst_pool_x=2,sst_pool_y=3,pool_strategy = 'progressive',use_gap=False): 
        super().__init__()
        
        self.use_sst = in_chans_sst > 0
        self.use_slp = in_chans_slp > 0
        self.early_fusion_sst = early_fusion_sst
        self.use_gap = use_gap
        
        if not self.use_sst and not self.use_slp:
            raise ValueError("Il faut au moins SST ou SLP.")

        # -- Branche SST --
        if self.use_sst:
            if self.early_fusion_sst:
                self.sst_branch = self._build_branch(
                    in_channels=in_chans_sst, depth=depth, base_feat=n_feat, 
                    filter_mult=filter_mult, kx_base=sst_kx, ky_base=sst_ky, dr=dr_conv, pool_type=pool_type, activation=activation, pool_x_base=sst_pool_x, pool_y_base=sst_pool_y, pool_strategy=pool_strategy
                )
            else:
                self.sst_branches = nn.ModuleList([
                    self._build_branch(
                        in_channels=1, depth=depth, base_feat=n_feat, 
                        filter_mult=filter_mult, kx_base=sst_kx, ky_base=sst_ky, dr=dr_conv, pool_type=pool_type, activation=activation, pool_x_base=sst_pool_x, pool_y_base=sst_pool_y, pool_strategy=pool_strategy
                    ) for _ in range(in_chans_sst)
                ])
            
        # -- Branche SLP --
        if self.use_slp:
            self.slp_branch = self._build_branch(
                in_channels=in_chans_slp, depth=depth, base_feat=n_feat, 
                filter_mult=filter_mult, kx_base=3, ky_base=3, dr=dr_conv, pool_type=pool_type, activation=activation, pool_x_base=2, pool_y_base=2, pool_strategy=pool_strategy # SLP est déjà plus carrée 3x3 et 2x2
            )

        # Le LazyLinear est magique ici : peu importe la profondeur (depth) 
        # ou le multiplicateur de filtres, il calculera l'aplatissement tout seul.
        final_act = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.regr = nn.Sequential(
            nn.Dropout(p=dr_fc), 
            nn.LazyLinear(fc_dim), # Estimation pour le nom, LazyLinear fait le reste
            final_act,
            nn.Linear(fc_dim, nb_out)
        )

    def _build_branch(self, in_channels, depth, base_feat, filter_mult, kx_base, ky_base, dr, pool_type, activation, pool_x_base, pool_y_base, pool_strategy):
        """
        Générateur dynamique de couches convolutives.
        Permet de varier la profondeur et d'augmenter le nombre de filtres de façon sécurisée.
        """
        layers = []
        current_in = in_channels
        current_out = base_feat
        
        for i in range(depth):
            # Le premier layer utilise les kernels personnalisés (ex: 3x5 pour SST)
            # Les suivants utilisent du 3x3 classique pour extraire des features abstraites
            kx = kx_base if i == 0 else 3
            ky = ky_base if i == 0 else 3
            
            # SÉCURITÉ SPATIALE : On pool (divise la taille) uniquement sur les 2 premières couches.
            # Au-delà, on utilise un pool de 1x1 (qui ne fait rien) pour pouvoir empiler 
            # 4, 5 ou 6 couches sans réduire la carte à une dimension négative.
            # Application de la stratégie choisie par Optuna
            if pool_strategy == 'progressive':
                mpx = pool_x_base if i == 0 else (2 if i == 1 else 1)
                mpy = pool_y_base if i == 0 else (2 if i == 1 else 1)
            else: # 'standard'
                mpx = pool_x_base if i == 0 else 2
                mpy = pool_y_base if i == 0 else 2
            
            layers.append(down_layer(current_in, current_out, dr, kx, ky, mpx, mpy, pool_type=pool_type, activation=activation))
            
            # On prépare les dimensions pour la couche suivante
            current_in = current_out
            current_out = int(current_out * filter_mult)
            
        return nn.Sequential(*layers)

    def forward(self, x_sst=None, x_slp=None):
        features = []
        
        if self.use_sst and x_sst is not None:
            if self.early_fusion_sst:
                out_sst = self.sst_branch(x_sst)
                if self.use_gap:
                    out_sst = F.adaptive_avg_pool2d(out_sst, (1, 1))  # Global Average Pooling
                out_sst = torch.flatten(out_sst, 1)
                features.append(out_sst)
            else:
                for i in range(x_sst.size(1)):
                    x_lag = x_sst[:, i:i+1, :, :] 
                    out_lag = self.sst_branches[i](x_lag)
                    if self.use_gap:
                        out_lag = F.adaptive_avg_pool2d(out_lag, (1, 1))  # Global Average Pooling
                    out_lag = torch.flatten(out_lag, 1)
                    features.append(out_lag)
            
        if self.use_slp and x_slp is not None:
            out_slp = self.slp_branch(x_slp)
            if self.use_gap:
                out_slp = F.adaptive_avg_pool2d(out_slp, (1, 1))  # Global Average Pooling
            out_slp = torch.flatten(out_slp, 1)
            features.append(out_slp)
            
        x = torch.cat(features, dim=1)
        output = self.regr(x)
        return output
    
# =============================================================================================
# MODÈLE CNN POUR LA REGRESSION NON TUNABLE (abandonné pour CNN_Latent_SLP_Multimodal1_tunable)
# =============================================================================================

class CNN_Latent_SLP_Multimodal1(nn.Module):
    def __init__(self, dr=0.1, nb_out=10, 
                 in_chans_sst=3, in_chans_slp=2, 
                 n_feat=8,early_fusion_sst = True):
        super().__init__()
        
        self.use_sst = in_chans_sst > 0
        self.use_slp = in_chans_slp > 0
        self.early_fusion_sst = early_fusion_sst
        
        if not self.use_sst and not self.use_slp:
            raise ValueError("Il faut au moins SST ou SLP.")

        # -- Branche SST --
        if self.use_sst:
            if self.early_fusion_sst:
                # Comportement original : 1 seule branche qui prend tous les lags
                self.sst_branch = nn.Sequential(
                    down_layer(in_chans_sst, n_feat, dr, 3, 5, 2, 3),
                    down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                    down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
                )
            else:
                # Late fusion : on crée une liste de branches indépendantes
                # Chaque branche prend 1 seul canal en entrée (1 lag)
                self.sst_branches = nn.ModuleList([
                    nn.Sequential(
                        down_layer(1, n_feat, dr, 3, 5, 2, 3), # 1 canal en entrée
                        down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                        down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
                    ) for _ in range(in_chans_sst)
                ])
            
        # -- Branche SLP --
        if self.use_slp:
            self.slp_branch = nn.Sequential(
                # On utilise des kernels plus petits (3,3 partout) 
                # et on pool moins agressivement sur la longitude
                down_layer(in_chans_slp, n_feat, dr, 3, 3, 2, 2), 
                down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
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
            if self.early_fusion_sst:
                out_sst = self.sst_branch(x_sst)
                out_sst = torch.flatten(out_sst, 1)
                features.append(out_sst)
            else:
                # On boucle sur la dimension des canaux (les lags)
                for i in range(x_sst.size(1)):
                    # On extrait le lag i tout en gardant la dimension du canal (B, 1, H, W)
                    # L'utilisation de i:i+1 permet de ne pas perdre la dimension "channel"
                    x_lag = x_sst[:, i:i+1, :, :] 
                    
                    # On le passe dans sa branche dédiée
                    out_lag = self.sst_branches[i](x_lag)
                    out_lag = torch.flatten(out_lag, 1)
                    features.append(out_lag)
            
        if self.use_slp and x_slp is not None:
            out_slp = self.slp_branch(x_slp)
            out_slp = torch.flatten(out_slp, 1)
            features.append(out_slp)
            
        # Concaténation des deux vecteurs aplatis (si les deux sont utilisés)
        # Ex: si SST sort un vecteur de taille 32000 et SLP de taille 16000 -> x fera 48000
        x = torch.cat(features, dim=1)
        
        output = self.regr(x)
        return output

# ==========================================================================================
# MODÈLE CNN POUR LA CLASSIFICATION (abandonné progressivement pendant le stage)
# ==========================================================================================
class CNN_Classifier_Multimodal(nn.Module):
    def __init__(self, dr=0.1, num_classes=4, 
                 in_chans_sst=3, in_chans_slp=1, 
                 n_feat=8, early_fusion_sst=True):
        super().__init__()
        
        self.use_sst = in_chans_sst > 0
        self.use_slp = in_chans_slp > 0
        self.early_fusion_sst = early_fusion_sst
        
        if not self.use_sst and not self.use_slp:
            raise ValueError("Il faut au moins SST ou SLP.")

        # -- Branche SST --
        if self.use_sst:
            if self.early_fusion_sst:
                self.sst_branch = nn.Sequential(
                    down_layer(in_chans_sst, n_feat, dr, 3, 5, 2, 3),
                    down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                    down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
                )
            else:
                # Late fusion : une branche indépendante par lag SST
                # Chaque sous-branche prend 1 seul canal (un seul lag)
                self.sst_branches = nn.ModuleList([
                    nn.Sequential(
                        down_layer(1, n_feat, dr, 3, 5, 2, 3), # 1 canal en entrée
                        down_layer(n_feat, n_feat, dr, 3, 3, 2, 2),
                        down_layer(n_feat, n_feat, dr, 3, 3, 1, 1)
                    ) for _ in range(in_chans_sst)
                ])
            
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
            if self.early_fusion_sst:
                out_sst = self.sst_branch(x_sst)
                out_sst = torch.flatten(out_sst, 1)
                features.append(out_sst)
            else:
                # On itère sur chaque lag (canal) de x_sst
                for i in range(x_sst.size(1)):
                    # Slicing i:i+1 pour garder les 4 dimensions (B, 1, H, W)
                    x_lag = x_sst[:, i:i+1, :, :] 
                    
                    # Passage dans la branche dédiée au lag i
                    out_lag = self.sst_branches[i](x_lag)
                    out_lag = torch.flatten(out_lag, 1)
                    features.append(out_lag)
            
        if self.use_slp and x_slp is not None:
            out_slp = self.slp_branch(x_slp)
            out_slp = torch.flatten(out_slp, 1)
            features.append(out_slp)
            
        # Concaténation des vecteurs (Late Fusion)
        x = torch.cat(features, dim=1)
        
        # Décision de classification
        logits = self.head(x) # Renvoie un tenseur de taille [Batch, num_classes]
        
        return logits