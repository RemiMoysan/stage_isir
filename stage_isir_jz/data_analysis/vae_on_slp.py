import torch
import xarray as xr
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import time
import argparse
import os
import matplotlib.pyplot as plt
import copy

# Pas vraiment de sens d'utiliser les memes weights multiplicateurs que pour la PCA en augmentant l'intensité de certains pixels étant donné qu'on fait quand meme des convolutions / calcul de gradient
# A la place on pondère juste la MSE de sortie. Attention cela veut dire qu'on ne peut pas pondérer l'entrée, donc c'est au réseau de s'adapter tout seul à comment pondérer ce qui peut être rendu compliqué par le fait qu'il y a des poids partagés dans un CNN. 

# ============================================================
# 1. CLASSES ET FONCTIONS (Modèle, Loss, Dataset, Visu)
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

# --- NOUVEAU : Fonction de calcul du std global ---
def compute_global_std(members, file_path_SLP, selected_months, duree_lissage=10, monthly_reduction=False, lat_weight=False):
    print("Calcul de slp_std rigoureux (Pass 1/2)...")
    total_sum_sq = 0.0
    total_weights = 0.0
    map_weight_sum = None

    for m in members:
        if monthly_reduction:
            filename = f'PSL_anom_LE2-{m}_1mo.nc'
        else:
            filename = f'PSL_anom_LE2-{m}_{duree_lissage}d.nc' if duree_lissage != 0 else f'PSL_anom_LE2-{m}.nc'

        ds = xr.open_dataset(os.path.join(file_path_SLP, filename))["PSL"]
        ds = ds.sel(time=ds['time'].dt.month.isin(selected_months))
        data = np.nan_to_num(ds.values, nan=0.0)

        if lat_weight:
            if map_weight_sum is None:
                lats = ds['lat'].values
                coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                wgts = np.sqrt(coslat).reshape(1, len(lats), 1)
                # Simplification directe : somme des poids 1D * nombre de longitudes
                n_lon = data.shape[2] # data.shape est (time, lat, lon)
                map_weight_sum = np.sum(coslat) * n_lon
            else:
                lats = ds['lat'].values
                coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                wgts = np.sqrt(coslat).reshape(1, len(lats), 1)

            data = data * wgts
            total_weights += data.shape[0] * map_weight_sum
        else:
            total_weights += data.size

        total_sum_sq += np.sum(data**2)

    slp_std_rigoureux = float(np.round(np.sqrt(total_sum_sq / total_weights), 2))
    print(f"--> slp_std calculé et arrondi : {slp_std_rigoureux}")
    return slp_std_rigoureux

class Dataset_SLP_VAE(torch.utils.data.Dataset):
    """Dataset ultra-léger qui ne charge QUE la SLP (pas de SST, pas de lag)"""
    def __init__(self, members, selected_months, file_path_SLP, duree_lissage=10, monthly_reduction=False, slp_std=1.0, lat_weight=False):
        self.members = members
        self.slp_std = slp_std
        self.lat_weight = lat_weight
        if monthly_reduction:
            self.SLP = xr.open_dataset(os.path.join(file_path_SLP, f'PSL_anom_LE2-{members[0]}_1mo.nc'))
        else:
            if duree_lissage != 0:
                self.SLP = xr.open_dataset(os.path.join(file_path_SLP, f'PSL_anom_LE2-{members[0]}_{duree_lissage}d.nc'))
            else:
                self.SLP = xr.open_dataset(os.path.join(file_path_SLP, f'PSL_anom_LE2-{members[0]}.nc'))
        self.SLP = self.SLP.drop_vars(["PSL"], errors="ignore") 

        for count, i in enumerate(members):
            if monthly_reduction:
                filename = f'PSL_anom_LE2-{i}_1mo.nc'
            else:
                filename = f'PSL_anom_LE2-{i}_{duree_lissage}d.nc' if duree_lissage != 0 else f'PSL_anom_LE2-{i}.nc'
            self.SLP[count] = xr.open_dataset(os.path.join(file_path_SLP, filename))["PSL"] 
            
        self.SLP = self.SLP.sel(time=self.SLP['time'].dt.month.isin(selected_months))
        self.list_dates = self.SLP.time.values.tolist()
        # Calcul anticipé des poids (Tenseur PyTorch) pour la Loss
        self.wgts_tensor = None
        if self.lat_weight:
            lats = self.SLP[0]['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            # NOUVEAU : On stocke directement cos(lat), sans la racine !
            self.wgts_tensor = torch.tensor(coslat.reshape(1, 1, len(lats), 1)).float()
        
    def __len__(self):
        return len(self.members) * len(self.SLP.time)

    def __getitem__(self, index):
        member_idx = index // len(self.SLP.time)
        t_Atm = self.list_dates[index % len(self.SLP.time)]        
        slp = self.SLP[member_idx].sel(time=t_Atm)
        # Format [Channel, H, W] -> [1, 53, 113]
        x = torch.nan_to_num(torch.tensor(np.array(slp.data)), nan=0).unsqueeze(0).float() / self.slp_std

        # ATTENTION : On NE PONDÈRE PLUS l'entrée ici ! Le CNN voit le monde réel normalisé.
        
        # Le VAE s'entraîne à reproduire l'entrée, donc X = cible
        return x, x, t_Atm.strftime('%Y-%m-%d')

def plot_vae_reconstructions(slp_true_list, slp_recon_list, time_list, outdir, epoch, slp_std=1.0, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000]):
    """Trace les vraies SLP vs les reconstructions du VAE"""
    y_true = np.concatenate(slp_true_list, axis=0).squeeze()
    y_recon = np.concatenate(slp_recon_list, axis=0).squeeze()
    
    N = y_true.shape[0]
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)
    
    if num_samples == 0: return

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
    fig.suptitle(f"VAE Reconstructions (Epoch {epoch})", fontsize=16)

    for i, idx in enumerate(valid_indices):
        true_map = y_true[idx]
        recon_map = y_recon[idx]
        date = time_list[idx]

        # === RETOUR À LA PHYSIQUE ===
        # PLUS BESOIN DU BLOC if wgts_2d is not None: ...
        
        # On multiplie directement
        true_map = true_map * slp_std
        recon_map = recon_map * slp_std

        # Échelle dynamique symétrique
        vlim = max(np.max(np.abs(true_map)), np.max(np.abs(recon_map)))
        vmin, vmax = -vlim, vlim

        ax_row = axes[i] if num_samples > 1 else axes

        # Vraie SLP
        im1 = ax_row[0].imshow(true_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax_row[0].set_title(f"Vraie SLP - {date}")
        cbar1 = fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)
        cbar1.set_label("Pa")

        # SLP Reconstruite
        im2 = ax_row[1].imshow(recon_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax_row[1].set_title(f"Reconstruction VAE")
        cbar2 = fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Pa")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"vae_reconstructions_epoch_{epoch}.png"), dpi=150)
    plt.close()

def loss_figure(epochs, train_losses, val_losses, outdir_new, epoch_times=None):
    """
    Plot of the loss figure with a dual X-axis for Epochs and Time.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    
    ax.plot(range(epochs), train_losses, label='Train Loss', color='C0')
    ax.plot(range(epochs), val_losses, label='Validation Loss', color='C1')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title("Loss Evolution during training\n", pad=20) # pad pour faire de la place au 2eme axe
    
    # --- NOUVEAU : Création du 2ème axe des abscisses ---
    if epoch_times is not None and len(epoch_times) == epochs:
        ax2 = ax.twiny() # Crée un second axe X qui partage le même axe Y
        
        # On force les limites à être strictement identiques
        ax2.set_xlim(ax.get_xlim())
        
        # On choisit combien de "ticks" on veut afficher (ex: 6 max pour que ce soit lisible)
        num_ticks = min(6, epochs)
        if epochs > 1:
            tick_indices = np.linspace(0, epochs - 1, num_ticks, dtype=int)
        else:
            tick_indices = [0]
            
        # On place les graduations aux mêmes endroits que les époques choisies
        ax2.set_xticks(tick_indices)
        # On écrit le temps formaté en minutes (ex: "15.2m")
        ax2.set_xticklabels([f"{epoch_times[i]:.1f}m" for i in tick_indices])
        ax2.set_xlabel("Temps d'entraînement cumulé (minutes)")
    
    figs_file = "Fig_loss-evolution-during-training.png"
    figs_filename = os.path.join(outdir_new, figs_file)
    plt.tight_layout()
    plt.savefig(figs_filename)
    plt.close()

# ============================================================
# 2. CONFIGURATION GÉNÉRALE
# ============================================================

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    parser.add_argument('--duree_lissage', type=int, default=30, help='Durée du lissage en jours (10 ou 30)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent')
    parser.add_argument('--beta_kld', type=float, default=1.0, help='Poids de la régularisation KL (augmenter si espace latent trop bruyant)')
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres d\'ensemble à utiliser pour l\'entraînement (max 87)')
    parser.add_argument('--nb_epochs', type=int, default=50, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate pour l\'optimiseur Adam')

    # --- NOUVEAUX ARGUMENTS ---
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Appliquer la pondération spatiale sqrt(cos(lat))')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[11, 12, 1, 2], help='Mois d\'hiver à considérer (ex: 11 12 1 2 pour NDJF)')

    args = parser.parse_args()

    bs = args.bs
    lr = args.lr # Légèrement plus élevé pour un VAE souvent
    beta_kld = args.beta_kld # Poids de la régularisation KL (augmenter si espace latent trop bruyant)
    duree_lissage = args.duree_lissage
    latent_dim = args.latent_dim
    monthly_reduction = args.monthly_reduction
    lat_weight = args.lat_weight
    winter_months = args.winter_months
    months_label = "_".join(map(str, winter_months))

    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    nb_members_train = args.nb_members_train
    train_members = train_members_87[:nb_members_train]
    val_members = ['1001.001',"1301.010"]
    # ou val_members = ["1301.010"]

    nb_epochs = args.nb_epochs           
    patience = 20         

    if args.machine == 'hacienda':
        path_slp = "/data/moysan/data/SLP/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/vae_slp/"
    elif args.machine == 'jean-zay-work':
        path_slp = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/"
    elif args.machine == 'jean-zay-scratch':
        path_slp = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/"
    else: # mac_local
        path_slp = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/vae_slp/"

    fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000]
    if monthly_reduction:
        fixed_indices = [i//30 for i in fixed_indices] 
    # --- CALCUL DYNAMIQUE DU SLP_STD ---
    dynamic_slp_std = compute_global_std(
        train_members, path_slp, winter_months, 
        duree_lissage=duree_lissage, 
        monthly_reduction=monthly_reduction, 
        lat_weight=lat_weight
    )

    if not monthly_reduction:
        outdir_name = f'ConvVAE_months_{months_label}_bs{bs}_lr{lr}_beta{beta_kld}_{nb_members_train}members_latent{latent_dim}_duree{duree_lissage}d_wgt{lat_weight}_slp_std{dynamic_slp_std}'
    else:
        outdir_name = f'ConvVAE_months_{months_label}_bs{bs}_lr{lr}_beta{beta_kld}_{nb_members_train}members_latent{latent_dim}_monthly_reduction_wgt{lat_weight}_slp_std{dynamic_slp_std}'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    # ============================================================
    # 3. INITIALISATION
    # ============================================================
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Number of VAE parameters : ", sum(p.numel() for p in model.parameters()))

    train_losses, val_losses = [], []
    best_val_loss = float('inf') 

    # ============================================================
    # 4. TRAINING LOOP
    # ============================================================
    start_time = time.time() 
    epoch_times = []

    patience_counter = 0

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")


    val_set = Dataset_SLP_VAE(members=val_members, selected_months=winter_months, file_path_SLP=path_slp, duree_lissage=duree_lissage,monthly_reduction=monthly_reduction, slp_std=dynamic_slp_std, lat_weight=lat_weight)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    training_set = Dataset_SLP_VAE(members=train_members, selected_months=winter_months, file_path_SLP=path_slp, duree_lissage=duree_lissage,monthly_reduction=monthly_reduction, slp_std=dynamic_slp_std, lat_weight=lat_weight)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    # NOUVEAU : Récupération du tenseur de pondération pour la Loss
    wgts_gpu = None
    if lat_weight and training_set.wgts_tensor is not None:
        wgts_gpu = training_set.wgts_tensor.to(device)

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_loss, running_mse, running_kld = 0.0, 0.0, 0.0
        
        for batch_idx, (inputs, _, _) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(inputs)
            loss, mse, kld = vae_loss(recon_batch, inputs, mu, logvar, beta=beta_kld, wgts_tensor=wgts_gpu)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_mse += mse.item()
            running_kld += kld.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.2f} (MSE: {running_mse/len(trainloader):.2f}, KLD: {running_kld/len(trainloader):.2f})')

        # ---------------- VALIDATION ----------------
        model.eval()
        running_val_loss = 0.0
        slp_true_list, slp_recon_list, time_list = [], [], []

        with torch.no_grad():
            for batch_idx, (inputs, _, dates) in enumerate(valloader):
                inputs = inputs.to(device, non_blocking=True)
                recon_batch, mu, logvar = model(inputs)
                
                loss, _, _ = vae_loss(recon_batch, inputs, mu, logvar, beta=beta_kld, wgts_tensor=wgts_gpu)
                running_val_loss += loss.item()

                if (epoch + 1) % 2 == 0: # On sauvegarde pour le plot tous les 2 epochs
                    time_list.extend(dates)
                    slp_true_list.append(inputs.cpu().numpy())
                    slp_recon_list.append(recon_batch.cpu().numpy())

        val_loss = running_val_loss / len(valloader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1} | Val Loss: {val_loss:.2f}')

        # ---------------- EARLY STOPPING & PLOTS ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{outdir}/best_vae.pth')
            patience_counter = 0
            print(f"--> Saved best VAE model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered.")
                break

        if (epoch + 1) % 2 == 0:
            plot_vae_reconstructions(slp_true_list, slp_recon_list, time_list, outdir, epoch=(epoch + 1),slp_std=dynamic_slp_std,fixed_indices=fixed_indices)
            
        # store cumulative training time in minutes (for the secondary X-axis)
        epoch_times.append((time.time() - start_time) / 60)
        loss_figure(len(train_losses), train_losses, val_losses,outdir, epoch_times)
        # Plot standard de la loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.savefig(os.path.join(outdir, 'loss_curve.png'))
        plt.close()

    elapsed_time = (time.time() - start_time) / 60
    print(f"Training complete in {elapsed_time:.2f} minutes.")