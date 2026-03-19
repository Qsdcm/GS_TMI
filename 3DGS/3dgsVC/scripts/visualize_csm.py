import os
import sys
import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to import data.transforms
sys.path.append(os.getcwd())

try:
    from data.transforms import ifft3c, fft3c
except ImportError:
    print("Error: Could not import data.transforms. Make sure you are running this script from the project root (3dgsVC directory).")
    sys.exit(1)

def normalize_image(img):
    """Normalize image to 0-1 range for visualization"""
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

def get_log_mag(kspace):
    """Get log magnitude of k-space"""
    return torch.log(torch.abs(kspace) + 1e-9)

def save_ortho_slices(vol, output_path, title_prefix, mode='magnitude'):
    """
    Save 3 orthogonal central slices of a 3D volume
    vol: 3D tensor (D, H, W)
    mode: 'magnitude', 'log_magnitude', 'phase', 'real'
    """
    cmap = 'gray'
    
    if torch.is_complex(vol):
        if mode == 'log_magnitude':
            vol_data = torch.log(torch.abs(vol) + 1e-9)
            cmap = 'inferno'
        elif mode == 'phase':
            vol_data = torch.angle(vol)
            cmap = 'twilight'
        else: # magnitude
            vol_data = torch.abs(vol)
            cmap = 'gray'
    else:
        vol_data = vol
        cmap = 'gray'
            
    vol_np = vol_data.cpu().numpy()
    d, h, w = vol_np.shape
    
    # Central slices
    s_d = vol_np[d//2, :, :]
    s_h = vol_np[:, h//2, :]
    s_w = vol_np[:, :, w//2]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine min/max for scaling (especially for phase which is -pi to pi)
    vmin, vmax = None, None
    if mode == 'phase':
        vmin, vmax = -np.pi, np.pi
    
    im0 = axs[0].imshow(s_d, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"{title_prefix} (Dim 0)")
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    im1 = axs[1].imshow(s_h, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"{title_prefix} (Dim 1)")
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    im2 = axs[2].imshow(s_w, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].set_title(f"{title_prefix} (Dim 2)")
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{title_prefix} - {mode}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize CSM Estimation (Full vs ACS)')
    parser.add_argument('--data_path', type=str, default="/data0/congcong/data/3D_data/h5data/ksp_full.h5", 
                        help='Path to full k-space h5 file')
    parser.add_argument('--acs_lines', type=int, default=32, help='Number of ACS lines for center calibration')
    parser.add_argument('--output_dir', type=str, default="debug_csm_vis", help='Directory to save visualization')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    with h5py.File(args.data_path, 'r') as f:
        # Determine key
        if 'kspace' in f.keys():
            key = 'kspace'
        elif 'ksp' in f.keys():
            key = 'ksp'
        else:
            key = list(f.keys())[0]
        
        kspace_data = f[key][:]
    
    kspace_full = torch.from_numpy(kspace_data).to(torch.complex64).to(device)
    print(f"K-space shape: {kspace_full.shape}")
    
    num_coils, kx, ky, kz = kspace_full.shape
    
    # ==========================================
    # Method 1: Full K-space Estimation
    # ==========================================
    print("\n--- Method 1: Full K-space Estimation ---")
    
    # A. The K-space Used
    # It is just kspace_full
    save_ortho_slices(kspace_full[0], 
                     os.path.join(args.output_dir, "01_Full_Kspace_Input_Mag.png"),
                     "Input K-space (Full, Dim 0)", mode='log_magnitude')
    save_ortho_slices(kspace_full[0], 
                     os.path.join(args.output_dir, "01_Full_Kspace_Input_Phase.png"),
                     "Input K-space (Full, Dim 0)", mode='phase')

    # B. Estimation Process
    img_full = ifft3c(kspace_full)
    rss_full = torch.sqrt(torch.sum(torch.abs(img_full) ** 2, dim=0)) + 1e-12
    csm_full = img_full / rss_full.unsqueeze(0)
    
    # C. Visualization
    # Visualize RSS (Anatomy)
    save_ortho_slices(rss_full, 
                     os.path.join(args.output_dir, "02_Full_RSS_Image.png"),
                     "RSS Image (Full K-space)", mode='real')
                     
    # Visualize CSM (Coil 0)
    save_ortho_slices(csm_full[0], 
                     os.path.join(args.output_dir, "03_Full_CSM_Coil0_Mag.png"),
                     "Estimated CSM Coil 0 (Full)", mode='magnitude')
    save_ortho_slices(csm_full[0], 
                     os.path.join(args.output_dir, "03_Full_CSM_Coil0_Phase.png"),
                     "Estimated CSM Coil 0 (Full)", mode='phase')


    # ==========================================
    # Method 2: ACS (Center) Estimation
    # ==========================================
    print(f"\n--- Method 2: ACS Estimation ({args.acs_lines} lines) ---")
    
    # A. Create ACS Mask & K-space
    acs_mask = torch.zeros_like(kspace_full, dtype=torch.float32)
    cy, cz = ky // 2, kz // 2
    half_acs = args.acs_lines // 2
    
    # Calculate indices (Assuming 2D Phase Encoding on dim 2 and 3, dim 1 is Readout/Full)
    y_start = max(0, cy - half_acs)
    y_end = min(ky, cy + half_acs)
    z_start = max(0, cz - half_acs)
    z_end = min(kz, cz + half_acs)
    
    # Masking
    acs_mask[:, :, y_start:y_end, z_start:z_end] = 1.0
    kspace_acs = kspace_full * acs_mask
    
    # Visualize The K-space Used (ACS)
    save_ortho_slices(kspace_acs[0], 
                     os.path.join(args.output_dir, "04_ACS_Kspace_Input_Mag.png"),
                     "Input K-space (ACS Only, Dim 0)", mode='log_magnitude')
    save_ortho_slices(kspace_acs[0], 
                     os.path.join(args.output_dir, "04_ACS_Kspace_Input_Phase.png"),
                     "Input K-space (ACS Only, Dim 0)", mode='phase')
                     
    # B. Estimation Process
    img_acs = ifft3c(kspace_acs)
    rss_acs = torch.sqrt(torch.sum(torch.abs(img_acs) ** 2, dim=0)) + 1e-12
    csm_acs = img_acs / rss_acs.unsqueeze(0)
    
    # C. Visualization
    # Visualize RSS (Low Res Anatomy)
    save_ortho_slices(rss_acs, 
                     os.path.join(args.output_dir, "05_ACS_RSS_Image.png"),
                     "RSS Image (ACS Low Res)", mode='real')
                     
    # Visualize CSM (Coil 0)
    save_ortho_slices(csm_acs[0], 
                     os.path.join(args.output_dir, "06_ACS_CSM_Coil0_Mag.png"),
                     "Estimated CSM Coil 0 (ACS)", mode='magnitude')
    save_ortho_slices(csm_acs[0], 
                     os.path.join(args.output_dir, "06_ACS_CSM_Coil0_Phase.png"),
                     "Estimated CSM Coil 0 (ACS)", mode='phase')

    print("\nDone! Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()
