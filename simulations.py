from isort import file
import numpy as np
import numbers
import scipy.stats as scipy
import torch
import matplotlib
from matplotlib import gridspec
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from torch_model_multi_te import torch_deltaM_multite_model_batch, torch_deltaM_multite_model_single
import nibabel as nib
from PI_NN import Net
from tqdm.auto import tqdm
from matplotlib.patches import Patch

def apply_academic_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "text.color": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "image.interpolation": "none",
        "figure.facecolor": "black",
    })

def sim_signal(
    SNR, TE, TI, ntes, tissue, sims=5, distribution='normal',
    rician=False, ATT_min=0.5, ATT_max=3.0, CBF_min=5, CBF_max=100, Texch_min=0.05, Texch_max=0.5,
    state=123, T1a=1.65, T2a=0.150,
    itt=0.2, lambd=0.9, alpha=0.68,
    BRAIN_SLICE = False, z_slice_pos=0.7,
    GM_ATT = 1.28, GM_CBF = 60, GM_Texch = 0.228,
    WM_ATT = 1.69, WM_CBF = 22, WM_Texch = 0.127,

):
    """
    Simulate the ASL signal based on the provided parameters.

    Returns:
        data_sim : np.ndarray, shape (sims, nTE, nPLD)
        param_grid : dict of np.ndarray with same shape, holding ATT, CBF, TE, PLD
    """

    rg = np.random.RandomState(state)
    device = torch.device("cpu")


    # ---- sample ATT / CBF ---------------------------------------------------
    if BRAIN_SLICE:
        ATT = np.zeros((sims,))
        CBF = np.zeros((sims,))
        texch = np.zeros((sims,))
        if tissue == "grey matter":
            ATT_torch = torch.full((sims,), GM_ATT, dtype=torch.float32, device=device)
            CBF_torch = torch.full((sims,), GM_CBF, dtype=torch.float32, device=device)
            Texch_torch = torch.full((sims,), GM_Texch, dtype=torch.float32, device=device)
        elif tissue == "white matter":
            ATT_torch = torch.full((sims,), WM_ATT, dtype=torch.float32, device=device)
            CBF_torch = torch.full((sims,), WM_CBF, dtype=torch.float32, device=device)
            Texch_torch = torch.full((sims,), WM_Texch, dtype=torch.float32, device=device)
    else:
        if distribution == 'normal':
            test = rg.standard_normal((sims, 1))
            ATT = np.abs((ATT_max + ATT_min) / 2 + test * (ATT_max - ATT_min) / 6)
            test = rg.standard_normal((sims, 1))
            CBF = np.abs((CBF_max + CBF_min) / 2 + test * (CBF_max - CBF_min) / 6)
            test = rg.standard_normal((sims, 1))
            texch = np.abs((Texch_max + Texch_min) / 2 + test * (Texch_max - Texch_min) / 6)
        elif distribution == 'uniform':
            test = rg.uniform(0, 1, (sims, 1))
            ATT = ATT_min + test * (ATT_max - ATT_min)
            test = rg.uniform(0, 1, (sims, 1))
            CBF = CBF_min + test * (CBF_max - CBF_min)
            test = rg.uniform(0, 1, (sims, 1))
            texch = Texch_min + test * (Texch_max - Texch_min)
        elif distribution == 'normal-wide':
            test = rg.standard_normal((sims, 1))
            ATT = np.abs((ATT_max + ATT_min) / 2 + test * (ATT_max - ATT_min) / 4)
            test = rg.standard_normal((sims, 1))
            CBF = np.abs((CBF_max + CBF_min) / 2 + test * (CBF_max - CBF_min) / 4)
            test = rg.standard_normal((sims, 1))
            texch = np.abs((Texch_max + Texch_min) / 2 + test * (Texch_max - Texch_min) / 4)
        elif distribution == 'uniform-wide':
            test = rg.uniform(0, 1, (sims, 1))
            ATT = ATT_min + test * (ATT_max - ATT_min) * 1.2 - (ATT_max - ATT_min) * 0.1
            test = rg.uniform(0, 1, (sims, 1))
            CBF = CBF_min + test * (CBF_max - CBF_min) * 1.2 - (CBF_max - CBF_min) * 0.1
            test = rg.uniform(0, 1, (sims, 1))
            texch = Texch_min + test * (Texch_max - Texch_min) * 1.2 - (Texch_max - Texch_min) * 0.1
        else:
            raise ValueError("distribution must be 'normal', 'normal-wide' or 'uniform'")
    
    texch = np.clip(texch, Texch_min, Texch_max)

    ATT = np.ravel(ATT)  # (sims,)
    CBF = np.ravel(CBF)  # (sims,)
    texch = np.ravel(texch)  # (sims,)


    TE = np.atleast_1d(np.array(TE, dtype=float))   # unique TEs per PLD
    TI = np.atleast_1d(np.array(TI, dtype=float)) # TIs / PLDs

    nTI = TI.shape[0]
    nTE = TE.shape[0]

    # ---- handle ntes: assume same number of echoes per PLD -------------------
    # This implementation assumes *uniform* number of echoes per PLD.
    
    if ntes is None:
        # assume same nTE for each TI
        ntes_vec = np.full(nTI, nTE, dtype=np.int32)
    else:
        if np.isscalar(ntes):
            ntes = int(ntes)
            if ntes != nTE:
                raise ValueError(
                    "Scalar ntes must equal len(TE) for this vectorised implementation."
                )
            ntes_vec = np.full(nTI, nTE, dtype=np.int32)
        else:
            ntes_arr = np.asarray(ntes, dtype=int)
            if ntes_arr.shape[0] != nTI:
                raise ValueError(
                    "ntes must be scalar or length len(TI); got shape {}".format(
                        ntes_arr.shape
                    )
                )
            if not np.all(ntes_arr == ntes_arr[0]):
                raise NotImplementedError(
                    "Variable ntes per TI not supported in this vectorised version."
                )
            if ntes_arr[0] != nTE:
                raise ValueError(
                    "ntes per TI must equal len(TE) for this vectorised version."
                )
            ntes_vec = ntes_arr.astype(np.int32)

    # flattened TE vector: [TEs for TI0, TEs for TI1, ...]
    tes_flat = np.tile(TE, nTI)  # length = nTI * nTE

    # ---- build grids for param_grid -----------------------------------------
    data_shape = (sims, nTE, nTI)

    ATT_grid = np.broadcast_to(ATT.reshape(sims, 1, 1), data_shape).copy()
    CBF_grid = np.broadcast_to(CBF.reshape(sims, 1, 1), data_shape).copy()
    Texch_grid = np.broadcast_to(texch.reshape(sims, 1, 1), data_shape).copy()
    TE_grid  = np.broadcast_to(TE.reshape(1, nTE, 1),   data_shape).copy()
    TI_grid = np.broadcast_to(TI.reshape(1, 1, nTI), data_shape).copy()

    param_grid = {
        "ATT": ATT_grid,
        "CBF": CBF_grid,
        "Texch": Texch_grid,
        "TE":  TE_grid,
        "TI": TI_grid,
    }

    if not BRAIN_SLICE:
        ATT_torch = torch.tensor(ATT, dtype=torch.float32, device=device)  # (sims,)
        CBF_torch = torch.tensor(CBF, dtype=torch.float32, device=device)  # (sims,)
        Texch_torch = torch.tensor(texch, dtype=torch.float32, device=device)  # (sims,)
    TI_torch = torch.tensor(TI, dtype=torch.float32, device=device)  # tis
    TE_torch  = torch.tensor(tes_flat, dtype=torch.float32, device=device)


    # ---- tissue-specific T1/T2 ----------------------------------------------
    if tissue == "grey matter":
        T1 = 1.615
        T2 = 0.083  
        # ntes_vec is length nTI
        data_flat = torch_deltaM_multite_model_batch(
        TI_torch, TE_torch, ntes_vec,
        ATT_torch, CBF_torch, Texch_torch,
        m0a=1.0, taus=1.5,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
        itt=itt, lambd=lambd, alpha=alpha, show_progress=True)

    elif tissue == "white matter":
        T1 = 0.911
        T2 = 0.075
        data_flat = torch_deltaM_multite_model_batch(
        TI_torch, TE_torch, ntes_vec,
        ATT_torch, CBF_torch, Texch_torch,
        m0a=1.0, taus=1.5,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
         itt=itt, lambd=lambd, alpha=alpha, show_progress=True
        )
    elif tissue == "both":
        #do half grey matter, half white matter
        print("Simulating mixed tissue: half grey matter, half white matter")
        T1_gm = 1.615
        T2_gm = 0.083  
        T1_wm = 0.911
        T2_wm = 0.075
        n_gm = sims // 2
        n_wm = sims - n_gm
        data_flat_gm = torch_deltaM_multite_model_batch(
        TI_torch, TE_torch, ntes_vec,
        ATT_torch[:n_gm], CBF_torch[:n_gm], Texch_torch[:n_gm],
        m0a=1.0, taus=1.5,
        t1=T1_gm, t1b=T1a, t2=T2_gm, t2b=T2a,
        itt=itt, lambd=lambd, alpha=alpha, show_progress=True)
        data_flat_wm = torch_deltaM_multite_model_batch(
        TI_torch, TE_torch, ntes_vec,
        ATT_torch[n_gm:], CBF_torch[n_gm:], Texch_torch[n_gm:],
        m0a=1.0, taus=1.5,
        t1=T1_wm, t1b=T1a, t2=T2_wm, t2b=T2a,
        itt=itt, lambd=lambd, alpha=alpha, show_progress=True)
        data_flat = torch.cat([data_flat_gm, data_flat_wm], dim=0)

        #BOOLEAN MASK FOR GM AND WM
        G_or_W_mask = torch.zeros((sims,), dtype=torch.bool, device=device)
        G_or_W_mask[:n_gm] = True  # first half GM, second half WM

    else:
        raise ValueError("tissue must be 'grey matter', 'white matter' or 'both'")


    # data_flat: (sims, nTI * nTE)

    # reshape to (sims, nTI, nTE) then to (sims, nTE, nTI)
    data_sim = data_flat.view(sims, nTI, nTE)#.permute(0, 2, 1)  # (sims, nTE, nTI)
    data_sim_noisy = torch.zeros_like(data_sim, device=device)

    # data_sim: (sims, nTE, nTI)

    if isinstance(SNR, numbers.Number):
        if SNR > 0 and not isinstance(SNR, np.ndarray):
            # estimate reference amplitude per voxel
            ref = data_sim.abs() 
            ref = torch.clamp(ref, min=1e-8)                     # avoid zero
            ref = torch.maximum(ref, data_sim.max() * 0.005)  # avoid tiny values
            noise_std = ref / float(SNR)

            noise_real = torch.normal(0.0, noise_std)

            if rician:
                noise_imag = torch.normal(0.0, noise_std)
                data_sim_noisy = torch.sqrt((data_sim + noise_real) ** 2 + noise_imag ** 2)
            else:
                data_sim_noisy = data_sim + noise_real

    else:
        SNR_np = np.asarray(SNR, dtype=float)
        SNR = torch.tensor(SNR_np, dtype=torch.float32, device=device)
        #Iterate over sims and add different noise levels
        for i in range(sims):

            SNR_index = np.random.randint(0, len(SNR_np))
            ref_i = data_sim[i].abs()

            ref_i = torch.clamp(ref_i, min=1e-8)
            ref_i = torch.maximum(ref_i, data_sim[i].max() * 0.02)

            noise_std_i = ref_i / float(SNR[SNR_index])
            noise_real_i = torch.normal(0.0, noise_std_i)
            

            if rician:
                noise_imag_i = torch.normal(0.0, noise_std_i)
                data_sim_noisy[i] = torch.sqrt((data_sim[i] + noise_real_i) ** 2 + noise_imag_i ** 2)
            else:
                data_sim_noisy[i] = data_sim[i] + noise_real_i

    # convert to numpy before returning
    data_sim_noisy = data_sim_noisy.detach().cpu().numpy()
    data_sim = data_sim.detach().cpu().numpy()

    
    if tissue == "both":
        param_grid["G_or_W_mask"] = G_or_W_mask.detach().cpu().numpy()

    return data_sim_noisy, param_grid, data_sim

def make_training_data(
    SNR, TE, TI, ntes, tissue, sims=1000,
    distribution="normal", state=123, z_slice_pos=0.7, rician=True, save_files=False
):
    """
    Generate ML-ready (X, y) from the ASL simulator.

    X: (sims, nTE * nTI)  -- simulated signals flattened
    y: (sims, 3)           -- columns: [ATT, CBF, Texch]
    """
    data_sim, param_grid, data_sim_clean = sim_signal(
        SNR=SNR, TE=TE, TI=TI, ntes=ntes, tissue=tissue,
        sims=sims, distribution=distribution, state=state
    )

    sims_n, nTE, nTI = data_sim.shape

    X = data_sim.reshape(sims_n, nTE * nTI)
    X_clean = data_sim_clean.reshape(sims_n, nTE * nTI)


    ATT = param_grid["ATT"][:, 0, 0]
    CBF = param_grid["CBF"][:, 0, 0]
    Texch = param_grid["Texch"][:, 0, 0]
    G_or_W_mask = param_grid.get("G_or_W_mask", None)
    
    y = np.stack([ATT, CBF, Texch], axis=1)

    if save_files:
    #save to X and y files for later use
        np.save("simulated_training_data.npy", X)
        np.save("simulated_training_labels.npy", y)
        np.save("simulated_training_data_clean.npy", X_clean)
        #save te and ti values
        np.save("training_TE_values.npy", TE)
        np.save("training_TI_values.npy", TI)
        np.save("training_G_or_W_mask.npy", G_or_W_mask)

    return X, y, X_clean, G_or_W_mask

def matter_masks(grey_matter_file_name, white_matter_file_name):
    """Load matter masks from NIfTI files.

    Returns:
        grey_matter_mask: np.ndarray of bool
        white_matter_mask: np.ndarray of bool
    """
    gm_img = nib.load(grey_matter_file_name)
    wm_img = nib.load(white_matter_file_name)

    gm_data = gm_img.get_fdata()
    wm_data = wm_img.get_fdata()
    #display basic data

    # Boolean masks (nonzero = masked)
    gm_mask = gm_data > 0
    wm_mask = wm_data > 0
    
    # Find overlap
    overlap = gm_mask & wm_mask

    if np.any(overlap):
        print(f"Resolving overlap in {overlap.sum()} voxels using intensity values.")

        # Where GM intensity is greater: GM stays, WM removed
        gm_wins = overlap & (gm_data >= wm_data)

        # Where WM intensity is greater: WM stays, GM removed
        wm_wins = overlap & (wm_data > gm_data)

        # Apply updates
        gm_mask[wm_wins] = False
        wm_mask[gm_wins] = False

    return gm_mask, wm_mask

def simulated_brain_data(
    grey_matter_file_name, white_matter_file_name, TE, TI, ntes, SNR=2, 
    rician=False, z_slice_pos=0.7):
    """Generate simulated ASL data for a brain volume.
    Returns:
        data_sim: np.ndarray, shape (nx, ny, nz, nTE*nTI)
        param_grid: dict of np.ndarray with same shape, holding ATT, CBF, TE, TI
    """

    gm_mask, wm_mask = matter_masks(grey_matter_file_name, white_matter_file_name)
    nx, ny, nz = gm_mask.shape
    n_voxels = nx * ny 
    print(f"Brain volume shape: {nx} x {ny} x {nz} = {n_voxels} voxels")

    TE = np.atleast_1d(np.array(TE, dtype=float))   # unique TEs per TI
    TI = np.atleast_1d(np.array(TI, dtype=float)) # TIs / TIs

    nTI = TI.shape[0]
    nTE = TE.shape[0]

    # ---- handle ntes: assume same number of echoes per TI -------------------
    # This implementation assumes *uniform* number of echoes per TI.
    if ntes is None:
        # assume same nTE for each TI
        ntes_vec = np.full(nTI, nTE, dtype=np.int32)
    else:
        if np.isscalar(ntes):
            ntes = int(ntes)
            if ntes != nTE:
                raise ValueError(
                    "Scalar ntes must equal len(TE) for this vectorised implementation."
                )
            ntes_vec = np.full(nTI, nTE, dtype=np.int32)
        else:
            ntes_arr = np.asarray(ntes, dtype=int)
            if ntes_arr.shape[0] != nTI:
                raise ValueError(
                    "ntes must be scalar or length len(TI); got shape {}".format(
                        ntes_arr.shape
                    )
                )
            if not np.all(ntes_arr == ntes_arr[0]):
                raise NotImplementedError(
                    "Variable ntes per TI not supported in this vectorised version."
                )
            if ntes_arr[0] != nTE:
                raise ValueError(
                    "ntes per TI must equal len(TE) for this vectorised version."
                )
            ntes_vec = ntes_arr.astype(np.int32)

    # Simulate slice at z_slice_pos
    z_slice = np.floor(z_slice_pos * nz).astype(int)
    gm_mask_slice = gm_mask[:, :, z_slice]
    wm_mask_slice = wm_mask[:, :, z_slice]
    data_sim = np.zeros((nx, ny, nTI*nTE), dtype=np.float32)

    if gm_mask_slice.sum() > 0:
        print("Simulating", gm_mask_slice.sum(), "grey matter voxels")
        data_sim_gm, _, _ = sim_signal(
            SNR=SNR, TE=TE, TI=TI, ntes=ntes, tissue="grey matter", sims=gm_mask_slice.sum(),
            distribution="normal", BRAIN_SLICE=True)
        

    if wm_mask_slice.sum() > 0:
        print("Simulating", wm_mask_slice.sum(), "white matter voxels")
        data_sim_wm, _ , _= sim_signal(
            SNR=SNR, TE=TE, TI=TI, ntes=ntes, tissue="white matter", sims=wm_mask_slice.sum(),
            distribution="normal", BRAIN_SLICE=True)
    
    # Fill in data_sim
    gm_indices = np.argwhere(gm_mask_slice)
    for idx, (x, y) in enumerate(gm_indices):
        data_sim[x, y, :] = data_sim_gm[idx].flatten()
    wm_indices = np.argwhere(wm_mask_slice)
    for idx, (x, y) in enumerate(wm_indices):
        data_sim[x, y, :] = data_sim_wm[idx].flatten()
        

    #plot brain slices of simulated data at first TE and a spread of TIs
    # data_sim shape: (nx, ny, nTE*nTI)
    slice_z = nz // 2
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    ti_indices = [0, nTI//3, 2*nTI//3, nTI-1]
    for i, ti_idx in enumerate(ti_indices):
        im = axes[i].imshow(data_sim[:, :, ti_idx * nTE], cmap='gray')
        axes[i].set_title(f'Simulated ASL Signal (TI={TI[ti_idx]:.2f}s, TE={TE[0]:.2f}s)')
        #fig.colorbar(im, ax=axes[i], label='Signal Intensity')
    fig.tight_layout()
    plt.show()
    
    nib.save(nib.Nifti1Image(data_sim, affine=np.eye(4)), 'simulated_asl_data.nii')

    return data_sim
    # load neural network model to predict parameters from simulated data
   
def plot_ti_te_mosaic(
    data_sim_slice, TE, TI, mask=None, title=None, savepath=None,
    cmap_name="gist_grey", percentile_range=(2, 98)
):
    """
    Compact TI/TE grid with ZERO inter-panel gaps by rendering a single mosaic image.

    data_sim_slice: (nx, ny, nTI*nTE) where index = ti_idx*nTE + te_idx
    TE: length nTE
    TI: length nTI
    mask: optional (nx, ny) boolean mask; outside becomes NaN (shown as white)
    """
    apply_academic_style()

    TE = np.asarray(TE, dtype=float).ravel()
    TI = np.asarray(TI, dtype=float).ravel()
    nTI, nTE = len(TI), len(TE)
    # cut off TI short
    TI = TI[:nTI]


    nx, ny, _ = data_sim_slice.shape
    grid = data_sim_slice.reshape(nx, ny, nTI, nTE)
    nTI = len(TI)
    grid = grid[:, :, :nTI, :nTE]  # cut off TI short

    if mask is not None:
        mask = mask.astype(bool)
        grid = np.where(mask[:, :, None, None], grid, np.nan)

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(finite, percentile_range)

    # (nx, ny, nTI, nTE) -> (nTI*nx, nTE*ny)
    mosaic = grid.transpose(3, 0, 2, 1).reshape(nTE * nx, nTI * ny)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(
        figsize=(1.7 * nTI + 2.0, 1.7 * nTE + 1.4),
        constrained_layout=True
    )
    im = ax.imshow(mosaic, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_axis_off()
    
    # Column labels (TE) along top
    for j in range(nTI):
        #text on the bottom of each column
        x = (j + 0.5) / nTI
        ax.text(
            x, -0.05, f"{TI[j]*1000:.0f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, 
        )
        if j == nTI//2:
            ax.text(
                x, -0.1, "TI (ms)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=12, fontweight='bold', 
            )
    for j in range(nTI+1):
        ax.axvline((j ) * ny - 0.5, color="white", lw=0.5, linestyle='--')

    # Row labels (TI) along left (TI[0] at bottom because origin='lower')
    for i in range(nTE):
        y = (i + 0.5) / nTE
        ax.text(
            -0.01, y, f"{TE[i]*1000:.0f}",
            transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=8
        )
    
        if i == nTE//2:
            ax.text(
                -0.02, y, "TE (ms)",
                transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=12, fontweight='bold'
            )
    for i in range(nTE+1):
        ax.axhline((i ) * nx - 0.5, color="white", lw=0.5, linestyle='--')

    if title:
        ax.set_title(title, pad=18)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.label.set_color("white")
    cbar.set_label("Simulated ASL signal (a.u.)")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()
    return fig

def plot_parameter_maps_clean(
    ATT_lsq, CBF_lsq, Texch_lsq, MSE_lsq,
    ATT_nn,  CBF_nn,  Texch_nn,  MSE_nn,
    brain_outline_mask=None,
    title="Parameter maps (LSQ vs PI-NN)",
    savebase="parameter_maps_clean",
    robust_percentiles=(2, 98)
):
    """
    Professional 2x4 panel figure:
    Rows: LSQ, PI-NN
    Cols: ATT, CBF, Texch, MSE
    - Shared robust scaling per parameter across rows
    - One shared colorbar per column across both rows
    - NaNs in white
    - Optional brain outline contour (recommended)
    """
    apply_academic_style()

    def robust_limits(a, b):
        x = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
        if x.size == 0:
            return 0.0, 1.0
        return np.percentile(x, robust_percentiles)

    att_vmin, att_vmax = robust_limits(ATT_lsq, ATT_nn)*1000
    cbf_vmin, cbf_vmax = robust_limits(CBF_lsq, CBF_nn)
    tex_vmin, tex_vmax = robust_limits(Texch_lsq, Texch_nn)*1000
    mse_vmin, mse_vmax = robust_limits(MSE_lsq, MSE_nn)

    cmap_param = plt.cm.magma.copy()
    cmap_param.set_bad("black")
    cmap_mse = plt.cm.viridis.copy()
    cmap_mse.set_bad("black")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)

    # Row labels (margin)
    fig.text(0.01, 0.73, "OLS", rotation=90, va="center", ha="left",
             fontsize=16, fontweight="bold")
    fig.text(0.01, 0.27, "PI-NN", rotation=90, va="center", ha="left",
             fontsize=16, fontweight="bold")

    # Plot LSQ row
    im_att_lsq = axes[0, 0].imshow(ATT_lsq*1000,   cmap=cmap_param, vmin=att_vmin, vmax=att_vmax, origin="lower")
    im_cbf_lsq = axes[0, 1].imshow(CBF_lsq,   cmap=cmap_param, vmin=cbf_vmin, vmax=cbf_vmax, origin="lower")
    im_tex_lsq = axes[0, 2].imshow(Texch_lsq*1000, cmap=cmap_param, vmin=tex_vmin, vmax=tex_vmax, origin="lower")
    im_mse_lsq = axes[0, 3].imshow(MSE_lsq,   cmap=cmap_mse,   vmin=mse_vmin, vmax=mse_vmax, origin="lower")

    # Plot PI-NN row
    im_att_nn  = axes[1, 0].imshow(ATT_nn*1000,   cmap=cmap_param, vmin=att_vmin, vmax=att_vmax, origin="lower")
    im_cbf_nn  = axes[1, 1].imshow(CBF_nn,   cmap=cmap_param, vmin=cbf_vmin, vmax=cbf_vmax, origin="lower")
    im_tex_nn  = axes[1, 2].imshow(Texch_nn*1000, cmap=cmap_param, vmin=tex_vmin, vmax=tex_vmax, origin="lower")
    im_mse_nn  = axes[1, 3].imshow(MSE_nn,   cmap=cmap_mse,   vmin=mse_vmin, vmax=mse_vmax, origin="lower")

    # Column titles (top row only)
    axes[0, 0].set_title("ATT", fontsize=16, fontweight='bold')
    axes[0, 1].set_title("CBF", fontsize=16, fontweight='bold')
    axes[0, 2].set_title(r"T$_{exch}$", fontsize=16, fontweight='bold')
    axes[0, 3].set_title("MSE", fontsize=16, fontweight='bold')

    # Optional brain outline on all panels
    if brain_outline_mask is not None:
        outline = brain_outline_mask.astype(float)
        for ax in axes.ravel():
            ax.contour(outline, levels=[0.5], linewidths=0.6, colors="k")

    # Clean axes
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Shared colorbars per column (span both rows)
    #set colorbar text to white
    cbar = fig.colorbar(im_att_lsq, ax=[axes[0, 0], axes[1, 0]], shrink=0.9, pad=0.01, label="ATT (ms)")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    cbar = fig.colorbar(im_cbf_lsq, ax=[axes[0, 1], axes[1, 1]], shrink=0.9, pad=0.01, label="CBF (mL/100g/min)")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    cbar = fig.colorbar(im_tex_lsq, ax=[axes[0, 2], axes[1, 2]], shrink=0.9, pad=0.01, label=r"T$_{exch}$ (ms)")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    cbar = fig.colorbar(im_mse_lsq, ax=[axes[0, 3], axes[1, 3]], shrink=0.9, pad=0.01, label="MSE")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    fig.suptitle(title, fontsize=13)

    # Publication-friendly exports
    fig.savefig(f"{savebase}.pdf", bbox_inches="tight")
    fig.savefig(f"{savebase}.png", bbox_inches="tight", dpi=300)

    plt.show()
    return fig

def plot_masks_figure(gm_mask, wm_mask, title="Tissue masks", savebase="masks"):
    """
    gm_mask, wm_mask: 2D boolean arrays (already rotated to display orientation)
    Creates a 1x3 figure: GM, WM, GM+WM overlay.
    """
    apply_academic_style()

    gm = gm_mask.astype(float)
    wm = wm_mask.astype(float)
    gm[~np.isfinite(gm)] = np.nan
    wm[~np.isfinite(wm)] = np.nan
    gm[gm == 0] = np.nan
    wm[wm == 0] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

    # Binary display colormap with white background
    cmap_bin = plt.cm.gray.copy()
    cmap_bin.set_bad("black")

    #make black background for binary masks
    # GM mask
    axes[0].imshow(np.zeros_like(gm), cmap=cmap_bin, vmin=0, vmax=1, origin="lower")  # white background
    axes[0].imshow(gm, cmap="gist_gray", vmin=0, vmax=1, origin="lower")
    axes[0].set_title("Grey matter", fontsize=12, fontweight='bold')
    # Put [A] on the top left corner
    axes[0].text(0.05, 0.95, "[A]", transform=axes[0].transAxes, fontsize=12, fontweight='bold', color='white', va='top')
    # WM mask
    axes[1].imshow(np.zeros_like(wm), cmap=cmap_bin, vmin=0, vmax=1, origin="lower")  # white background
    axes[1].imshow(wm, cmap="gist_gray", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("White matter", fontsize=12, fontweight='bold')
    # Put [B] on the top left corner
    axes[1].text(0.05, 0.95, "[B]", transform=axes[1].transAxes, fontsize=12, fontweight='bold', color='white', va='top')
    #set title color to white
    axes[0].title.set_color("white")
    axes[1].title.set_color("white")

    # Overlay: GM (red) + WM (blue)
    axes[2].imshow(np.zeros_like(gm), cmap=cmap_bin, vmin=0, vmax=1, origin="lower")  # white background
    axes[2].imshow(gm, cmap="Reds",  vmin=0, vmax=1, alpha=0.75, origin="lower")
    axes[2].imshow(wm, cmap="Blues", vmin=0, vmax=1, alpha=0.75, origin="lower")
    axes[2].set_title("Overlay")

    legend_handles = [
        Patch(facecolor=plt.cm.Reds(0.7),  edgecolor="none", label="Grey matter", ),
        Patch(facecolor=plt.cm.Blues(0.7), edgecolor="none", label="White matter"),
    ]
    axes[2].legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.savefig(f"{savebase}.pdf", bbox_inches="tight")
    fig.savefig(f"{savebase}.png", bbox_inches="tight", dpi=300)
    plt.show()

    return fig

def predict_and_plot_brain_slice(
    simulated_data_file, pi_nn_model_file,
    gm_file, wm_file,
    TE, TI, ntes, arg, z_slice_pos=0.66,
    mosaic_swap_ti_te=False
):
    apply_academic_style()

    # Load network
    net = Net(TE, TI, arg.net_pars, arg.other_params)
    net.load_state_dict(torch.load(pi_nn_model_file, map_location=torch.device("cpu")))
    net.eval()

    # LSQ coefficients
    beta = np.loadtxt("lsq_regression_coefficients_f.txt")  # (nTE*nTI + 1, 3)

    # Load simulated data (expected shape: (nx, ny, nTI*nTE))
    data_img = nib.load(simulated_data_file)
    data_sim = data_img.get_fdata().astype(np.float32)

    TE = np.asarray(TE, dtype=float).ravel()
    TI = np.asarray(TI, dtype=float).ravel()
    nTI, nTE = len(TI), len(TE)

    # Masks
    gm_mask, wm_mask = matter_masks(gm_file, wm_file)
    nx, ny, nz = gm_mask.shape
    slice_z = int(np.floor(z_slice_pos * nz))

    gm_slice = gm_mask[:, :, slice_z].astype(bool)
    wm_slice = wm_mask[:, :, slice_z].astype(bool)
    combined_mask = gm_slice | wm_slice

    # Extract masked voxel timecourses for inference
    data_masked = data_sim[combined_mask, :]

    # PI-NN prediction
    with torch.no_grad():
        data_tensor = torch.tensor(data_masked, dtype=torch.float32)
        _, ATT_masked, CBF_masked, Texch_masked = net(data_tensor, G_or_W_bool=torch.zeros(data_tensor.shape[0], dtype=torch.bool), synthesize=False)

    ATT_pred = np.full((nx, ny), np.nan, dtype=np.float32)
    CBF_pred = np.full((nx, ny), np.nan, dtype=np.float32)
    Texch_pred = np.full((nx, ny), np.nan, dtype=np.float32)
    ATT_pred[combined_mask] = ATT_masked.cpu().numpy().ravel()
    CBF_pred[combined_mask] = CBF_masked.cpu().numpy().ravel()
    Texch_pred[combined_mask] = Texch_masked.cpu().numpy().ravel()

    # LSQ baseline
    data_aug = np.hstack([data_masked, np.ones((data_masked.shape[0], 1), dtype=np.float32)])
    y_lsq = data_aug @ beta

    ATT_lsq = np.full((nx, ny), np.nan, dtype=np.float32)
    CBF_lsq = np.full((nx, ny), np.nan, dtype=np.float32)
    Texch_lsq = np.full((nx, ny), np.nan, dtype=np.float32)
    ATT_lsq[combined_mask] = y_lsq[:, 0]
    CBF_lsq[combined_mask] = y_lsq[:, 1]
    Texch_lsq[combined_mask] = y_lsq[:, 2]

    # Ground-truth maps used for MSE (as per your earlier logic)
    ATT_true = np.full((nx, ny), np.nan, dtype=np.float32)
    CBF_true = np.full((nx, ny), np.nan, dtype=np.float32)
    Texch_true = np.full((nx, ny), np.nan, dtype=np.float32)

    ATT_true[gm_slice] = 1.28
    CBF_true[gm_slice] = 60.0
    Texch_true[gm_slice] = 0.228
    
    ATT_true[wm_slice] = 1.69
    CBF_true[wm_slice] = 22.0
    Texch_true[wm_slice] = 0.127

    
        # ----------------------------
    # Equal-scaled (dimensionless) per-voxel MSE
    # ----------------------------
    valid = combined_mask  # voxels where truth is defined

    # Use a per-parameter scale so each term contributes comparably.
    # Here: scale = (max - min) of the ground-truth values present in this slice.
    # If you prefer fixed scales, replace these with global training ranges/bounds.
    eps = 1e-8
    att_scale   = float(np.nanmax(ATT_true[valid])   - np.nanmin(ATT_true[valid]))
    cbf_scale   = float(np.nanmax(CBF_true[valid])   - np.nanmin(CBF_true[valid]))
    texch_scale = float(np.nanmax(Texch_true[valid]) - np.nanmin(Texch_true[valid]))

    att_scale   = max(att_scale, eps)
    cbf_scale   = max(cbf_scale, eps)
    texch_scale = max(texch_scale, eps)

    # Normalized squared errors (dimensionless)
    att_se_net   = ((ATT_pred   - ATT_true)   / att_scale) ** 2
    cbf_se_net   = ((CBF_pred   - CBF_true)   / cbf_scale) ** 2
    texch_se_net = ((Texch_pred - Texch_true) / texch_scale) ** 2

    att_se_lsq   = ((ATT_lsq   - ATT_true)   / att_scale) ** 2
    cbf_se_lsq   = ((CBF_lsq   - CBF_true)   / cbf_scale) ** 2
    texch_se_lsq = ((Texch_lsq - Texch_true) / texch_scale) ** 2

    # Mean over parameters -> equal weighting across ATT/CBF/Texch
    MSE_net = np.nanmean(np.stack([att_se_net, cbf_se_net, texch_se_net], axis=0), axis=0).astype(np.float32)
    MSE_lsq = np.nanmean(np.stack([att_se_lsq, cbf_se_lsq, texch_se_lsq], axis=0), axis=0).astype(np.float32)


    # Rotate for visual orientation (keep consistent across all maps)
    data_sim_rot = np.rot90(data_sim, axes=(0, 1))
    gm_rot = np.rot90(gm_slice)
    wm_rot = np.rot90(wm_slice)
    ATT_lsq_rot = np.rot90(ATT_lsq)
    CBF_lsq_rot = np.rot90(CBF_lsq)
    Texch_lsq_rot = np.rot90(Texch_lsq)
    ATT_pred_rot = np.rot90(ATT_pred)
    CBF_pred_rot = np.rot90(CBF_pred)
    Texch_pred_rot = np.rot90(Texch_pred)
    MSE_lsq_rot = np.rot90(MSE_lsq)
    MSE_net_rot = np.rot90(MSE_net)
    combined_rot = np.rot90(combined_mask)
    
    """
    # ----------------------------
    # Tissue masks figure
    # ----------------------------
    plot_masks_figure(
        gm_mask=gm_rot,
        wm_mask=wm_rot,
        title="Tissue masks (loaded file; masked)",
        savebase="tissue_masks",
    )

    # ----------------------------
    # (1) TI/TE mosaic (still called)
    # ----------------------------
    plot_ti_te_mosaic(
        data_sim_slice=data_sim_rot,
        TE=TE, TI=TI,
        mask=combined_rot,
        title="Simulated ASL signal across TI/TE (loaded file; masked)",
        savepath="ti_te_mosaic.png",
    )
    """
    # ----------------------------
    # (2) Clean professional parameter maps (only parameter maps)
    # ----------------------------
    plot_parameter_maps_clean(
        ATT_lsq=ATT_lsq_rot, CBF_lsq=CBF_lsq_rot, Texch_lsq=Texch_lsq_rot, MSE_lsq=MSE_lsq_rot,
        ATT_nn=ATT_pred_rot, CBF_nn=CBF_pred_rot, Texch_nn=Texch_pred_rot, MSE_nn=MSE_net_rot,
        brain_outline_mask=combined_rot,  # outline makes it look “paper-ready”
        title="Parameter estimation on simulated ASL slice",
        savebase="parameter_maps_clean"
    )

    return None
