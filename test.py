import simulations
from simulations import simulated_brain_data, predict_and_plot_brain_slice
import PI_NN  
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from PI_NN import Net
from LSQ_Regression import train_least_squares_from_simulator
from torch_model_multi_te import torch_deltaM_multite_model_batch, torch_deltaM_multite_model_single

class Dot:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# build the arg object the net expects
arg = Dot(
    net_pars=Dot(
        depth=3,
        width=128,
        dropout=0.0,
        batch_norm=False,
        con="sigmoid",          # bounded outputs
        cons_min=[0.5, 5, 0.05],    # ATT_min, CBF_min, texch_min
        cons_max=[3.0, 100, 0.5],    # ATT_max, CBF_max, texch_max
    ),
    other_params=Dot(
        width=128,
        ntes=None,              # let the net infer ntes = [len(TE)] * len(PLD)
        GM_T1=1.615,
        WM_T1=0.911,
        T1a=1.65,
        GM_T2=0.083,
        WM_T2=0.075,
        T2a=0.150,
        itt=0.2,
        lambd=0.9,
        alpha=0.68,
    ),
    train_pars=Dot(
        loss_function="MSE",
        optimizer="Adam",
        learning_rate=1e-3,
        batch_size=32,
        validation_split=0.2,
        epochs=1000,
        phys_frac=0.95,
        patience=100,
        physics_informed=True,
    ),
)

# simulator settings
sim_kwargs = dict(
    rician=True,
    SNR= 10,
    TE= np.arange(0.01, 0.21, 0.04), 
    TI = np.arange(1.0, 7.0, 0.25),
    ntes = None,
    tissue="both",
    sims=10000,
    distribution="uniform",
    z_slice_pos=0.66,
)

def make_training_data_demo():
    # generate training data and save to files
    sim_kwargs_local = sim_kwargs.copy()
    sim_kwargs_local["save_files"] = True
    X, y, X_clean, G_or_W_mask = simulations.make_training_data(**sim_kwargs_local)
    print("Generated training data:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X_clean shape:", X_clean.shape)
    return X, y, X_clean, G_or_W_mask

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def training_data_fig():
    local_sim_kwargs = sim_kwargs.copy()
    local_sim_kwargs["sims"] = 2  # single voxel
    local_sim_kwargs["SNR"] = 2
    local_sim_kwargs["TE"] = np.arange(0.01, 0.21, 0.04)
    local_sim_kwargs["TI"] = np.arange(0.5, 7, 0.25)

    X, y, X_clean, G_or_W_mask = simulations.make_training_data(**local_sim_kwargs)
    TE = local_sim_kwargs["TE"]
    TI = local_sim_kwargs["TI"]
    nTE = len(TE)
    nTI = len(TI)

    # reshape so row j corresponds to TE[j]
    X = X[0, :nTE * nTI].reshape(nTI, nTE).T
    X_clean = X_clean[0, :nTE * nTI].reshape(nTI, nTE).T

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    TE = TE*1000  # convert to ms for plotting
    TI = TI*1000  # convert to ms for plotting

    # draw from back (smallest TE) to front (largest TE)
    for j in range(nTE):
        te = TE[j]
        xs = TI
        ys = np.full_like(xs, te)
        zs_noisy = X[j, :]
        zs_clean = X_clean[j, :]

        # --- fill under the clean curve (floor at z = 0.0) ---
        verts = [list(zip(xs, ys, zs_clean)) +
                 list(zip(xs[::-1], ys[::-1], np.zeros_like(xs)))]

        poly = Poly3DCollection(verts,
                                facecolor='gray',
                                alpha=0.3)
        poly.set_zorder(0)         # behind lines
        ax.add_collection3d(poly)

        # --- lines on top of the fill ---
        ax.plot(xs, ys, zs_noisy,
                color='black', alpha=0.7,
                zorder=10,
                label=f'Noisy Simulated Signal' if j == 0 else None, linestyle='dashed')
        ax.plot(xs, ys, zs_clean,
                color='black', alpha=1,
                zorder=10, linewidth=2.0, label=f'Noise-free Simulated Signal' if j == 0 else None)

    # axis formatting
    ax.set_ylim(TE.max(), TE.min())  # so that smallest TI is at back
    ax.set_xlabel('TI (ms)')
    ax.set_ylabel('TE (ms)')
    ax.set_zlabel('Signal Intensity (a.u.)', labelpad=10)
    ax.set_title('Simulated ASL Signal across TE and TI')

    # floor at 0
    ax.set_zlim(0.0, 1.05 * max(np.max(X), np.max(X_clean)))
    ax.set_yticks(TE[::-1])
    ax.set_yticklabels([f"{te:.2f}" for te in TE[::-1]])
    ax.view_init(elev=25, azim=-44)
    ax.set_box_aspect([1,1.3,0.7])  # aspect ratio
    #move legend outside plot numerically
    ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 0.7))
    # set axes to 2 decimal places
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.2g'))


    plt.show()

    print("Figure generated.")
    print("Corresponding parameters:")
    print(f"ATT: {y[0,0]:.3f} s, CBF: {y[0,1]:.3f} ml/100g/min, Texch: {y[0,2]:.3f} s")
    return fig

def net_train_demo():
    # initialize and train the PI-NN
    net, train_hist, val_hist = PI_NN.train_from_simulator(sim_kwargs, arg, simulations)
    
    # save the trained model
    torch.save(net.state_dict(), "pi_nn_asl_gm.pt")

    # Make a txt file for all the parameters used in training
    with open("pi_nn_asl_params_gm.txt", "w") as f:
        f.write("Simulator settings:\n")
        for key, value in sim_kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("\nNetwork parameters:\n")
        for key, value in arg.net_pars.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\nOther parameters:\n")
        for key, value in arg.other_params.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTraining parameters:\n")
        for key, value in arg.train_pars.__dict__.items():
            f.write(f"{key}: {value}\n")
    print("Training complete.")

    plt.figure()
    plt.plot(train_hist, label="Train loss")
    plt.plot(val_hist, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation loss")
    plt.tight_layout()
    plt.show()
    
    return net

def lsq_regression_demo():
    # train least-squares baseline
    beta, _, _ = train_least_squares_from_simulator(sim_kwargs, arg, simulations)

    # Save the beta coefficients to a text file
    np.savetxt("lsq_regression_coefficients_gm.txt", beta)
    return None

def generate_brain_slices_demo():
    """Small demo run of sim_signal.

    Runs `sim_signal` with a very small `sims` value.
    """
    simulated_brain_data(
        grey_matter_file_name='c1sub-sub1_run-1_T1w.nii',
        white_matter_file_name='c2sub-sub1_run-1_T1w.nii',
        TE=sim_kwargs['TE'],
        TI=sim_kwargs['TI'],
        ntes=sim_kwargs['ntes'],
        SNR=sim_kwargs['SNR'],
        rician=False,
        z_slice_pos=sim_kwargs['z_slice_pos'],
    )
    return None

def test_predict_and_plot_brain_slice():

    """Small demo run of predict_and_plot_brain_slice.

    Runs `predict_and_plot_brain_slice` on simulated data.
    """
    predict_and_plot_brain_slice(
        simulated_data_file='simulated_asl_data.nii',
        pi_nn_model_file='pi_nn_asl_f2.pt',
        gm_file='c1sub-sub1_run-1_T1w.nii',
        wm_file='c2sub-sub1_run-1_T1w.nii',
        TE=sim_kwargs['TE'],
        TI=sim_kwargs['TI'],
        ntes=sim_kwargs['ntes'],
        arg=arg,
        z_slice_pos=sim_kwargs['z_slice_pos'],
    )


    return None

def test_models_on_simulated_data():
    local_sim_kwargs = sim_kwargs.copy()
    local_sim_kwargs["sims"] = 500  # number of test samples
    local_sim_kwargs["SNR"] = 2         # moderate noise
    local_sim_kwargs["tissue"] = "both"
    # 1) Simulate test data
    X_test, y_test, _, _ = simulations.make_training_data(**local_sim_kwargs)  # X_test: (sims, nTE, nTI)

    ATT_true   = y_test[:, 0]
    CBF_true   = y_test[:, 1]
    Texch_true = y_test[:, 2]

    # Flatten X_test to match training representation: (sims, nTE*nTI)
    n_sims = X_test.shape[0]

    # 2) Load trained PI-NN model
    net = Net(sim_kwargs["TE"], sim_kwargs["TI"], arg.net_pars, arg.other_params)
    net.load_state_dict(torch.load("pi_nn_asl_f2.pt", map_location="cpu"))
    net.eval()

    # Convert test data to tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Predict parameters with PI-NN
    with torch.no_grad():
        _, ATT_pred_net, CBF_pred_net, Texch_pred_net = net(X_test_tensor, None, synthesize=False)

    ATT_pred_net   = ATT_pred_net.cpu().numpy().ravel()
    CBF_pred_net   = CBF_pred_net.cpu().numpy().ravel()
    Texch_pred_net = Texch_pred_net.cpu().numpy().ravel()

    # 3) Least-squares regression baseline
    beta = np.loadtxt("lsq_regression_coefficients_f.txt")  # shape: (nTE*nTI + 1, 3)

    # Add bias column to flattened X_test
    X_test_aug = np.hstack([X_test.reshape(n_sims, -1), np.ones((n_sims, 1))])  # (n_sims, nTE*nTI + 1)

    # Compute LSQ predictions
    y_pred_lsq = X_test_aug @ beta  # (n_sims, 3)
    ATT_pred_lsq   = y_pred_lsq[:, 0]
    CBF_pred_lsq   = y_pred_lsq[:, 1]
    Texch_pred_lsq = y_pred_lsq[:, 2]

    #Alter ATT, Texch to ms for error analysis
    ATT_true   = ATT_true * 1000.0
    Texch_true = Texch_true * 1000.0
    ATT_pred_net   = ATT_pred_net * 1000.0
    Texch_pred_net = Texch_pred_net * 1000.0
    ATT_pred_lsq   = ATT_pred_lsq * 1000.0
    Texch_pred_lsq = Texch_pred_lsq * 1000.0

    # Error analysis
    ATT_err_net   = ATT_pred_net   - ATT_true
    CBF_err_net   = CBF_pred_net   - CBF_true
    Texch_err_net = Texch_pred_net - Texch_true

    ATT_err_lsq   = ATT_pred_lsq   - ATT_true
    CBF_err_lsq   = CBF_pred_lsq   - CBF_true
    Texch_err_lsq = Texch_pred_lsq - Texch_true

    ATT_mpe_net   = np.mean(np.abs(ATT_err_net   / ATT_true)) * 100.0
    CBF_mpe_net   = np.mean(np.abs(CBF_err_net   / CBF_true)) * 100.0
    Texch_mpe_net = np.mean(np.abs(Texch_err_net / Texch_true)) * 100.0 

    ATT_mpe_lsq   = np.mean(np.abs(ATT_err_lsq   / ATT_true)) * 100.0
    CBF_mpe_lsq   = np.mean(np.abs(CBF_err_lsq   / CBF_true)) * 100.0
    Texch_mpe_lsq = np.mean(np.abs(Texch_err_lsq / Texch_true)) * 100.0

    # ---------------------------------------------------------------------
    # Global plotting style: black & white, serif
    # ---------------------------------------------------------------------

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # Pack parameters + MPEs in a convenient structure
    params = [
        ("ATT",   ATT_true,   ATT_pred_lsq,   ATT_pred_net,   "s",
         ATT_mpe_lsq,   ATT_mpe_net),
        ("CBF",   CBF_true,   CBF_pred_lsq,   CBF_pred_net,   "ml/100g/min",
         CBF_mpe_lsq,   CBF_mpe_net),
        ("Texch", Texch_true, Texch_pred_lsq, Texch_pred_net, "s",
         Texch_mpe_lsq, Texch_mpe_net),
    ]

    # ---------------------------------------------------------------------
    # 4) 4×3 grid:
    #    columns = ATT / CBF / Texch
    #    rows    = LSQ scatter / LSQ BA / PI-NN scatter / PI-NN BA
    # ---------------------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 3)) # LSQ plots
    fig2 = plt.figure(figsize=(14, 3)) # PI-NN plots

    gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.03, wspace=0.24, top=0.93, bottom=0.18)
    axes1 = np.empty((2, 3), dtype=object)
    for col in range(3):
        axes1[0, col] = fig1.add_subplot(gs1[0, col])  # LSQ scatter
        axes1[1, col] = fig1.add_subplot(gs1[1, col])  # LSQ BA

    gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.03, wspace=0.24, top=0.93, bottom=0.18)
    axes2 = np.empty((2, 3), dtype=object)
    for col in range(3):
        axes2[0, col] = fig2.add_subplot(gs2[0, col])  # PI-NN scatter
        axes2[1, col] = fig2.add_subplot(gs2[1, col], sharey=axes1[1, col])  # PI-NN BA

    marker_kwargs = dict(
        s=8,
        alpha=0.6,
        facecolors='none',
        edgecolors='k',
        marker='o',
    )

    for col, (name, y_true, y_pred_lsq, y_pred_net, unit,
              mpe_lsq, mpe_net) in enumerate(params):

        y_max = max(np.max(y_true), np.max((y_true+y_pred_lsq)/2))
        # Column title on top row

        # --------------------------------------------------------------
        # Row 0: LSQ scatter (True vs LSQ predicted)
        # --------------------------------------------------------------
        ax_sc_lsq = axes1[0, col]
        ax_sc_lsq.scatter(y_true, y_pred_lsq, **marker_kwargs)
        ax_sc_lsq.plot([0, y_max], [0, y_max],
                       linestyle='dotted', linewidth=1.0, color='black')
        ax_sc_lsq.xaxis.set_label_position('top')
        ax_sc_lsq.xaxis.set_ticks_position('top')
        ax_sc_lsq.tick_params(top=True, labeltop=True)
        ax_sc_lsq.set_xlabel("Ground Truth", fontsize=11)
        
        # Only label y-axis on left column to reduce clutter
        if col == 0:
            ax_sc_lsq.set_ylabel("Predicted", fontsize=11)
        else:
            ax_sc_lsq.set_ylabel("")

        # No x-label on scatter rows (0 and 2) to avoid crowding;
        # we let the Bland–Altman rows carry the x-axis labels.
       
        ax_sc_lsq.tick_params(labelsize=8, bottom=False)
        ax_sc_lsq.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        """
        ax_sc_lsq.text(
            0.04, 0.96,
            f"MPE = {mpe_lsq:.2f}%",
            transform=ax_sc_lsq.transAxes,
            ha="left", va="top", fontsize=8
        )"""

        # --------------------------------------------------------------
        # Row 1: LSQ Bland–Altman
        # --------------------------------------------------------------
        ax_ba_lsq = axes1[1, col]
        
        mean_diff, sd_diff, loa_upper, loa_lower = bland_altman_plot(
            y_true, y_pred_lsq)
        ax_ba_lsq.axhline(0, color='black', linestyle='dotted', linewidth=1.0)
        ax_ba_lsq.plot([0, y_max], [0, 0], linewidth=0, markersize=0)  # dummy plot for consistent axes
        ax_ba_lsq.axhline(loa_upper, color='black', linestyle='dashed', linewidth=1.0)
        ax_ba_lsq.axhline(loa_lower, color='black', linestyle='dashed', linewidth=1.0)
        ax_ba_lsq.set_xlabel(f"Mean of ground truth and predicted", fontsize=11)
        if col == 0:
            ax_ba_lsq.set_ylabel("Difference", fontsize=11)
        else:
            ax_ba_lsq.set_ylabel("")
        ax_ba_lsq.scatter(
            (y_true + y_pred_lsq) / 2.0, (y_pred_lsq - y_true), **marker_kwargs)
        ax_ba_lsq.tick_params(labelsize=8)
        ax_ba_lsq.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        # share x axis between scatter and BA plots
        ax_ba_lsq.get_shared_x_axes().joined(ax_ba_lsq, ax_sc_lsq)

        # Set LQS label on the left between two plots
        if col == 0:
            sc_pos = ax_sc_lsq.get_position()
            ba_pos = ax_ba_lsq.get_position()
            fig1.text(
                sc_pos.x0 - 0.06,
            (sc_pos.y0 + ba_pos.y1) / 2,
            "OLS",
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold"
        )
        

        #
        # --------------------------------------------------------------
        # Row 2 PINN scatter (True vs PINN predicted)
        # --------------------------------------------------------------
        ax_sc_net = axes2[0, col]
        ax_sc_net.scatter(y_true, y_pred_net, **marker_kwargs)
        ax_sc_net.plot([0, y_max], [0, y_max],
                       linestyle='dotted', linewidth=1.0, color='black')
        ax_sc_net.xaxis.set_label_position('top')
        ax_sc_net.xaxis.set_ticks_position('top')
        ax_sc_net.tick_params(top=True, labeltop=True)
        ax_sc_net.set_xlabel("Ground Truth", fontsize=11)
        
        # Only label y-axis on left column to reduce clutter
        if col == 0:
            ax_sc_net.set_ylabel("Predicted", fontsize=11)
        else:
            ax_sc_net.set_ylabel("")

        # No x-label on scatter rows (0 and 2) to avoid crowding;
        # we let the Bland–Altman rows carry the x-axis labels.
        ax_sc_net.tick_params(labelsize=8, bottom=False)
        ax_sc_net.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        """
        ax_sc_net.text(
            0.04, 0.96,
            f"MPE = {mpe_net:.2f}%",
            transform=ax_sc_net.transAxes,
            ha="left", va="top", fontsize=8
        )"""

        # --------------------------------------------------------------
        # Row 3: PINN Bland–Altman
        # --------------------------------------------------------------
        ax_ba_net = axes2[1, col]
        mean_diff, sd_diff, loa_upper, loa_lower = bland_altman_plot(
            y_true, y_pred_net)
        ax_ba_net.plot([0, y_max], [0, 0], linewidth=0, markersize=0)  # dummy plot for consistent axes
        ax_ba_net.axhline(0, color='black', linestyle='dotted', linewidth=1.0)
        ax_ba_net.axhline(loa_upper, color='black', linestyle='dashed', linewidth=1.0)
        ax_ba_net.axhline(loa_lower, color='black', linestyle='dashed', linewidth=1.0)
        ax_ba_net.set_xlabel(f"Mean of ground truth and predicted", fontsize=11)
        if col == 0:
            ax_ba_net.set_ylabel("Difference", fontsize=11)
        else:
            ax_ba_net.set_ylabel("")
        ax_ba_net.scatter(
            (y_true + y_pred_net) / 2.0, (y_pred_net - y_true), **marker_kwargs)
        ax_ba_net.tick_params(labelsize=8)
        ax_ba_net.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        if col == 0:
            sc_pos = ax_sc_net.get_position()
            ba_pos = ax_ba_net.get_position()
            fig2.text(
                sc_pos.x0 - 0.06,
            (sc_pos.y0 + ba_pos.y1) / 2,
            "PINN",
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold"
        )

    #SHOW FIGURES
    print("Absolute error numerical results:\n")
    print(f"ATT MAE LSQ: {np.mean(np.abs(ATT_err_lsq)):.4f} ms, PINN: {np.mean(np.abs(ATT_err_net)):.4f} ms")
    print(f"CBF MAE LSQ: {np.mean(np.abs(CBF_err_lsq)):.4f} ml/100g/min, PINN: {np.mean(np.abs(CBF_err_net)):.4f} ml/100g/min")
    print(f"Texch MAE LSQ: {np.mean(np.abs(Texch_err_lsq)):.4f} ms, PINN: {np.mean(np.abs(Texch_err_net)):.4f} ms")

    print("Scatter numerical results:")
    print(f"ATT MPE LSQ: {ATT_mpe_lsq:.2f} %, PINN: {ATT_mpe_net:.2f} %")
    print(f"CBF MPE LSQ: {CBF_mpe_lsq:.2f} %, PINN: {CBF_mpe_net:.2f} %")
    print(f"Texch MPE LSQ: {Texch_mpe_lsq:.2f} %, PINN: {Texch_mpe_net:.2f} %")

    print("Bland–Altman numerical results:")
    print("LSQ:")
    print(f"ATT mean diff: {np.mean(ATT_err_lsq):.4f} ms, SD: {np.std(ATT_err_lsq):.4f} s")
    print(f"CBF mean diff: {np.mean(CBF_err_lsq):.4f} ml/100g/min, SD: {np.std(CBF_err_lsq):.4f} ml/100g/min")
    print(f"Texch mean diff: {np.mean(Texch_err_lsq):.4f} ms, SD: {np.std(Texch_err_lsq):.4f} ms")
    print("PINN:")
    print(f"ATT mean diff: {np.mean(ATT_err_net):.4f} ms, SD: {np.std(ATT_err_net):.4f} s")
    print(f"CBF mean diff: {np.mean(CBF_err_net):.4f} ml/100g/min, SD: {np.std(CBF_err_net):.4f} ml/100g/min")
    print(f"Texch mean diff: {np.mean(Texch_err_net):.4f} ms, SD: {np.std(Texch_err_net):.4f} ms")


    #export figures as pdf images
    fig1.savefig("lsq_regression_results.png", dpi=1000, bbox_inches='tight', pad_inches=0.02)
    fig2.savefig("pi_nn_regression_results.png", dpi=1000, bbox_inches='tight', pad_inches=0.02)
    
    # stack figures vertically and show
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.02, "top": 0.95})

    ax[0].imshow(plt.imread("lsq_regression_results.png"))
    ax[0].axis("off")
    ax[1].imshow(plt.imread("pi_nn_regression_results.png"))
    ax[1].axis("off")
    # Optional extra tightening of outer margins
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    #column labels
    fig.text(0.21, 0.97, "ATT (ms)", ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(0.535, 0.97, "CBF (ml/100g/min)", ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(0.8625, 0.97, "Texch (ms)", ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.show()

    return None

def model_signal_regeneration_test():
    """Test to visualise the simulated ASL signal for given parameters."""

    sim_kwargs_local = sim_kwargs.copy()
    sim_kwargs_local["state"] = 42  # for reproducibility
    sim_kwargs_local["sims"] = 500  # simulate just one voxel
    sim_kwargs_local["SNR"] = 2  
    sim_kwargs_local["tissue"] = "both"

    X_test, y, X_test_clean, G_or_W_mask_test = simulations.make_training_data(**sim_kwargs_local)
    scale = np.max(X_test_clean)
    n_sims = X_test.shape[0]
    
    g_or_w_mask_tensor = torch.tensor(G_or_W_mask_test, dtype=torch.float32)
    #----------------------------------------------------------
    # Load trained PI-NN model
    # ---------------------------------------------------------------------
    net = Net(sim_kwargs["TE"], sim_kwargs["TI"], arg.net_pars, arg.other_params)
    net.load_state_dict(torch.load("pi_nn_asl_f2.pt", map_location="cpu"))
    net.eval()

    # ---------------------------------------------------------------------
    # Load least-squares coefficients and predict parameters (LSQ)
    # ---------------------------------------------------------------------
    beta = np.loadtxt("lsq_regression_coefficients_f.txt")  # shape: (nTE*nTI + 1, 3)

    X_test_aug = np.hstack([
        X_test.reshape(n_sims, -1),
        np.ones((n_sims, 1))
    ])  # (n_sims, nTE*nTI + 1)
    y_pred_lsq = X_test_aug @ beta  # (n_sims, 3)

    # For LSQ signal regeneration (torch tensors)
    ATT_pred_lsq_torch   = torch.tensor(y_pred_lsq[:, 0], dtype=torch.float32)
    CBF_pred_lsq_torch   = torch.tensor(y_pred_lsq[:, 1], dtype=torch.float32)
    Texch_pred_lsq_torch = torch.tensor(y_pred_lsq[:, 2], dtype=torch.float32)

    # ---------------------------------------------------------------------
    # Use model to regenerate signal from predicted parameters (LSQ)
    # ---------------------------------------------------------------------
    TE = np.atleast_1d(np.array(sim_kwargs["TE"], dtype=float))   # unique TEs per PLD
    TI = np.atleast_1d(np.array(sim_kwargs["TI"], dtype=float))   # TIs / PLDs

    TI_torch = torch.tensor(TI, dtype=torch.float32)
    nTE = len(TE)
    nTI = len(TI)
    ntes_vec = np.full(nTI, nTE, dtype=np.int32)
    tes_flat = np.tile(TE, nTI)
    TE_torch = torch.tensor(tes_flat, dtype=torch.float32)
    LSQ_signal = torch.zeros((n_sims, nTE * nTI), dtype=torch.float32)

    for i in range(n_sims):
        if G_or_W_mask_test[i] == 1:  # grey matter
            deltaM_lsq = torch_deltaM_multite_model_single(
                TI_torch, TE_torch, ntes_vec,
                ATT_pred_lsq_torch[i], CBF_pred_lsq_torch[i], Texch_pred_lsq_torch[i],
                m0a=1.0, taus=1.5, t1=1.615, t1b=1.65, t2=0.083, t2b=0.150,
                itt=0.2, lambd=0.9, alpha=0.68)
            LSQ_signal[i, :] = deltaM_lsq
        else:  # white matter
            deltaM_lsq = torch_deltaM_multite_model_single(
                TI_torch, TE_torch, ntes_vec,
                ATT_pred_lsq_torch[i], CBF_pred_lsq_torch[i], Texch_pred_lsq_torch[i],
                m0a=1.0, taus=1.5, t1=0.911, t1b=1.65, t2=0.075, t2b=0.150,
                itt=0.2, lambd=0.9, alpha=0.68)
            LSQ_signal[i, :] = deltaM_lsq

    # ---------------------------------------------------------------------
    # Predict parameters with PI-NN and regenerate signal
    # ---------------------------------------------------------------------
    X_test_tensor = torch.tensor(X_test.reshape(n_sims, -1), dtype=torch.float32)
    with torch.no_grad():
        PI_NN_signal, ATT_pred_net, CBF_pred_net, Texch_pred_net = net(
            X_test_tensor, G_or_W_bool = g_or_w_mask_tensor, synthesize=True
        )

    ATT_pred_net   = ATT_pred_net.cpu().numpy().ravel()
    CBF_pred_net   = CBF_pred_net.cpu().numpy().ravel()
    Texch_pred_net = Texch_pred_net.cpu().numpy().ravel()

    # ---------------------------------------------------------------------
    # Calculate absolute mean error between regenerated and clean signals 
    # This is the mean error of all curves, not each point individually
    # ---------------------------------------------------------------------
    X_clean_t = torch.as_tensor(
    X_test_clean,
    dtype=LSQ_signal.dtype,
    device=LSQ_signal.device)

    lsq_abs_err_per_curve = (LSQ_signal - X_clean_t).abs().reshape(n_sims, -1).sum(dim=1)  # (n_sims,)
    avg_abs_err_LSQ = lsq_abs_err_per_curve.mean()  # scalar

    X_clean_t_pinn = X_clean_t.to(device=PI_NN_signal.device, dtype=PI_NN_signal.dtype)
    pinn_abs_err_per_curve = (PI_NN_signal - X_clean_t_pinn).abs().reshape(n_sims, -1).sum(dim=1)  # (n_sims,)
    avg_abs_err_PINN = pinn_abs_err_per_curve.mean()  # scalar


    # ---------------------------------------------------------------------
    # Flatten signals so that each TE segment follows the previous on 1 x-axis
    # ---------------------------------------------------------------------
    TE = sim_kwargs["TE"]
    TI = sim_kwargs["TI"]
    x_axis = np.arange(len(TI))

    X_test_reshape = X_test[0].flatten() 
    X_test_clean_reshape = (X_test_clean[0].flatten()) 
    LSQ_signal_reshape = (LSQ_signal[0].flatten()) 
    PI_NN_signal_reshape = (PI_NN_signal[0].flatten()) 

    X_test_reshape = X_test_reshape.reshape(nTI, nTE).T.flatten()
    X_test_clean_reshape = X_test_clean_reshape.reshape(nTI, nTE).T.flatten()
    LSQ_signal_reshape = LSQ_signal_reshape.reshape(nTI, nTE).T.flatten()
    PI_NN_signal_reshape = PI_NN_signal_reshape.reshape(nTI, nTE).T.flatten()

    X_test_reshape = X_test_reshape[:len(TI)]
    X_test_clean_reshape = X_test_clean_reshape[:len(TI)]
    LSQ_signal_reshape = LSQ_signal_reshape[:len(TI)]
    PI_NN_signal_reshape = PI_NN_signal_reshape[:len(TI)]

    #Residuals
    Residuals_LSQ = (np.asarray(LSQ_signal[0]) - np.asarray(X_test_clean[0])).flatten() 
    Residuals_PINN = (np.asarray(PI_NN_signal[0]) - np.asarray(X_test_clean[0])).flatten() 
    Residuals_LSQ = Residuals_LSQ.reshape(nTI, nTE).T.flatten()
    Residuals_PINN = Residuals_PINN.reshape(nTI, nTE).T.flatten()

    Residuals_LSQ = Residuals_LSQ[:len(TI)]
    Residuals_PINN = Residuals_PINN[:len(TI)]
    

    # ---------------------------------------------------------------------
    # Compute true values and MPE statistics
    # ---------------------------------------------------------------------
    # True parameters for all sims
    ATT_true_all   = y[:, 0]
    CBF_true_all   = y[:, 1]
    Texch_true_all = y[:, 2]

    # Predicted parameters for all sims
    ATT_pred_lsq_all   = y_pred_lsq[:, 0]
    CBF_pred_lsq_all   = y_pred_lsq[:, 1]
    Texch_pred_lsq_all = y_pred_lsq[:, 2]

    ATT_pred_net_all   = ATT_pred_net
    CBF_pred_net_all   = CBF_pred_net
    Texch_pred_net_all = Texch_pred_net

    y_true_all = np.column_stack([ATT_true_all, CBF_true_all, Texch_true_all])
    y_pred_lsq_all = np.column_stack([
        ATT_pred_lsq_all, CBF_pred_lsq_all, Texch_pred_lsq_all
    ])
    y_pred_net_all = np.column_stack([
        ATT_pred_net_all, CBF_pred_net_all, Texch_pred_net_all
    ])

    def mean_absolute_percentage_error(y_true, y_pred):
        # MPE per parameter in percent
        return 100.0 * np.mean(np.abs((y_pred - y_true) / y_true), axis=0)

    mpe_lsq = mean_absolute_percentage_error(y_true_all, y_pred_lsq_all)   # [ATT, CBF, Texch]
    mpe_net = mean_absolute_percentage_error(y_true_all, y_pred_net_all)   # [ATT, CBF, Texch]

    # True + predicted values for the voxel that is plotted (first one)
    ATT_true_voxel   = ATT_true_all[0]
    CBF_true_voxel   = CBF_true_all[0]
    Texch_true_voxel = Texch_true_all[0]

    ATT_pred_lsq_voxel   = ATT_pred_lsq_all[0]
    CBF_pred_lsq_voxel   = CBF_pred_lsq_all[0]
    Texch_pred_lsq_voxel = Texch_pred_lsq_all[0]

    ATT_pred_net_voxel   = ATT_pred_net_all[0]
    CBF_pred_net_voxel   = CBF_pred_net_all[0]
    Texch_pred_net_voxel = Texch_pred_net_all[0]


    # ---------------------------------------------------------------------
    # Plot: left = signals, right = academic-style text panel
    # ---------------------------------------------------------------------
    # Use a serif font for a more academic style
    plt.rcParams["font.family"] = "Times New Roman"


    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 4, 1], hspace=0)

    # Left: signal comparison
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_sig.plot(TI*1000, X_test_reshape, label="Noisy simulated signal", alpha=0.95, linestyle='dashed', color='dimgray')
    ax_sig.plot(TI*1000, X_test_clean_reshape, label="Clean simulated signal", color='black', linewidth=2)
    ax_sig.plot(TI*1000, LSQ_signal_reshape, label="OLS regenerated signal", color='green', linewidth=2, linestyle='dashed')
    ax_sig.plot(TI*1000, PI_NN_signal_reshape, label="PI-NN regenerated signal", color='red', linewidth=2, linestyle='dashed')

    ax_sig.set_xlabel("TI (ms)")
    ax_sig.set_ylabel("Signal intensity (a.u.)")
    ax_sig.set_title("ASL signal comparison", fontweight="bold")
    ax_sig.legend()
    ax_sig.grid(True)

    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_sig)
    ax_res.plot(TI*1000, Residuals_LSQ, label="OLS residuals", color='green', linestyle='dashed')
    ax_res.plot(TI*1000, Residuals_PINN, label="PI-NN residuals", color='red', linestyle='dashed')
    ax_res.set_xlabel("TI (ms)")
    ax_res.set_ylabel("Residuals (a.u.)")
    ax_res.legend()
    ax_res.grid(True)
    #Line at zero
    ax_res.axhline(0, color='black', linestyle='dotted', linewidth=1.0)

    # Right: parameter summary and MPE in a clean text block
    ax_text = fig.add_subplot(gs[2, 0])
    ax_text.axis("off")

    stats_text = (
        "\n\n\n\n\n\nTrue parameters (voxel):   "
        f"  ATT   = {ATT_true_voxel:.2f} s   "
        f"  CBF   = {CBF_true_voxel:.2f} ml/100g/min   "
        f"  Texch = {Texch_true_voxel:.2f} s\n\n"
        "LSQ predictions (voxel):   "
        f"  ATT   = {ATT_pred_lsq_voxel:.2f} s  "
        f"  CBF   = {CBF_pred_lsq_voxel:.2f} ml/100g/min    "
        f"  Texch = {Texch_pred_lsq_voxel:.2f} s\n\n"
        "PI-NN predictions (voxel):   "
        f"  ATT   = {ATT_pred_net_voxel:.2f} s   "
        f"  CBF   = {CBF_pred_net_voxel:.2f} ml/100g/min    "
        f"  Texch = {Texch_pred_net_voxel:.2f} s\n\n"
        "Mean Percentage Errors (all sims):\n"
        "  Parameter   |    LSQ     |   PI-NN   \n"
        "---------------------------------------\n"
        f"     ATT     |  {mpe_lsq[0]:.3g}%  |  {mpe_net[0]:.3g}% \n"
        f"     CBF     |  {mpe_lsq[1]:.3g}%  |  {mpe_net[1]:.3g}% \n"
        f"    Texch    |  {mpe_lsq[2]:.3g}%  |  {mpe_net[2]:.3g}% \n\n"
        f"Absolute Mean Error of regenerated signals:\n"
        f"  LSQ:   {avg_abs_err_LSQ:.3g} a.u.\n"
        f"  PI-NN: {avg_abs_err_PINN:.3g} a.u.\n"
    )

    ax_text.text(
        0.0, 1.0, stats_text,
        transform=ax_text.transAxes,
        va="top", ha="left",
        fontsize=10
    )

    fig.tight_layout()
    plt.show()

    return None

def bland_altman_plot(ref, method):
    """
    Make a Bland-Altman plot for two 1D arrays of measurements
    (ref and method must be same length).

    Black & white style: open circles + black lines.
    If ax is None, a new figure is created and `label` is used as the figure title.
    """
    ref    = np.asarray(ref).ravel()
    method = np.asarray(method).ravel()

    mean_vals = (ref + method) / 2.0
    diff_vals = method - ref          # method minus reference

    mean_diff = np.mean(diff_vals)
    sd_diff   = np.std(diff_vals, ddof=1)

    # limits of agreement
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff


    return mean_diff, sd_diff, loa_upper, loa_lower

if __name__ == "__main__":
    #make_training_data_demo()
    #net = net_train_demo()
    #lsq_regression_demo()
    #generate_brain_slices_demo()
    test_predict_and_plot_brain_slice()
    #test_models_on_simulated_data()
    #model_signal_regeneration_test()
    #training_data_fig()