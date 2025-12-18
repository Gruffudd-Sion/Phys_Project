# import libraries
from matplotlib import scale
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import copy
import warnings
from model_multi_te import deltaM_multite_model
from torch_model_multi_te import torch_deltaM_multite_model_batch
from tqdm.auto import tqdm

class Net(nn.Module):
    def __init__(self, TE, TI, net_pars, other_params):
        super(Net, self).__init__()

        self.TE = np.array(TE)   # shape (nTE,)
        self.TI = np.array(TI)   # shape (nTI,)

        TE_flat = np.tile(self.TE, len(self.TI))          # (nTE * nTI,)
        TI_flat = np.repeat(self.TI, len(self.TE))        # (nTE * nTI,)
        self.register_buffer("TE_flat", torch.tensor(TE_flat, dtype=torch.float32))
        self.register_buffer("TI_flat", torch.tensor(TI_flat, dtype=torch.float32))
        self.net_pars = net_pars
        self.other_params = other_params

        if self.other_params.width == 0:
            self.other_params.width = 128


        # fc0  -> ATT branch
        # fc1  -> CBF branch
        # fc2  -> Texch branch
        self.fc0 = nn.ModuleList()
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()

        # input width = #TE * #TI
        width = len(self.TI) * len(self.TE)

        for i in range(self.net_pars.depth):
            # ----- ATT branch -----
            self.fc0.append(nn.Linear(width, self.net_pars.width))
            if self.net_pars.batch_norm:
                self.fc0.append(nn.BatchNorm1d(self.net_pars.width))
            self.fc0.append(nn.ELU())
            if self.net_pars.dropout != 0 and i != (self.net_pars.depth - 1):
                self.fc0.append(nn.Dropout(p=self.net_pars.dropout))

            # ----- CBF branch -----
            self.fc1.append(nn.Linear(width, self.net_pars.width))
            if self.net_pars.batch_norm:
                self.fc1.append(nn.BatchNorm1d(self.net_pars.width))
            self.fc1.append(nn.ELU())
            if self.net_pars.dropout != 0 and i != (self.net_pars.depth - 1):
                self.fc1.append(nn.Dropout(p=self.net_pars.dropout))

            # ----- Texch branch -----
            self.fc2.append(nn.Linear(width, self.net_pars.width))
            if self.net_pars.batch_norm:
                self.fc2.append(nn.BatchNorm1d(self.net_pars.width))
            self.fc2.append(nn.ELU())
            if self.net_pars.dropout != 0 and i != (self.net_pars.depth - 1):
                self.fc2.append(nn.Dropout(p=self.net_pars.dropout))

            # after first layer, both branches have width = net_pars.width
            width = self.net_pars.width

        # final heads:
        # encoder0: ATT branch → 1 output
        # encoder1: CBF branch → 1 output
        # encoder2: Texch branch → 1 output
        self.encoder0 = nn.Sequential(*self.fc0, nn.Linear(self.net_pars.width, 1))
        self.encoder1 = nn.Sequential(*self.fc1, nn.Linear(self.net_pars.width, 1))
        self.encoder2 = nn.Sequential(*self.fc2, nn.Linear(self.net_pars.width, 1))

    def forward(self, x_signal, G_or_W_bool, synthesize=False):
        """
        x_signal: (batch, nTE * nTI)  # just the measured/simulated signal
        """
        batch_size, n_feats = x_signal.shape
        device = x_signal.device

        x = x_signal   

        # ---- your existing param prediction code ----
        ATT_min = self.net_pars.cons_min[0]
        ATT_max = self.net_pars.cons_max[0]
        CBF_min = self.net_pars.cons_min[1]
        CBF_max = self.net_pars.cons_max[1]
        texch_min = self.net_pars.cons_min[2]
        texch_max = self.net_pars.cons_max[2]

        raw_ATT = self.encoder0(x)   # (batch, 1) now sees signal+TE+TI
        raw_CBF = self.encoder1(x)   # (batch, 1)
        raw_texch = self.encoder2(x)   # (batch, 1)

        ATT = ATT_min + (ATT_max - ATT_min) * torch.sigmoid(raw_ATT)
        CBF = CBF_min + (CBF_max - CBF_min) * torch.sigmoid(raw_CBF)
        texch = texch_min + (texch_max - texch_min) * torch.sigmoid(raw_texch)

        pred_signal = None
        if synthesize:
            ntes = self.other_params.ntes
            if ntes is None:
                ntes = [len(self.TE)] * len(self.TI)

            # if TE list already flattened, ok; else tile
            if len(self.TE) == sum(int(n) for n in (ntes if isinstance(ntes, (list,tuple)) else ntes.tolist())):
                te_flat = self.TE
            else:
                te_flat = []
                for _ in range(len(self.TI)):
                    te_flat.extend(self.TE)

            tis = torch.as_tensor(self.TI, dtype=torch.float32, device=device)
            tes = torch.as_tensor(te_flat, dtype=torch.float32, device=device)

            matter_mask = G_or_W_bool
            if isinstance(matter_mask, torch.Tensor):
                matter_mask = matter_mask.to(device).view(-1).bool()
            else:
                matter_mask = torch.full((batch_size,), bool(matter_mask), device=device, dtype=torch.bool)

            pred_signal = torch.empty((batch_size, n_feats), device=device, dtype=torch.float32)

            gm_idx = matter_mask
            wm_idx = ~matter_mask

            if gm_idx.any():
                pred_signal[gm_idx] = torch_deltaM_multite_model_batch(
                    tis=tis, tes=tes, ntes=ntes,
                    att=ATT[gm_idx], cbf=CBF[gm_idx], texch=texch[gm_idx],
                    m0a=torch.tensor(1.0, device=device),
                    taus=1.5,
                    t1=self.other_params.GM_T1,
                    t1b=self.other_params.T1a,
                    t2=self.other_params.GM_T2,
                    t2b=self.other_params.T2a,
                    itt=self.other_params.itt,
                    lambd=self.other_params.lambd,
                    alpha=self.other_params.alpha,
                )

            if wm_idx.any():
                pred_signal[wm_idx] = torch_deltaM_multite_model_batch(
                    tis=tis, tes=tes, ntes=ntes,
                    att=ATT[wm_idx], cbf=CBF[wm_idx], texch=texch[wm_idx],
                    m0a=torch.tensor(1.0, device=device),
                    taus=1.5,
                    t1=self.other_params.WM_T1,
                    t1b=self.other_params.T1a,
                    t2=self.other_params.WM_T2,
                    t2b=self.other_params.T2a,
                    itt=self.other_params.itt,
                    lambd=self.other_params.lambd,
                    alpha=self.other_params.alpha,
                )
        return pred_signal, ATT, CBF, texch

def train_from_simulator(sim_kwargs, arg, simulations_module):
    """
    Train PI-NN on simulated data, with physics-informed loss for training
    and param-only loss for validation.
    """
    # 1) load training data from file
    X = np.load("simulated_training_data.npy")
    y = np.load("simulated_training_labels.npy")
    X_clean = np.load("simulated_training_data_clean.npy")
    try:
        G_or_W_mask = np.load("simulated_training_GM_WM_mask.npy")
    except:
        # if no mask file, assume all grey matter
        G_or_W_mask = np.ones((X.shape[0],), dtype=bool)
    print(f"[PI-NN] Data shape: X={X.shape}, y={y.shape}, X_clean={X_clean.shape}")

    TE = np.load("training_TE_values.npy")
    TI = np.load("training_TI_values.npy")


    physics_informed = arg.train_pars.physics_informed

    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_clean_tensor = torch.tensor(X_clean, dtype=torch.float32)

    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_mean = y_tensor.mean(dim=0, keepdim=True)
    y_std = y_tensor.std(dim=0, keepdim=True) + 1e-8

    G_or_W_mask_tensor = torch.tensor(G_or_W_mask, dtype=torch.bool)

    dataset = utils.TensorDataset(X_tensor, y_tensor, X_clean_tensor, G_or_W_mask_tensor)
    # split
    split_ratio = arg.train_pars.validation_split
    split = int(np.floor(split_ratio * len(dataset)))
    train_set, val_set = utils.random_split(dataset, [len(dataset) - split, split])

    train_loader = utils.DataLoader(train_set, batch_size=arg.train_pars.batch_size, shuffle=True)
    val_loader = utils.DataLoader(val_set, batch_size=arg.train_pars.batch_size, shuffle=False)

    # 2) build net
    net = Net(TE, TI, arg.net_pars, arg.other_params)

    # 3) loss
    if arg.train_pars.loss_function == 'MSE':
        criterion = nn.MSELoss()
    elif arg.train_pars.loss_function == 'MAE':
        criterion = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function")

    # 4) optimizer
    optimizer = load_optimizer(net, arg)

    phys_frac = getattr(arg.train_pars, 'phys_frac', 1.0)
    patience = getattr(arg.train_pars, 'patience', None)
    epochs = getattr(arg.train_pars, 'epochs', 10)

    best_val_loss = float('inf')
    best_state_dict = copy.deepcopy(net.state_dict())
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        net.train()
        running_total = 0.0
        running_param = 0.0
        xb, yb, xb_clean, matter_bool = next(iter(train_loader))

        for xb, yb, xb_clean, matter_bool in train_loader:
            optimizer.zero_grad()

            # physics-informed forward
            if physics_informed:
                pred_signal, ATT_pred, CBF_pred, Texch_pred = net(xb, matter_bool, synthesize=True)

            else:
                pred_signal, ATT_pred, CBF_pred, Texch_pred = net(xb, matter_bool, synthesize=False)

            # param loss
            params_pred = torch.cat([ATT_pred, CBF_pred, Texch_pred], dim=1)  # (batch, 3)
            params_true = yb                                                 # (batch, 3)

            mean = y_mean.to(params_true.device)
            std = y_std.to(params_true.device)

            params_norm_pred = (params_pred- mean) / std
            params_norm_true = (params_true - mean) / std

            param_loss = criterion(params_norm_pred, params_norm_true)

            #print("Weighted param loss:", (1-phys_frac) * param_loss.item())
            diff = params_norm_pred - params_norm_true
            #per_param_mse = torch.mean(diff ** 2, dim=0)
            #print(f"Per-parameter MSE: ATT={(1-phys_frac)*per_param_mse[0].item():.3f}, CBF={(1-phys_frac)*per_param_mse[1].item():.3f}, Texch={(1-phys_frac)*per_param_mse[2].item():.3f}")

            # physics loss
            if physics_informed:
                #scale each point by its max signal value to normalize
                normalized_pred_signal = pred_signal / (torch.max(torch.abs(xb_clean), dim=1, keepdim=True)[0] + 1e-8)
                normalized_xb_clean = xb_clean / (torch.max(torch.abs(xb_clean), dim=1, keepdim=True)[0] + 1e-8)
                phys_loss = criterion(normalized_pred_signal, normalized_xb_clean) #/ scale.to(pred_signal.device)
                
                #print("Weighted physics loss:", (phys_frac * phys_loss.item()))
                loss = (1-phys_frac) * param_loss + phys_frac * phys_loss
            else:
                phys_loss = 0.0
                loss = param_loss
            
            loss.backward()
            optimizer.step()

            running_total += loss.item() * xb.size(0)
            running_param += param_loss.item() * xb.size(0)

        train_loss = running_total / len(train_loader.dataset)
        train_param_loss = running_param / len(train_loader.dataset)


        ATT_errors = []
        CBF_errors = []
        Texch_errors = []
        ATT_percent_errors = []
        CBF_percent_errors = []
        Texch_percent_errors = []
        # validation: param-only so it's interpretable
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, xb_clean, matter_bool in val_loader:
                _, ATT_pred, CBF_pred, Texch_pred = net(xb, matter_bool, synthesize=False)
                ATT_true = yb[:, 0:1]
                CBF_true = yb[:, 1:2]
                Texch_true = yb[:, 2:3]

                ATT_errors.append((ATT_pred - ATT_true).cpu().numpy())
                CBF_errors.append((CBF_pred - CBF_true).cpu().numpy())
                Texch_errors.append((Texch_pred - Texch_true).cpu().numpy())
                ATT_percent_errors.append(((ATT_pred - ATT_true) / ATT_true).cpu().numpy())
                CBF_percent_errors.append(((CBF_pred - CBF_true) / CBF_true).cpu().numpy())
                Texch_percent_errors.append(((Texch_pred - Texch_true) / Texch_true).cpu().numpy())


                params_pred = torch.cat([ATT_pred, CBF_pred, Texch_pred], dim=1)
                params_true = yb

                mean = y_mean.to(params_true.device)
                std = y_std.to(params_true.device)

                params_norm_pred = (params_pred - mean) / std
                params_norm_true = (params_true - mean) / std
                
                #weights = torch.tensor([1.0, 1.0, 30.0], device=params_true.device)
                #param_weighted_difference = (params_norm_pred - params_norm_true) * weights
                #loss = criterion(param_weighted_difference, torch.zeros_like(param_weighted_difference))
                loss = criterion(params_norm_pred, params_norm_true)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        ATT_errors = np.concatenate(ATT_errors, axis=0)
        CBF_errors = np.concatenate(CBF_errors, axis=0)
        Texch_errors = np.concatenate(Texch_errors, axis=0)
        ATT_percent_errors = np.concatenate(ATT_percent_errors, axis=0)
        CBF_percent_errors = np.concatenate(CBF_percent_errors, axis=0)
        Texch_percent_errors = np.concatenate(Texch_percent_errors, axis=0)

        rmse_ATT = np.sqrt(np.mean(ATT_errors ** 2))
        rmse_CBF = np.sqrt(np.mean(CBF_errors ** 2))
        rmse_Texch = np.sqrt(np.mean(Texch_errors ** 2))

        mean_percent_error_ATT = np.mean(np.abs(ATT_percent_errors)) * 100
        mean_percent_error_CBF = np.mean(np.abs(CBF_percent_errors)) * 100
        mean_percent_error_Texch = np.mean(np.abs(Texch_percent_errors)) * 100

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        last_ATT_true, last_ATT_pred = ATT_true, ATT_pred
        last_CBF_true, last_CBF_pred = CBF_true, CBF_pred
        last_Texch_true, last_Texch_pred = Texch_true, Texch_pred

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
    
    net.load_state_dict(best_state_dict)

    print(
        f"Final (best) val loss: {best_val_loss:.6f}\n"
        f"RMSE ATT: {rmse_ATT:.4f}, CBF: {rmse_CBF:.4f}\n, Texch: {rmse_Texch:.4f}\n"
        f"Mean % error ATT: {mean_percent_error_ATT:.2f}%, CBF: {mean_percent_error_CBF:.2f}%, Texch: {mean_percent_error_Texch:.2f}%\n"
        f"Last batch ATT_true: {last_ATT_true}, ATT_pred: {last_ATT_pred}\n"
        f"Last batch CBF_true: {last_CBF_true}, CBF_pred: {last_CBF_pred}\n"
        f"Last batch Texch_true: {last_Texch_true}, Texch_pred: {last_Texch_pred}"
    )

    return net, train_loss_history, val_loss_history

   

def load_optimizer(net, arg):
    """
    Load the optimizer for training.
    Parameters:
    net: nn.Module
        Neural network model
    arg: object
        Additional arguments for training (expects arg.train_pars.*)
    Returns:
    torch.optim.Optimizer
        Configured optimizer
    """
    # simple: always optimize all network parameters (both branches)
    params = net.parameters()

    if arg.train_pars.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=arg.train_pars.learning_rate)
    elif arg.train_pars.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=arg.train_pars.learning_rate, momentum=0.9)
    elif arg.train_pars.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(params, lr=arg.train_pars.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {arg.train_pars.optimizer}")

    return optimizer