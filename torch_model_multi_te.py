import os
import torch

# tqdm is no longer required (no Python loops), but keeping the import
# is harmless if other files expect it.
from tqdm.auto import tqdm  # noqa: F401


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Compute exp(x) with clamping to avoid overflow."""
    return torch.exp(torch.clamp(x, max=80.0))


# -----------------------------
# Automatic thread configuration (Runpod-friendly)
# -----------------------------
_TORCH_THREADS_CONFIGURED = False


def _cgroup_cpu_quota_cpus() -> int | None:
    """
    Returns an integer CPU quota derived from cgroup v2 cpu.max if present.
    Example cpu.max: "2040000 100000" => 20.4 CPUs => returns 20
    """
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as f:
            raw = f.read().strip().split()
        if len(raw) != 2:
            return None
        quota_s, period_s = raw
        if quota_s == "max":
            return None
        quota = float(quota_s)
        period = float(period_s)
        if period <= 0:
            return None
        cpus = quota / period
        # floor to a sensible integer >= 1
        return max(1, int(cpus))
    except Exception:
        return None


def _configure_torch_threads_once() -> None:
    """
    Configure PyTorch thread counts to avoid massive oversubscription.
    Safe to call repeatedly; it will only apply once per process.
    """
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return

    quota = _cgroup_cpu_quota_cpus()
    host = os.cpu_count() or 1
    num_threads = min(host, quota) if quota is not None else host

    # Practical defaults: intra-op = quota, inter-op small
    try:
        torch.set_num_threads(int(num_threads))
        torch.set_num_interop_threads(int(min(4, num_threads)))
    except Exception:
        # If torch disallows setting threads in this context, ignore.
        pass

    _TORCH_THREADS_CONFIGURED = True


# -----------------------------
# Helpers for TI/TE expansion
# -----------------------------
def _expand_ti_tau(
    tis,
    ntes,
    taus,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expand per-TI arrays into per-TE vectors of length L=sum(ntes).
    Returns:
      ti_per_te:  (L,)
      tau_per_te: (L,)
    """
    tis_t = torch.as_tensor(tis, dtype=torch.float32, device=device)
    ntes_t = torch.as_tensor(ntes, dtype=torch.long, device=device)

    if ntes_t.ndim != 1:
        ntes_t = ntes_t.view(-1)
    if tis_t.ndim != 1:
        tis_t = tis_t.view(-1)

    if tis_t.numel() != ntes_t.numel():
        raise ValueError(f"tis and ntes must have same length; got {tis_t.numel()} and {ntes_t.numel()}")

    ti_per_te = tis_t.repeat_interleave(ntes_t)  # (L,)

    if isinstance(taus, (float, int)):
        tau_t = torch.full_like(tis_t, float(taus))
    else:
        tau_t = torch.as_tensor(taus, dtype=torch.float32, device=device)
        if tau_t.ndim != 1:
            tau_t = tau_t.view(-1)
        if tau_t.numel() != tis_t.numel():
            raise ValueError(f"taus must be scalar or same length as tis; got {tau_t.numel()} vs {tis_t.numel()}")

    tau_per_te = tau_t.repeat_interleave(ntes_t)  # (L,)
    return ti_per_te, tau_per_te


# -----------------------------
# Vectorized batch model (fast)
# -----------------------------
def torch_deltaM_multite_model_batch(
    tis, tes, ntes, att, cbf, texch, m0a, taus,
    t1=1.3, t1b=1.65, t2=0.050, t2b=0.150,
    itt=0.2, lambd=0.9, alpha=0.68,
    show_progress=False,  # kept for API compatibility; unused in vectorized implementation
):
    """
    Fully vectorized batch wrapper.
      att, cbf, texch: (B,) or (B,1)
    Returns:
      (B, sum(ntes))
    """
    _configure_torch_threads_once()

    # Ensure tensors and shapes
    if not torch.is_tensor(att):
        att = torch.as_tensor(att, dtype=torch.float32)
    if not torch.is_tensor(cbf):
        cbf = torch.as_tensor(cbf, dtype=torch.float32)
    if not torch.is_tensor(texch):
        texch = torch.as_tensor(texch, dtype=torch.float32)

    att = att.view(-1, 1)      # (B,1)
    cbf = cbf.view(-1, 1)      # (B,1)
    texch = texch.view(-1, 1)  # (B,1)
    device = att.device
    B = att.shape[0]

    # Expand TI and tau to per-TE vectors
    ti_per_te, tau_per_te = _expand_ti_tau(tis, ntes, taus, device=device)
    L = int(ti_per_te.numel())

    tes_t = torch.as_tensor(tes, dtype=torch.float32, device=device).view(-1)
    if tes_t.numel() != L:
        raise ValueError(f"tes must have length sum(ntes)={L}, got {tes_t.numel()}")

    te_base = tes_t.view(1, L)             # (1,L)
    ti_base = ti_per_te.view(1, L)         # (1,L)
    tau_base = tau_per_te.view(1, L)       # (1,L)

    # Broadcast to (B,L)
    te = te_base.expand(B, L)
    ti = ti_base.expand(B, L)
    tau = tau_base.expand(B, L)

    # cbf in ml/min/100g -> ml/s/g (keep exactly as your original code)
    f = cbf / 100.0 * 60.0 / 6000.0  # (B,1)

    # Constants as tensors
    itt_t = torch.tensor(float(itt), device=device, dtype=torch.float32)
    inv_t1 = torch.tensor(1.0 / float(t1), device=device, dtype=torch.float32)
    inv_t1b = torch.tensor(1.0 / float(t1b), device=device, dtype=torch.float32)
    inv_t2 = torch.tensor(1.0 / float(t2), device=device, dtype=torch.float32)
    inv_t2b = torch.tensor(1.0 / float(t2b), device=device, dtype=torch.float32)

    # Precompute TE-only exponentials
    exp_te_t2b = safe_exp(-te_base * inv_t2b).expand(B, L)  # (B,L)
    exp_te_t2 = safe_exp(-te_base * inv_t2).expand(B, L)    # (B,L)

    # TE/texch depends on sample
    exp_te_tex = safe_exp(-te / texch)  # (B,L)

    # Common time points
    att_itt = att + itt_t  # (B,1)
    att_tau = att + tau    # (B,L) broadcasted
    att_itt_tau = att_itt + tau  # (B,L)

    # Common exponentials (att, ti)
    exp_neg_att_t1b = safe_exp(-att * inv_t1b)            # (B,1)
    exp_pos_att_t1b = safe_exp(att * inv_t1b)             # (B,1)
    exp_neg_attitt_t1b = safe_exp(-att_itt * inv_t1b)     # (B,1)
    exp_pos_attitt_t1b = safe_exp(att_itt * inv_t1b)      # (B,1)

    exp_neg_ti_t1b = safe_exp(-ti * inv_t1b)              # (B,L)
    exp_pos_ti_t1b = safe_exp(ti * inv_t1b)               # (B,L)

    # Outputs
    S_bl1 = torch.zeros((B, L), dtype=torch.float32, device=device)
    S_bl2 = torch.zeros((B, L), dtype=torch.float32, device=device)
    S_ex  = torch.zeros((B, L), dtype=torch.float32, device=device)

    # -------------------------
    # Case definitions (mirror your original branching)
    # -------------------------
    mask1 = (ti > 0.0) & (ti < att)  # all-zero

    mask2 = (ti >= att) & (ti < att_itt)  # Case 2

    # Your original Case 3 condition in code:
    #   (att + itt) <= ti < (att + tau)
    mask3 = (ti >= att_itt) & (ti < att_tau)

    # Case 4:
    mask4 = (ti >= att_tau) & (ti < att_itt_tau)

    # Case 5 is the original "else"
    mask5 = ~(mask1 | mask2 | mask3 | mask4)

    # -------------------------
    # CASE 2
    # -------------------------
    # base_term = 2*f*t1b*exp(-att/t1b)*exp(-ti/t1b)*(exp(ti/t1b)-exp(att/t1b))
    base_term2 = (
        2.0 * f * float(t1b)
        * exp_neg_att_t1b
        * exp_neg_ti_t1b
        * (exp_pos_ti_t1b - exp_pos_att_t1b)
    )  # (B,L)

    boundary2 = att_itt - ti  # (B,L)

    # Subcase partition exactly like your if/elif/else:
    sub21 = mask2 & (te >= 0.0) & (te < boundary2)
    sub22 = mask2 & (te >= boundary2) & (te < itt_t)
    sub23 = mask2 & ~(sub21 | sub22)

    # Subcase 2.1
    S_bl1 = torch.where(sub21, base_term2 * exp_te_t2b, S_bl1)

    # Subcase 2.2 (transition)
    denom22 = (ti - att)
    denom22_safe = torch.where(denom22 == 0.0, torch.ones_like(denom22), denom22)
    tf22 = (te - boundary2) / denom22_safe
    tf22 = torch.where(sub22, tf22, torch.zeros_like(tf22))

    S_bl1 = torch.where(sub22, (base_term2 * (1.0 - tf22)) * exp_te_t2b, S_bl1)
    S_bl2 = torch.where(sub22, (tf22 * base_term2) * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub22, (tf22 * base_term2) * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    # Subcase 2.3
    S_bl2 = torch.where(sub23, base_term2 * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub23, base_term2 * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    # -------------------------
    # Shared terms for CASE 3/4/5
    # -------------------------
    # term1_base = 2*f*t1b*exp(-att/t1b)*exp(-ti/t1b)*(exp(ti/t1b)-exp(att/t1b))
    term1_base = base_term2  # (B,L)

    # term2_base = 2*f*t1b*exp(-(att+itt)/t1b)*exp(-ti/t1b)*(exp(ti/t1b)-exp((att+itt)/t1b))
    term2_base = (
        2.0 * f * float(t1b)
        * exp_neg_attitt_t1b
        * exp_neg_ti_t1b
        * (exp_pos_ti_t1b - exp_pos_attitt_t1b)
    )  # (B,L)

    base_diff3 = term1_base - term2_base  # (B,L)

    # k_b = (1/t1b) + (1/texch)
    k_b = inv_t1b + (1.0 / texch)  # (B,1)
    k_b_ti = k_b * ti              # (B,L)
    k_b_attitt = k_b * att_itt     # (B,1) broadcastable

    exp_pos_kb_ti = safe_exp(k_b_ti)                      # (B,L)
    exp_neg_kb_ti = safe_exp(-k_b_ti)                     # (B,L)
    exp_pos_kb_attitt = safe_exp(k_b_attitt).expand(B, L) # (B,L)

    A_pref = (
        2.0 * f
        * exp_neg_attitt_t1b
        / k_b
        * exp_neg_kb_ti
        * (exp_pos_kb_ti - exp_pos_kb_attitt)
    )  # (B,L)

    # ex_term1 for cases 3 & 4:
    # base1 = (2*f*exp(-(1/t1b)*(att+itt)))/(1/t1) * exp(-(1/t1)*ti) * (exp((1/t1)*ti) - exp((1/t1)*(att+itt)))
    inv_t1_ti = inv_t1 * ti
    exp_pos_it1_ti = safe_exp(inv_t1_ti)
    exp_neg_it1_ti = safe_exp(-inv_t1_ti)
    exp_pos_it1_attitt = safe_exp(inv_t1 * att_itt).expand(B, L)

    base1_34 = (
        2.0 * f
        * exp_neg_attitt_t1b
        / inv_t1
        * exp_neg_it1_ti
        * (exp_pos_it1_ti - exp_pos_it1_attitt)
    )  # (B,L)

    k_1 = inv_t1 + (1.0 / texch)  # (B,1)
    k1_ti = k_1 * ti              # (B,L)
    k1_attitt = k_1 * att_itt     # (B,1)

    exp_pos_k1_ti = safe_exp(k1_ti)
    exp_neg_k1_ti = safe_exp(-k1_ti)
    exp_pos_k1_attitt = safe_exp(k1_attitt).expand(B, L)

    base2_34 = (
        2.0 * f
        * exp_neg_attitt_t1b
        / k_1
        * exp_neg_k1_ti
        * (exp_pos_k1_ti - exp_pos_k1_attitt)
    )  # (B,L)

    ex_term1_34 = (base1_34 - base2_34) * exp_te_t2  # (B,L)

    # -------------------------
    # CASE 3
    # -------------------------
    # Subcase partition mirrors: if 0 <= te < itt else
    sub31 = mask3 & (te >= 0.0) & (te < itt_t)
    sub32 = mask3 & ~sub31

    tf31 = te / itt_t  # safe; itt constant > 0

    S_bl1 = torch.where(sub31, (base_diff3 * (1.0 - tf31)) * exp_te_t2b, S_bl1)
    S_bl2 = torch.where(sub31, (A_pref + tf31 * base_diff3) * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub31, ex_term1_34 + (A_pref + tf31 * base_diff3) * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    S_bl2 = torch.where(sub32, (A_pref + base_diff3) * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub32, ex_term1_34 + (A_pref + base_diff3) * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    # -------------------------
    # CASE 4
    # -------------------------
    # term1_4 = 2*f*t1b*exp(-att/t1b)*exp(-ti/t1b)*(exp((att+tau)/t1b)-exp(att/t1b))
    exp_pos_att_tau_t1b = safe_exp(att_tau * inv_t1b)  # (B,L)
    term1_4 = (
        2.0 * f * float(t1b)
        * exp_neg_att_t1b
        * exp_neg_ti_t1b
        * (exp_pos_att_tau_t1b - exp_pos_att_t1b)
    )  # (B,L)

    base_diff4 = term1_4 - term2_base  # (B,L)

    bound4 = itt_t - (ti - att_tau)  # (B,L) = itt - (ti-(att+tau))

    sub41 = mask4 & (te >= 0.0) & (te < bound4)
    sub42 = mask4 & ~sub41

    denom41_safe = torch.where(bound4 == 0.0, torch.ones_like(bound4), bound4)
    tf41 = te / denom41_safe
    tf41 = torch.where(sub41, tf41, torch.zeros_like(tf41))

    S_bl1 = torch.where(sub41, (base_diff4 * (1.0 - tf41)) * exp_te_t2b, S_bl1)
    S_bl2 = torch.where(sub41, (A_pref + tf41 * base_diff4) * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub41, ex_term1_34 + (A_pref + tf41 * base_diff4) * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    S_bl2 = torch.where(sub42, (A_pref + base_diff4) * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(sub42, ex_term1_34 + (A_pref + base_diff4) * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    # -------------------------
    # CASE 5 (original "else")
    # -------------------------
    # B_pref = 2*f*exp(-(1/t1b)*(att+itt))/((1/t1b)+(1/texch)) * exp(-k_b*ti) *
    #          (exp(k_b*(att+itt+tau)) - exp(k_b*(att+itt)))
    exp_pos_kb_attitt_tau = safe_exp(k_b * att_itt_tau)   # (B,L)
    B_pref = (
        2.0 * f
        * exp_neg_attitt_t1b
        / k_b
        * exp_neg_kb_ti
        * (exp_pos_kb_attitt_tau - exp_pos_kb_attitt)
    )  # (B,L)

    # ex_term1 for case 5:
    exp_pos_it1_attitt_tau = safe_exp(inv_t1 * att_itt_tau)  # (B,L)
    base1_5 = (
        2.0 * f
        * exp_neg_attitt_t1b
        / inv_t1
        * exp_neg_it1_ti
        * (exp_pos_it1_attitt_tau - exp_pos_it1_attitt)
    )  # (B,L)

    exp_pos_k1_attitt_tau = safe_exp(k_1 * att_itt_tau)  # (B,L)
    base2_5 = (
        2.0 * f
        * exp_neg_attitt_t1b
        / k_1
        * exp_neg_k1_ti
        * (exp_pos_k1_attitt_tau - exp_pos_k1_attitt)
    )  # (B,L)

    ex_term1_5 = (base1_5 - base2_5) * exp_te_t2  # (B,L)

    # Apply case 5 assignments only where mask5
    S_bl1 = torch.where(mask5, torch.zeros_like(S_bl1), S_bl1)
    S_bl2 = torch.where(mask5, B_pref * exp_te_t2b * exp_te_tex, S_bl2)
    S_ex  = torch.where(mask5, ex_term1_5 + B_pref * (1.0 - exp_te_tex) * exp_te_t2, S_ex)

    # Final delta M
    delta_M = (S_bl1 + S_bl2 + S_ex) * m0a * alpha / lambd  # (B,L)
    return delta_M


def torch_deltaM_multite_model_single(
    tis, tes, ntes, att, cbf, texch, m0a, taus,
    t1=1.3, t1b=1.65, t2=0.050, t2b=0.150,
    itt=0.2, lambd=0.9, alpha=0.68, device=None
):
    """
    Compatibility wrapper: returns a 1D tensor of length sum(ntes) for one sample.
    This calls the fully vectorized batch implementation with B=1.
    """
    if device is None:
        device = att.device if torch.is_tensor(att) else torch.device("cpu")

    att_t = torch.as_tensor(att, dtype=torch.float32, device=device).view(1)
    cbf_t = torch.as_tensor(cbf, dtype=torch.float32, device=device).view(1)
    texch_t = torch.as_tensor(texch, dtype=torch.float32, device=device).view(1)

    out = torch_deltaM_multite_model_batch(
        tis=tis, tes=tes, ntes=ntes,
        att=att_t, cbf=cbf_t, texch=texch_t,
        m0a=m0a, taus=taus,
        t1=t1, t1b=t1b, t2=t2, t2b=t2b,
        itt=itt, lambd=lambd, alpha=alpha,
        show_progress=False
    )
    return out.view(-1)
