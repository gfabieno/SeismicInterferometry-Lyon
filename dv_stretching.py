import torch
import torch.nn.functional as F

def stretching_dvv_torch_vec(
    cur: torch.Tensor,                 # [..., T]
    ref: torch.Tensor,                 # [..., T] broadcastable to cur
    epsilons: torch.Tensor,            # [E]
    tmin: int | torch.Tensor = 0,      # int or [...]
    tmax: int | torch.Tensor = None,   # int or [...]
    demean: bool = True,
    return_cc_curve: bool = False,
    eps: float = 1e-12,
):
    """
    Vectorized stretching dv/v estimation in PyTorch.

    cur:  [..., T]  current waveforms (e.g., N,S,T)
    ref:  [..., T]  reference waveforms, broadcastable (e.g., N,1,T or 1,1,T)
    tmin: int or [...]   window start sample (inclusive)
    tmax: int or [...]   window end sample (exclusive); if None, use T
    demean: bool          whether to demean traces in the window
    epsilons: [E]   trial stretch values; dv/v = -epsilon
    Returns:
      dvv:      [...]      best dv/v for each trace
      cc_best:  [...]      max correlation
      (optional) cc_curve: [..., E]
    """
    if cur.ndim < 1 or ref.ndim < 1:
        raise ValueError("cur and ref must be tensors with at least 1 dimension (time).")
    if cur.shape[-1] != ref.shape[-1]:
        raise ValueError(f"Time dimension mismatch: cur T={cur.shape[-1]} vs ref T={ref.shape[-1]}")

    device = cur.device
    cur = cur.float()
    ref = ref.float()
    epsilons = epsilons.to(device=device, dtype=torch.float32)
    T = cur.shape[-1]

    # Window in time
    if tmax is None:
        tmax = T
    if tmin is None:
        tmin = 0
    time_idx = torch.arange(T, device=device)
    if not isinstance(tmin, torch.Tensor):
        tmin = torch.full(cur.shape[:-1], tmin, device=device, dtype=torch.long)
    if not isinstance(tmax, torch.Tensor):
        tmax = torch.full(cur.shape[:-1], tmax, device=device, dtype=torch.long)
    mask = (time_idx >= tmin[..., None]) & (time_idx < tmax[..., None])
    mask = mask.float()
    curw = cur * mask
    refw = ref * mask
    Tw = curw.shape[-1]

    # Broadcast ref to cur shape (except time already matches); this is cheap (view-based)
    refw = torch.broadcast_to(refw, curw.shape)

    if demean:
        curw = curw - curw.mean(dim=-1, keepdim=True)
        refw = refw - refw.mean(dim=-1, keepdim=True)

    # Flatten batch dims -> B
    batch_shape = curw.shape[:-1]
    B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1

    cur2 = curw.reshape(B, Tw)   # [B, Tw]
    ref2 = refw.reshape(B, Tw)   # [B, Tw]

    # Build normalized coordinate grid [-1,1] for time samples
    base = torch.linspace(0, 1.0, Tw, device=device, dtype=torch.float32)  # [Tw]
    # For each epsilon, scale the grid (small-epsilon approximation widely used in stretching)
    # grid: [E, Tw]
    #base = base[None, :]# - tmin[:, None]  # shift if tmin != 0
    grid_x = base[None, :] * (1.0 + epsilons[:, None])
    grid_x = grid_x * 2.0 - 1.0  # to [-1,1] for grid_sample

    # grid_sample expects grid [N, H, W, 2]; we use H=1, W=Tw
    # grid_y is 0 (single row)
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)  # [E, Tw, 2]
    grid = grid[:, None, :, :]  # [E, 1, Tw, 2]

    # Prepare "image" tensor for grid_sample: [N, C, H, W]
    # We want all (epsilon, batch) combinations in one call:
    # Expand ref to [E, B, 1, 1, Tw] then merge (E*B) into N.
    ref_img = ref2[None, :, None, None, :]               # [1, B, 1, 1, Tw]
    ref_img = ref_img.expand(epsilons.numel(), B, 1, 1, Tw)
    ref_img = ref_img.reshape(epsilons.numel() * B, 1, 1, Tw)      # [E*B, 1, 1, Tw]
    #ref_img = ref_img.contiguous().to(dtype=torch.float32, device=device)

    grid_rep = grid[:, None, :, :, :].expand(epsilons.numel(), B, 1, Tw, 2)
    grid_rep = grid_rep.reshape(epsilons.numel() * B, 1, Tw, 2)    # [E*B, 1, Tw, 2]
    #grid_rep = grid_rep.contiguous().to(dtype=torch.float32, device=device)

    # Resample stretched refs
    stretched = F.grid_sample(
        ref_img,
        grid_rep,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [E*B, 1, 1, Tw]
    stretched = stretched[:, 0, 0, :].reshape(epsilons.numel(), B, Tw)  # [E, B, Tw]

    if demean:
        stretched = stretched - stretched.mean(dim=-1, keepdim=True)

    # Normalized correlation per epsilon and batch:
    # num: [E, B]
    num = torch.sum(stretched * cur2[None, :, :], dim=-1)
    # Normalize current once
    cur_norm = torch.sqrt(torch.sum(cur2 **2, dim=-1) + eps)  # [B]
    stretched_norm = torch.sqrt(torch.sum(stretched **2, dim=-1) + eps)  # [E, B]
    den = stretched_norm * cur_norm[None, :]
    cc = num / (den + eps)  # [E, B]

    # Best epsilon per batch
    best_idx = torch.argmax(cc, dim=0)  # [B]
    eps_best = epsilons[best_idx]       # [B]
    dvv = (-eps_best).reshape(batch_shape)          # [...]
    cc_best = torch.gather(cc, 0, best_idx[None, :]).squeeze(0).reshape(batch_shape)  # [...]

    if return_cc_curve:
        # return as [..., E] rather than [E,B]
        cc_curve = cc.permute(1, 0).reshape(*batch_shape, epsilons.numel())  # [..., E]
        return dvv, cc_best, cc_curve

    return dvv, cc_best
