import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from dv_stretching import stretching_dvv_torch_vec


# ---------------- Synthetic test with plots ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

N, S, T = 4, 6, 3000
dt = 0.01

true_dvv = -3.5e-4
true_eps = -true_dvv

noise_level = 0.001  # << smaller so CC stays high

t = torch.arange(T, device=device) * dt
ref0 = (
    torch.sin(2 * math.pi * 1.2 * t)
    + 0.6 * torch.sin(2 * math.pi * 2.8 * t)
    + 0.3 * torch.sin(2 * math.pi * 4.5 * t)
)
env = torch.exp(-t / 6.0)  # coda-like decay
ref0 = ref0 * env
ref0 = ref0 - ref0.mean()
ref0 = ref0 / (ref0.norm() + 1e-12)

# Reference shape (N, 1, T): one ref per station, shared across segments
ref = ref0[None, None, :].repeat(N, 1, 1)

# Create stretched current using same grid_sample mechanism (so the model matches perfectly)
base = torch.linspace(0, 1.0, T, device=device)
grid_x = base * (1.0 + true_eps)
grid_x = grid_x * 2.0 - 1.0  # to [-1,1] for grid_sample
grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)[None, None, :, :]  # (1,1,T,2)
ref_img = ref0[None, None, None, :]  # (1,1,1,T)

cur0 = F.grid_sample(
    ref_img, grid,
    mode="bilinear", padding_mode="zeros", align_corners=True
)[0, 0, 0, :]

# Expand to (N,S,T) and add noise
cur = cur0[None, None, :].repeat(N, S, 1)
cur = cur + noise_level * torch.randn_like(cur)

# dv/v estimation
epsilons = torch.linspace(-1e-3, 1e-3, 401, device=device)

tmin, tmax = 300, 2500
dvv_est, cc_best, cc_curve = stretching_dvv_torch_vec(
    cur, ref, epsilons, tmin=tmin, tmax=tmax, return_cc_curve=True,
    demean=False
)

print(f"Device         : {device}")
print(f"True dv/v      : {true_dvv:.2e}")
print(f"Estimated mean : {dvv_est.mean().item():.2e}")
print(f"Bias           : {(dvv_est.mean().item() - true_dvv):.2e}")
print(f"Mean CC        : {cc_best.mean().item():.3f}")

# ---- Plots ----
i_station, i_seg = 0, 0
tw = (torch.arange(tmax - tmin, device=device) * dt).cpu().numpy()

ref_ex = ref[i_station, 0, tmin:tmax].detach().cpu().numpy()
cur_ex = cur[i_station, i_seg, tmin:tmax].detach().cpu().numpy()

# 1) Waveform overlay
plt.figure()
plt.plot(tw, ref_ex, label="reference (windowed)")
plt.plot(tw, cur_ex, label="current (windowed)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Synthetic test: reference vs current")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Correlation curve vs dv/v for the example trace
cc_ex = cc_curve[i_station, i_seg, :].detach().cpu().numpy()
dvv_axis = (-epsilons).detach().cpu().numpy()

plt.figure()
plt.plot(dvv_axis, cc_ex)
plt.axvline(true_dvv, linestyle="--", label="true dv/v")
plt.axvline(dvv_est[i_station, i_seg].item(), color="red",
            linestyle="--", label="estimated dv/v")
plt.xlabel("dv/v")
plt.ylabel("Correlation coefficient")
plt.title("Stretching search curve (example trace)")
plt.legend()
plt.tight_layout()
plt.show()

# 3) dv/v map + histogram
dvv_cpu = dvv_est.detach().cpu().numpy()

plt.figure()
plt.imshow(dvv_cpu, aspect="auto")
plt.colorbar(label="dv/v")
plt.xlabel("Segment index (S)")
plt.ylabel("Station index (N)")
plt.title("Estimated dv/v map")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(dvv_cpu.ravel(), bins=20)
plt.axvline(true_dvv, linestyle="--", label="true dv/v")
plt.xlabel("dv/v")
plt.ylabel("Count")
plt.title("Distribution of estimated dv/v")
plt.legend()
plt.tight_layout()
plt.show()