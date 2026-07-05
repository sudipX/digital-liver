import torch


class CollapseMonitor:
    """Measures whether the latent representation has collapsed.

    Collapse happens when the encoder maps all inputs to nearly the
    same vector. The loss can still be low (predictor matches target)
    but the latent carries no useful information.

    Stop-gradient in model.forward() is the prevention mechanism.
    This class is the detection mechanism - run it independently of
    the loss, ideally on held-out data.
    """

    def __init__(self, std_threshold: float = 0.1):
        self.std_threshold = std_threshold

    def report(self, z: torch.Tensor, label: str = "latent", verbose: bool = True) -> dict:
        """Run collapse diagnostic on a batch of latent vectors.
        Pass a large diverse sample (hundreds of points) for reliable results.

        Args:
            z: shape (n_samples, latent_dim)
            label: string for print output
            verbose: if False, skip printing (still returns the dict) -
                     use this to log effective_rank every epoch without
                     spamming stdout.
        """
        with torch.no_grad():
            std = z.std(dim=0)
            mean_std  = std.mean().item()
            min_std = std.min().item()
            min_dim = std.argmin().item()
            collapsed = (std < self.std_threshold).sum().item()
            d = z.shape[1]

            # Effective rank - how many independent dimensions are actually used
            # Low effective rank means collapse even if per-dim variance looks ok
            try:
                z_c = z - z.mean(dim=0, keepdim=True)
                sv  = torch.linalg.svdvals(z_c)
                sv_norm  = sv / sv.sum()
                eff_rank = torch.exp(
                    -torch.sum(sv_norm * torch.log(sv_norm + 1e-12))
                ).item()
            except Exception:
                eff_rank = float('nan')

        if collapsed > d * 0.5:
            verdict = "SEVERE COLLAPSE"
        elif collapsed > 0:
            verdict = f"PARTIAL COLLAPSE ({collapsed} dims below threshold)"
        elif eff_rank < d * 0.4:
            verdict = f"REDUNDANCY WARNING (eff rank {eff_rank:.1f} / {d})"
        else:
            verdict = "HEALTHY"

        if verbose:
            print(f"\n[CollapseMonitor] {label}")
            print(f" mean std: {mean_std:.4f}")
            print(f" min std: {min_std:.4f}  (dim {min_dim})")
            print(f" collapsed dims: {collapsed} / {d}")
            print(f" effective rank: {eff_rank:.2f} / {d}")
            print(f" verdict: {verdict}")

        return {
            'mean_std': mean_std,
            'min_std': min_std,
            'collapsed_dims': collapsed,
            'effective_rank': eff_rank,
            'verdict': verdict,
        }