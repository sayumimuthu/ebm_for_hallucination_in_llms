"""v1 fusion: a small MLP energy over the four normalized energies.

.. important::
    The v0 linear fusion (``training.contrastive.train_linear_nce``,
    ``E(e) = w^T e + b``) can only represent a weighted *sum* of the four
    energies. Per the empirical result in the counterfactual-negatives
    addendum, even correctly trained and scale-normalized, it tops out at
    AUROC 0.712 on real validation labels — still below invariance energy
    used alone (0.723). A linear combination cannot represent an
    interaction like "high codelength surprisal AND low evidence support
    is especially damning, more than either alone suggests" — that needs
    a genuinely nonlinear function of the four energies. This module
    replaces ``f(e) = w^T e`` with a small MLP, keeping everything else
    identical: the same 4-dim normalized-energy input (``ENERGY_ORDER``),
    the same NCE contrastive objective, the same negatives. Any AUROC
    change is attributable to the added nonlinearity, not a confound.

    ``torch`` is already a core project dependency (used for generation),
    so this adds no new install.

Caveat: a 4 -> hidden -> 1 MLP has more parameters than the 5-parameter
linear model (e.g. hidden=16 gives 97 parameters) against the same
~200-example calibration set. Watch for a suspiciously large gap between
training-time NCE loss and held-out ``composite_auroc`` as a sign of
overfitting; nothing here currently guards against it beyond the L2
weight decay passed to the optimizer.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MLPEnergy(nn.Module):
    """``E_theta(e) = MLP(e)``: a 2-layer network replacing the linear
    ``w^T e + b``. Lower output = more trustworthy, matching the linear
    energy's convention (``composite_energy``, ``linear_energy``)."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_nce(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    hidden_dim: int = 16,
    steps: int = 300,
    lr: float = 0.01,
    l2: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[MLPEnergy, float]:
    """Mirrors ``train_linear_nce``'s contrastive objective and input
    shapes (``features_pos``: (N, D); ``features_neg``: (N, K, D) or
    (N, D), auto-expanded to K=1), but fits an ``MLPEnergy`` via torch
    autograd + Adam instead of a linear model via manual gradient descent.
    Returns the trained model (already in eval mode) and the final NCE
    training loss.
    """
    input_dim = features_pos.shape[-1] if features_pos.size else 4
    if features_pos.size == 0 or features_neg.size == 0:
        return MLPEnergy(input_dim, hidden_dim).eval(), 0.0
    if features_neg.ndim == 2:
        features_neg = features_neg[:, None, :]

    torch.manual_seed(seed)
    model = MLPEnergy(features_pos.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    pos = torch.tensor(features_pos, dtype=torch.float32)
    neg = torch.tensor(features_neg, dtype=torch.float32)  # (N, K, D)
    N, K, D = neg.shape

    final_loss = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        E_pos = model(pos)  # (N,)
        E_neg = model(neg.reshape(N * K, D)).reshape(N, K)
        # NCE: L = -log( exp(-E_pos) / (exp(-E_pos) + sum_j exp(-E_neg_j)) )
        #        = E_pos + logsumexp([-E_pos, -E_neg_1, ..., -E_neg_K])
        stacked_neg_energies = torch.cat([(-E_pos).unsqueeze(1), -E_neg], dim=1)  # (N, K+1)
        loss = (E_pos + torch.logsumexp(stacked_neg_energies, dim=1)).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if verbose and step % max(1, steps // 10) == 0:
            print(f"step={step} nce_loss={final_loss:.4f}")

    model.eval()
    return model, final_loss


def mlp_energy_score(model: MLPEnergy, features: np.ndarray) -> np.ndarray:
    """Score a batch of (N, D) normalized energy vectors, returning (N,)
    energies with no gradient tracking (for composite AUROC / inference)."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        return model(x).numpy()
