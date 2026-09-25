"""v1 fusion: nonlinear energy models over the four normalized energies.

.. important::
    The v0 linear fusion (``training.contrastive.train_linear_nce``,
    ``E(e) = w^T e + b``) can only represent a weighted *sum* of the four
    energies. Per the empirical result in the counterfactual-negatives
    addendum, even correctly trained and scale-normalized, it tops out at
    AUROC 0.712 on real validation labels — still below invariance energy
    used alone (0.723). A linear combination cannot represent an
    interaction like "high codelength surprisal AND low evidence support
    is especially damning, more than either alone suggests" — that needs
    a genuinely nonlinear function of the four energies.

    A first attempt, ``MLPEnergy`` (``E(e) = MLP(e)`` from scratch), made
    things *worse* (AUROC 0.660, below even equal-weighting) despite a
    lower training-time NCE loss than the linear model ever achieved. The
    reason: NCE only constrains each positive's score to beat *its own*
    few synthetic negatives — a **local** constraint. The linear model is
    so structurally constrained that satisfying this everywhere forces
    something globally sensible; a from-scratch MLP has enough freedom to
    satisfy every local constraint via a non-monotonic function that
    doesn't preserve any global ordering consistent with real labels —
    i.e. it overfits the contrastive *task* itself, not the sample size.

    ``ResidualMLPEnergy`` (``E(e) = w^T e + b + MLP_residual(e)``, residual
    branch zero-initialized) fixes this by construction: at
    initialization it is *exactly* the already-validated linear model;
    joint training can only introduce nonlinearity where doing so reduces
    the NCE loss more than the linear term alone already does. This tests
    "does adding nonlinearity on top of the linear structure help" rather
    than "does discarding the linear structure and starting over help."
    Prefer this over ``MLPEnergy`` unless you have a specific reason to
    test the from-scratch model.

    ``torch`` is already a core project dependency (used for generation),
    so this adds no new install.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MLPEnergy(nn.Module):
    """``E_theta(e) = MLP(e)``: a from-scratch 2-layer network replacing
    the linear ``w^T e + b``. See the module docstring for why
    ``ResidualMLPEnergy`` is preferred in practice. Lower output = more
    trustworthy, matching the linear energy's convention."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualMLPEnergy(nn.Module):
    """``E_theta(e) = w^T e + b + MLP_residual(e)``, with the residual
    branch's final layer zero-initialized so the model starts out
    identical to the linear energy. See the module docstring."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x) + self.residual(x)).squeeze(-1)

    def linear_component(self) -> Tuple[np.ndarray, float]:
        """The learned linear part alone, for comparison against
        ``train_linear_nce``'s ``(w, b)`` — how much of the fused score
        the model still explains linearly after joint training."""
        w = self.linear.weight.detach().numpy().ravel().astype(np.float32)
        b = float(self.linear.bias.detach().item())
        return w, b


def _run_nce_training(
    model: nn.Module,
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    steps: int,
    lr: float,
    l2: float,
    seed: int,
    verbose: bool,
) -> float:
    """Shared NCE training loop for any ``model(x) -> (N,) energy``
    module. Returns the final training loss (0.0 if given no data, in
    which case ``model`` is left at its initialization)."""
    if features_pos.size == 0 or features_neg.size == 0:
        return 0.0
    if features_neg.ndim == 2:
        features_neg = features_neg[:, None, :]

    torch.manual_seed(seed)
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
    return final_loss


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
    """Fits a from-scratch ``MLPEnergy``. See the module docstring for why
    ``train_residual_mlp_nce`` is preferred in practice."""
    input_dim = features_pos.shape[-1] if features_pos.size else 4
    model = MLPEnergy(input_dim, hidden_dim)
    final_loss = _run_nce_training(model, features_pos, features_neg, steps, lr, l2, seed, verbose)
    model.eval()
    return model, final_loss


def train_residual_mlp_nce(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    hidden_dim: int = 16,
    steps: int = 300,
    lr: float = 0.01,
    l2: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[ResidualMLPEnergy, float]:
    """Fits ``ResidualMLPEnergy`` (linear + zero-initialized nonlinear
    correction, jointly trained) — the recommended v1 fusion model. Same
    NCE objective and input shapes as ``train_linear_nce``/``train_mlp_nce``."""
    input_dim = features_pos.shape[-1] if features_pos.size else 4
    model = ResidualMLPEnergy(input_dim, hidden_dim)
    final_loss = _run_nce_training(model, features_pos, features_neg, steps, lr, l2, seed, verbose)
    model.eval()
    return model, final_loss


def mlp_energy_score(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Score a batch of (N, D) normalized energy vectors, returning (N,)
    energies with no gradient tracking (for composite AUROC / inference).
    Works for any of this module's model classes."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        return model(x).numpy()
