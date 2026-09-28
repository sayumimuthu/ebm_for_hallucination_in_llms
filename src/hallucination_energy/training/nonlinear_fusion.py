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
    branch zero-initialized) fixes the *starting point* by construction:
    at initialization it is exactly the already-validated linear model.
    But a second real run showed this isn't enough on its own: even
    starting from a good linear solution, joint NCE training can still
    drift the *linear* component itself toward a sign-flipped, locally
    contrastive-optimal but globally label-inconsistent solution (observed:
    the codelength weight ended up strongly *negative*, opposite its own
    standalone AUROC direction), landing below equal-weighting again
    (0.671 vs 0.697). Zero-init only fixes where training starts, not
    where the (label-blind) NCE objective wanders during training.

    ``monitor_fn`` (in ``train_mlp_nce``/``train_residual_mlp_nce``) fixes
    the actual gap: track a real-label metric (typically
    ``evaluation.diagnostics.model_composite_auroc``) periodically during
    training and keep the best-scoring checkpoint instead of whichever
    step training happens to end on. This is standard validation-based
    checkpoint selection, not new training signal — the model never sees
    labels in its gradient — but it stops "train for N steps and hope the
    last step is good" from silently reporting a worse-than-best result.

    ``torch`` is already a core project dependency (used for generation),
    so this adds no new install.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional, Tuple

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
    monitor_fn: Optional[Callable[[nn.Module], float]] = None,
    monitor_every: int = 10,
) -> Tuple[float, Optional[float]]:
    """Shared NCE training loop for any ``model(x) -> (N,) energy``
    module. Returns ``(final_loss, best_monitor_score)`` (0.0/None if given
    no data, in which case ``model`` is left at its initialization).

    If ``monitor_fn`` is given, it's called every ``monitor_every`` steps
    (and after the last step) with the model in eval mode; whichever
    snapshot scores highest is restored into ``model`` before returning —
    see the module docstring for why this matters (NCE training can drift
    away from a good solution with nothing to stop it, since the loss
    itself never sees real labels).
    """
    if features_pos.size == 0 or features_neg.size == 0:
        return 0.0, None
    if features_neg.ndim == 2:
        features_neg = features_neg[:, None, :]

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    pos = torch.tensor(features_pos, dtype=torch.float32)
    neg = torch.tensor(features_neg, dtype=torch.float32)  # (N, K, D)
    N, K, D = neg.shape

    best_score: Optional[float] = None
    best_state = None
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

        if monitor_fn is not None and (step % monitor_every == 0 or step == steps - 1):
            model.eval()
            score = monitor_fn(model)
            model.train()
            if best_score is None or score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return final_loss, best_score


def train_mlp_nce(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    hidden_dim: int = 16,
    steps: int = 300,
    lr: float = 0.01,
    l2: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
    monitor_fn: Optional[Callable[[nn.Module], float]] = None,
    monitor_every: int = 10,
) -> Tuple[MLPEnergy, float, Optional[float]]:
    """Fits a from-scratch ``MLPEnergy``. See the module docstring for why
    ``train_residual_mlp_nce`` is preferred in practice. Returns
    ``(model, final_loss, best_monitor_score)``."""
    input_dim = features_pos.shape[-1] if features_pos.size else 4
    model = MLPEnergy(input_dim, hidden_dim)
    final_loss, best_score = _run_nce_training(
        model, features_pos, features_neg, steps, lr, l2, seed, verbose, monitor_fn, monitor_every
    )
    model.eval()
    return model, final_loss, best_score


def train_residual_mlp_nce(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    hidden_dim: int = 16,
    steps: int = 300,
    lr: float = 0.01,
    l2: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
    monitor_fn: Optional[Callable[[nn.Module], float]] = None,
    monitor_every: int = 10,
) -> Tuple[ResidualMLPEnergy, float, Optional[float]]:
    """Fits ``ResidualMLPEnergy`` (linear + zero-initialized nonlinear
    correction, jointly trained) — the recommended v1 fusion model. Same
    NCE objective and input shapes as ``train_linear_nce``/``train_mlp_nce``.
    Returns ``(model, final_loss, best_monitor_score)``."""
    input_dim = features_pos.shape[-1] if features_pos.size else 4
    model = ResidualMLPEnergy(input_dim, hidden_dim)
    final_loss, best_score = _run_nce_training(
        model, features_pos, features_neg, steps, lr, l2, seed, verbose, monitor_fn, monitor_every
    )
    model.eval()
    return model, final_loss, best_score


def mlp_energy_score(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Score a batch of (N, D) normalized energy vectors, returning (N,)
    energies with no gradient tracking (for composite AUROC / inference).
    Works for any of this module's model classes."""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        return model(x).numpy()
