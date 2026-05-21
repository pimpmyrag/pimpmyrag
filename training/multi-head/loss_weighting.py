"""
Dynamic loss weighting strategies for multi-task learning.

Three modes:
  - fixed:       Traditional fixed lambdas (current behavior)
  - uncertainty:  Kendall et al. 2018 — learnable homoscedastic uncertainty
  - gradnorm:    Chen et al. ICML 2018 — gradient norm balancing

All strategies preserve ramp scheduling from bash (ramp_lambda * dynamic_weight).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, Optional

TASK_KEYS = [
    "boundary", "coarse", "fine",
    "svo_boundary", "svo", "role",
    "voice", "certainty", "morpho",
    "verb_ptr", "compat",
]


class FixedWeighting(nn.Module):
    """Passthrough — returns CLI lambdas unchanged."""

    def combine(self, raw_losses: Dict[str, torch.Tensor],
                ramp_lambdas: Dict[str, float]) -> torch.Tensor:
        device = next(iter(raw_losses.values())).device
        total = torch.tensor(0.0, device=device)
        for k in TASK_KEYS:
            if k in raw_losses and ramp_lambdas.get(k, 0.0) > 0:
                total = total + ramp_lambdas[k] * raw_losses[k]
        return total

    def get_effective_weights(self, ramp_lambdas: Dict[str, float]) -> Dict[str, float]:
        return {k: ramp_lambdas.get(k, 0.0) for k in TASK_KEYS}


class UncertaintyWeighting(nn.Module):
    """
    Kendall et al. 2018 — Multi-Task Learning Using Uncertainty to Weigh Losses.

    Each task i has a learnable log_sigma_i.
    Effective loss = loss_i / (2 * sigma_i^2) + log(sigma_i)
                   = loss_i * exp(-2*log_sigma_i) / 2 + log_sigma_i

    The ramp_lambda from bash multiplies the result, so ramp=0 still zeroes a task.
    Initial log_sigma is set so that effective weight ≈ initial_lambda.
    """

    def __init__(self, initial_lambdas: Dict[str, float]):
        super().__init__()
        # Init: effective_weight = 1/(2*sigma²) ≈ lambda → sigma² = 1/(2*lambda)
        # log_sigma = 0.5 * log(1/(2*lambda)) = -0.5 * log(2*lambda)
        self.log_sigmas = nn.ParameterDict()
        for k in TASK_KEYS:
            lam = max(initial_lambdas.get(k, 0.1), 0.01)
            init_val = -0.5 * math.log(2.0 * lam)
            self.log_sigmas[k] = nn.Parameter(torch.tensor(init_val))

    def combine(self, raw_losses: Dict[str, torch.Tensor],
                ramp_lambdas: Dict[str, float]) -> torch.Tensor:
        device = next(iter(raw_losses.values())).device
        total = torch.tensor(0.0, device=device)
        for k in TASK_KEYS:
            ramp = ramp_lambdas.get(k, 0.0)
            if k not in raw_losses or ramp == 0.0:
                continue
            log_s = self.log_sigmas[k].clamp(-4.0, 4.0)
            # loss_i / (2 * sigma²) + log(sigma)  =  loss_i * exp(-2*log_s) / 2 + log_s
            precision = torch.exp(-2.0 * log_s)
            weighted = 0.5 * precision * raw_losses[k] + log_s
            total = total + ramp * weighted
        return total

    def get_effective_weights(self, ramp_lambdas: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for k in TASK_KEYS:
            ramp = ramp_lambdas.get(k, 0.0)
            log_s = self.log_sigmas[k].item()
            log_s = max(-4.0, min(4.0, log_s))
            precision = math.exp(-2.0 * log_s)
            result[k] = ramp * 0.5 * precision
        return result


class GradNormWeighting(nn.Module):
    """
    GradNorm — Chen et al. ICML 2018.

    Adjusts per-task weights so gradient norms are balanced across tasks.
    Reference task = boundary (the most critical one).

    The ramp_lambda from bash multiplies the dynamic weight.
    """

    def __init__(self, initial_lambdas: Dict[str, float], alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha
        # Learnable log-weights, initialized from CLI lambdas
        self.log_lambdas = nn.ParameterDict()
        for k in TASK_KEYS:
            lam = max(initial_lambdas.get(k, 0.1), 0.01)
            self.log_lambdas[k] = nn.Parameter(torch.tensor(math.log(lam)))

        # Track initial losses for relative inverse training rate
        self.register_buffer("initial_losses",
                             torch.zeros(len(TASK_KEYS)), persistent=False)
        self._initial_set = False
        self._step_count = 0

    def combine(self, raw_losses: Dict[str, torch.Tensor],
                ramp_lambdas: Dict[str, float]) -> torch.Tensor:
        device = next(iter(raw_losses.values())).device
        total = torch.tensor(0.0, device=device)

        # Record initial losses (first step — only tasks with actual signal)
        if not self._initial_set:
            with torch.no_grad():
                any_real = False
                for i, k in enumerate(TASK_KEYS):
                    if k in raw_losses and raw_losses[k].item() > 1e-6:
                        self.initial_losses[i] = raw_losses[k].detach()
                        any_real = True
                    elif self.initial_losses[i] == 0:
                        self.initial_losses[i] = 1.0  # placeholder, updated later
                # Only mark as set once we have at least boundary + 2 others
                if any_real and self.initial_losses[0] > 1e-6:
                    # Update any still-placeholder values on subsequent calls
                    self._initial_set = True

        for k in TASK_KEYS:
            ramp = ramp_lambdas.get(k, 0.0)
            if k not in raw_losses or ramp == 0.0:
                continue
            dynamic_w = torch.exp(self.log_lambdas[k])
            total = total + ramp * dynamic_w * raw_losses[k]

        self._step_count += 1
        return total

    def gradnorm_loss(self, raw_losses: Dict[str, torch.Tensor],
                      shared_layer: nn.Module,
                      ramp_lambdas: Dict[str, float]) -> Optional[torch.Tensor]:
        """
        Compute the GradNorm loss to update log_lambdas.
        Call this separately, backward only on log_lambdas.

        Returns None if not enough active tasks.
        """
        device = next(iter(raw_losses.values())).device
        active_keys = [k for k in TASK_KEYS
                       if k in raw_losses
                       and ramp_lambdas.get(k, 0.0) > 0
                       and raw_losses[k].requires_grad
                       and raw_losses[k].item() > 1e-7]  # skip zero-loss tasks (empty mask batches)

        if len(active_keys) < 2:
            return None

        # Get shared layer weight for gradient computation
        shared_param = None
        for p in shared_layer.parameters():
            if p.requires_grad:
                shared_param = p
                break
        if shared_param is None:
            return None

        # Compute per-task gradient norms
        grad_norms = {}
        for k in active_keys:
            w_k = torch.exp(self.log_lambdas[k])
            weighted_loss = w_k * raw_losses[k]
            grad = torch.autograd.grad(
                weighted_loss, shared_param,
                retain_graph=True, create_graph=True
            )[0]
            grad_norms[k] = grad.norm()

        # Average gradient norm (target)
        avg_gn = torch.stack(list(grad_norms.values())).mean().detach()

        # Relative inverse training rates
        loss_ratios = {}
        for i, k in enumerate(TASK_KEYS):
            if k in active_keys:
                L0 = self.initial_losses[i].clamp(min=1e-6)
                L_now = raw_losses[k].detach().clamp(min=1e-6)
                loss_ratios[k] = (L_now / L0)

        if not loss_ratios:
            return None

        avg_ratio = torch.stack(list(loss_ratios.values())).mean()

        # GradNorm loss: ||G_k - avg_G * r_k^alpha||
        gn_loss = torch.tensor(0.0, device=device)
        for k in active_keys:
            r_k = (loss_ratios[k] / avg_ratio.clamp(min=1e-6)) ** self.alpha
            target = (avg_gn * r_k).detach()
            gn_loss = gn_loss + torch.abs(grad_norms[k] - target)

        return gn_loss

    def renormalize(self, ramp_lambdas: Dict[str, float]):
        """Renormalize weights to sum to same total as initial (after GradNorm update)."""
        with torch.no_grad():
            active = [k for k in TASK_KEYS if ramp_lambdas.get(k, 0.0) > 0]
            if not active:
                return
            total = sum(torch.exp(self.log_lambdas[k]).item() for k in active)
            target = sum(max(ramp_lambdas.get(k, 0.0), 0.01)
                         for k in active)
            if total > 0:
                ratio = target / total
                for k in active:
                    self.log_lambdas[k].data += math.log(ratio)

    def get_effective_weights(self, ramp_lambdas: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for k in TASK_KEYS:
            ramp = ramp_lambdas.get(k, 0.0)
            dynamic = math.exp(self.log_lambdas[k].item())
            result[k] = ramp * dynamic
        return result


def create_weighting(mode: str, initial_lambdas: Dict[str, float],
                     alpha: float = 1.5) -> nn.Module:
    if mode == "fixed":
        return FixedWeighting()
    elif mode == "uncertainty":
        return UncertaintyWeighting(initial_lambdas)
    elif mode == "gradnorm":
        return GradNormWeighting(initial_lambdas, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss weighting mode: {mode}")



