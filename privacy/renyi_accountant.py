"""
privacy/renyi_accountant.py — Per-task Rényi DP accounting for VFL-MTL.

Wraps opacus.accountants.RDPAccountant (one instance per task) so each task
can be tracked independently. In stratified mode each task has a different σ
and therefore a different ε budget consumption rate.

Additionally accumulates gradient cosine-similarity values logged during
training (from fl/server.py::compute_task_gradient_similarity) to estimate
the cross-task coupling matrix ρ used in the multi-task label inference bound.

Reference: Mironov (2017) Rényi DP, CSF.  Yousefpour et al. (2021) Opacus.
"""

from __future__ import annotations

from opacus.accountants import RDPAccountant

_TASKS = ("ihm", "decomp", "pheno")
_COUPLING_KEYS = (
    "grad_sim_ihm_decomp",
    "grad_sim_ihm_pheno",
    "grad_sim_decomp_pheno",
)


class RenyiAccountant:
    """
    Per-task Rényi DP accountant for VFL-MTL training.

    Usage
    -----
    # Create once before training:
    accountant = RenyiAccountant()

    # After each training round (uniform mode):
    accountant.step(noise_multiplier=sigma, sample_rate=bs/N, num_steps=n_batches)

    # After each training round (stratified mode):
    accountant.step_stratified(
        sigma_map={'ihm': s1, 'decomp': s2, 'pheno': s3},
        sample_rate=bs/N, num_steps=n_batches
    )

    # Query privacy budget consumed so far:
    eps = accountant.get_epsilon(delta=1e-5)  # {'ihm': ε1, 'decomp': ε2, 'pheno': ε3}

    # Log gradient similarity for coupling matrix (from compute_task_gradient_similarity):
    accountant.log_grad_sim({'grad_sim_ihm_decomp': 0.3, ...})

    # After ≥1 logged round, retrieve coupling matrix:
    rho = accountant.cross_task_coupling_matrix()
    """

    def __init__(self) -> None:
        self._accountants: dict[str, RDPAccountant] = {
            t: RDPAccountant() for t in _TASKS
        }
        self._grad_sim_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Stepping

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
        task: str | None = None,
    ) -> None:
        """
        Advance the privacy accountant by num_steps steps.

        task=None  → step all task accountants (uniform mode, same σ for all).
        task='ihm' → step only the IHM accountant.

        Opacus merges consecutive identical steps internally (O(1) history),
        so calling step(num_steps=N) is equivalent to N individual calls.
        """
        targets = _TASKS if task is None else (task,)
        for t in targets:
            for _ in range(num_steps):
                self._accountants[t].step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )

    def step_stratified(
        self,
        sigma_map: dict[str, float],
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """
        Step each task accountant with its own σ (stratified mode).

        Parameters
        ----------
        sigma_map : {'ihm': σ_ihm, 'decomp': σ_decomp, 'pheno': σ_pheno}
        sample_rate : batch_size / N_train
        num_steps   : number of batches in the round
        """
        for t, sigma in sigma_map.items():
            if t in self._accountants:
                for _ in range(num_steps):
                    self._accountants[t].step(
                        noise_multiplier=sigma,
                        sample_rate=sample_rate,
                    )

    # ------------------------------------------------------------------
    # Querying

    def get_epsilon(self, delta: float = 1e-5) -> dict[str, float]:
        """
        Return {task: ε_k} for each task at the given δ.

        Returns nan for tasks whose accountant has no history (not yet stepped).
        """
        result: dict[str, float] = {}
        for t, acc in self._accountants.items():
            try:
                result[t] = float(acc.get_epsilon(delta=delta))
            except Exception:
                result[t] = float("nan")
        return result

    # ------------------------------------------------------------------
    # Gradient coupling matrix

    def log_grad_sim(self, grad_sim_dict: dict[str, float]) -> None:
        """
        Accumulate one round of gradient cosine-similarity values.

        Expects the dict format produced by
        fl/server.py::compute_task_gradient_similarity():
          {'grad_sim_ihm_decomp': float, 'grad_sim_ihm_pheno': float,
           'grad_sim_decomp_pheno': float}
        """
        self._grad_sim_history.append(
            {k: v for k, v in grad_sim_dict.items() if k in _COUPLING_KEYS}
        )

    def cross_task_coupling_matrix(self) -> dict[str, float]:
        """
        Mean gradient cosine similarity (ρ proxy) across all logged rounds.

        Used to estimate cross-task coupling for the multi-task label inference
        bound (Liu et al. 2022 extension). Returns empty dict if no data logged.

        Keys match fl/server.py::compute_task_gradient_similarity() output:
          'grad_sim_ihm_decomp', 'grad_sim_ihm_pheno', 'grad_sim_decomp_pheno'
        """
        if not self._grad_sim_history:
            return {}

        result: dict[str, float] = {}
        for key in _COUPLING_KEYS:
            vals = [
                d[key] for d in self._grad_sim_history
                if key in d and d[key] == d[key]   # skip NaN
            ]
            result[key] = float(sum(vals) / len(vals)) if vals else float("nan")
        return result

    def coupling_epsilon_inflation(self, delta: float = 1e-5) -> float:
        """
        Additive privacy inflation from multi-task composition.

        In standard composition: ε_total ≤ Σ_k ε_k.
        Inflation = Σ_k ε_k − max_k ε_k quantifies the extra cost beyond the
        worst-case task budget — attributable to tasks sharing the MMoE experts.

        Returns 0.0 if no steps have been recorded yet.
        """
        eps = self.get_epsilon(delta=delta)
        valid = [v for v in eps.values() if v == v]    # skip NaN
        if not valid:
            return 0.0
        return float(sum(valid) - max(valid))

    # ------------------------------------------------------------------

    @property
    def n_logged_rounds(self) -> int:
        """Number of rounds for which gradient similarity has been logged."""
        return len(self._grad_sim_history)

    def __repr__(self) -> str:
        try:
            eps = self.get_epsilon()
            eps_str = ", ".join(f"{t}={v:.3f}" for t, v in eps.items())
        except Exception:
            eps_str = "not stepped"
        return f"RenyiAccountant(epsilon={{{eps_str}}}, logged_rounds={self.n_logged_rounds})"
