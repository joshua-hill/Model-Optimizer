# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aumann-Shapley sensitivity scoring for AutoQuantize.

The ``"aumann_shapley"`` method scores each (runtime group, candidate format) with a damage
attribution in nats of KL divergence against the model's own reference outputs. The
reference keeps any fixed or forced-single-format groups quantized, so every score and quote
is incremental KL relative to the resolved baseline recorded in
``damage_model["damage_reference"]`` (``{"type": "unquantized"}`` when nothing is pinned). At each midpoint node
``t = (k + 1/2) / num_path_nodes`` of the joint quantization path, every scored module
propagates ``y + t * (Q(y) - y)`` using detached local-replay differences (the module's own
forward re-run with the candidate's quantizers active), and one backward pass accumulates
``<dL/dy, Q(y) - y>`` per (group, format). A path integral is required for a KL loss because
``dKL = 0`` exactly at the unquantized point, which is why the gradient method must square
its Taylor term (a Fisher proxy needing a labeled loss) while this method is label-free.

Per batch, scoring costs one reference forward, one aggressive-corner forward, and one
forward+backward per (candidate format, path node) -- independent of how many configurations
the solver later considers. The measured corner anchors a coverage link
``damage = c * (1 - exp(-sum(b)))``: a fixed-point inversion converts attributions into
per-group log-headroom ``b`` written into ``candidate_stats["scores"]``, so the standard
solve is the coverage-optimal allocation and the chosen recipe carries a ``predicted_damage``
quote in measured units, with validity recorded in ``damage_model["valid"]`` and
``approximation_flags``. Solver scores are additionally projected onto the monotone
compression ladder by raising more-aggressive entries, so quotes are conservative:
``projected_corner_damage`` may exceed the measured corner, while ``b_unprojected`` keeps
the exactly corner-anchored inversion. The quote is an internal-model estimate, not a bound
on realized deployment KL.

This is an engineering variant of the method in https://arxiv.org/abs/2607.12266 -- validated
empirically against it -- with known deviations: the paper perturbs weight and input
quantizer boundaries separately (here: module outputs, one local replay covering both), uses
more path nodes (here: 1 by default, based on measured cost/quality parity), and freezes MoE
routing (here: routing shifts along the path count as damage). ``completeness`` in the
damage model is an empirical diagnostic of attribution quality, not a guarantee. Further
limitations: a module invoked multiple times in one forward keeps only its last invocation's
replay data; groups with differing candidate ladders make the joint coverage interpretation
approximate; the inversion uses the single-midpoint surrogate for any node count. Candidates
whose measured attribution is non-finite are removed from the search space (recorded in
``damage_model["excluded_candidates"]``); a non-finite corner invalidates the fit, as does
removing any group's most aggressive format, since the measured corner ran with it active.
A group left with only non-finite candidates keeps its least aggressive entry as a forced
choice (``damage_model["forced_candidates"]``) and the search reports unsatisfied.

Method-specific ``method_options``:

- ``num_path_nodes`` (default 1): quadrature nodes for the path integral.
- ``damage_link`` (default ``"coverage"``): ``"coverage"`` or ``"additive"`` (raw scores).
- ``solver`` (default ``"lp"``): budget-mode solver -- ``"lp"`` (default, exact) or ``"dp"``
  (deterministic, grid-approximate).
- ``max_predicted_damage`` (default None): minimize weight cost subject to predicted damage <=
  this bound (mutually exclusive with an ``effective_bits`` constraint). Always solved on
  the grid-approximate DP path with conservative rounding, regardless of ``solver``.
"""

import gc
import math
import types

import numpy as np
import torch

from modelopt.torch.opt.searcher import SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import (
    create_param_grad_clear_hook,
    print_rank_0,
    report_memory,
    warn_rank_0,
)
from modelopt.torch.utils.distributed import DistributedProcessGroup

from .algorithms import (
    AUTO_QUANTIZE_SEARCHERS,
    AutoQuantizeGradientSearcher,
    QuantRecipe,
    QuantRecipeHparam,
    _AutoQuantizeBaseSearcher,
    _get_kl_div_loss,
    _get_lm_head,
    _get_log_prob,
)

__all__ = ["AutoQuantizeAumannShapleySearcher"]

_DP_GRID = 4096


def _as_seed_coverage(attributions, c, *, p=0.5, iters=200, tol=1e-10, damping=0.5):
    """Invert ``AS_i = c a_i prod_{j != i} (1 - p a_j)`` for break rates given ceiling ``c``.

    Returns ``(a, b = -log(1 - a), converged)``. Zero attributions map to exactly zero break
    rates; arbitrarily small positive attributions map to proportionally small ones. The
    system is infeasible when the attribution mass is too large for the ceiling; the
    iteration then diverges to the clip and the caller should inflate ``c`` and retry (see
    :func:`_anchor_ceiling`).
    """
    attributions = np.maximum(np.asarray(attributions, dtype=float), 0.0)
    if not (attributions > 0).any():
        zeros = np.zeros_like(attributions)
        return zeros, zeros.copy(), True
    a = np.clip(attributions / max(c, 1e-12), 0.0, 0.999)
    converged = False
    for _ in range(iters):
        log_prod = np.log(np.clip(1.0 - p * a, 1e-9, None)).sum()
        discount = np.exp(log_prod - np.log(np.clip(1.0 - p * a, 1e-9, None)))
        new = np.clip(attributions / (max(c, 1e-12) * np.clip(discount, 1e-6, None)), 0.0, 0.999)
        nxt = damping * a + (1.0 - damping) * new
        delta = float(np.abs(nxt - a).max())
        a = nxt
        if delta < tol:
            converged = bool(a.max() < 0.995)
            break
    b = -np.log(np.clip(1.0 - a, 1e-9, None))
    return a, b, converged


def _anchor_ceiling(as_by_key, f_corner, corner_mask_by_key, max_inflation=10.0):
    """Invert per-format attributions into break rates with one corner-anchored ceiling.

    The ceiling starts at the measured corner damage and is inflated minimally until the
    inversion converges for every format; ``b`` is then rescaled by ``kappa`` so the link stays
    exact at the corner. Returns ``(c, b_by_key, kappa, inflation, converged)``. Raises
    ``ValueError`` on non-finite inputs (callers must screen measurements first).
    """
    if not math.isfinite(f_corner) or not all(
        bool(np.isfinite(v).all()) for v in as_by_key.values()
    ):
        raise ValueError("corner damage and attributions must be finite")
    f_corner = max(float(f_corner), 1e-12)
    c = f_corner
    # The exit conditions bound this well below the cap; the cap is a backstop only.
    for _ in range(1 + math.ceil(math.log(max_inflation) / math.log(1.3))):
        inversions = {k: _as_seed_coverage(v, c=c) for k, v in as_by_key.items()}
        if all(conv for _a, _b, conv in inversions.values()) or c > max_inflation * f_corner:
            break
        c *= 1.3
    converged = all(conv for _a, _b, conv in inversions.values())
    b_by_key = {k: inv[1] for k, inv in inversions.items()}

    def corner_b_sum():
        return sum(float(b_by_key[k][mask].sum()) for k, mask in corner_mask_by_key.items())

    kappa = 1.0
    tolerance = 0.01 * f_corner
    if abs(_predict_damage(c, corner_b_sum()) - f_corner) > tolerance:
        # Exact anchoring needs strict headroom above the corner (kappa solves
        # c * (1 - exp(-kappa * sum(b_corner))) == f_corner, impossible at c == f_corner).
        if c < 1.01 * f_corner:
            c = 1.01 * f_corner
            inversions = {k: _as_seed_coverage(v, c=c) for k, v in as_by_key.items()}
            converged = all(conv for _a, _b, conv in inversions.values())
            b_by_key = {k: inv[1] for k, inv in inversions.items()}
        total = corner_b_sum()
        if total > 0:
            kappa = -np.log(1.0 - f_corner / c) / total
            b_by_key = {k: b * kappa for k, b in b_by_key.items()}
    anchored = abs(_predict_damage(c, corner_b_sum()) - f_corner) <= tolerance
    return float(c), b_by_key, float(kappa), float(c / f_corner), converged and anchored


def _predict_damage(c, b_sum):
    return float(c * (1.0 - np.exp(-max(float(b_sum), 0.0))))


def _mckp_max_value(values, costs, budget):
    """Exact multiple-choice knapsack: max ``sum(values)`` s.t. ``sum(costs) <= budget``.

    Choice 0 of each row must have zero cost so every row has a feasible pick.
    """
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=np.int64)
    n, num_choices = values.shape
    if (costs[:, 0] != 0).any():
        raise ValueError("choice 0 must have zero cost")
    grid = int(budget)
    dp = np.zeros(grid + 1)
    choice = np.zeros((n, grid + 1), dtype=np.int32)
    candidates = np.empty((num_choices, grid + 1))
    for i in range(n):
        for k in range(num_choices):
            cost = int(costs[i, k])
            if cost > grid:
                candidates[k] = -np.inf
                continue
            candidates[k, :cost] = -np.inf
            candidates[k, cost:] = dp[: grid + 1 - cost] + values[i, k]
        best = candidates.argmax(axis=0)
        choice[i] = best
        dp = candidates[best, np.arange(grid + 1)]
    selection = np.zeros(n, dtype=int)
    g = grid
    for i in range(n - 1, -1, -1):
        k = int(choice[i, g])
        selection[i] = k
        g -= int(costs[i, k])
    return selection, float(dp[grid])


class AutoQuantizeAumannShapleySearcher(AutoQuantizeGradientSearcher):
    """AutoQuantize searcher scoring with Aumann-Shapley damage attributions (see module doc)."""

    method_name = "aumann_shapley"
    method_options_keys = frozenset(
        {"num_path_nodes", "damage_link", "solver", "max_predicted_damage"}
    )

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        config = super().default_search_config
        config.update(
            {
                "num_path_nodes": 1,
                "damage_link": "coverage",
                "solver": "lp",
                "max_predicted_damage": None,
            }
        )
        return config

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        state = super().default_state_dict
        state["damage_model"] = None
        state["scoring_signature"] = None
        return state

    def _damage_reference(self, no_quant) -> dict:
        """The baseline every score, corner, and quote is measured against.

        Groups pinned to one quantized format -- via ``fixed_quantization_config`` or a
        single-candidate ``module_search_spaces`` entry with ``allow_no_quant=False`` -- stay
        active during the reference passes, so all damage values are INCREMENTAL KL relative
        to this resolved baseline, not total degradation from the unquantized model.
        """
        forced_groups = {
            name: str(stat["formats"][0])
            for name, stat in self.candidate_stats.items()
            if len(stat["formats"]) == 1 and stat["formats"][0] != no_quant
        }
        if getattr(self, "fixed_quantization_config", None) is None and not forced_groups:
            return {"type": "unquantized"}
        return {
            "type": "quantized_baseline",
            "fixed_quantization_config_signature": getattr(
                self, "fixed_quantization_config_signature", None
            ),
            "forced_groups": forced_groups,
        }

    def _current_scoring_signature(self) -> dict:
        # Settings that change what the stored scores MEAN; solver and max_predicted_damage only
        # change how they are solved and may differ across a checkpoint resume.
        return {
            "version": 1,
            "path_variant": "module_output_replay_v1",
            "num_path_nodes": int(self.config["num_path_nodes"]),
            "damage_link": self.config["damage_link"],
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        for ignored_key in ["score_func", "loss_func", "forward_backward_step"]:
            if config.get(ignored_key) is not None:
                warn_rank_0(
                    f"`{ignored_key}` is ignored for Aumann-Shapley `auto_quantize`: the loss "
                    "is fixed to KL divergence against the model's own reference outputs."
                )
                config.pop(ignored_key)
        config = _AutoQuantizeBaseSearcher.sanitize_search_config(self, config)
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for Aumann-Shapley `auto_quantize`. "
            "`forward_step(model, data)` should return model logits."
        )
        nodes = config["num_path_nodes"]
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1:
            raise ValueError(f"num_path_nodes must be an integer >= 1, got {nodes!r}")
        if config["damage_link"] not in ("coverage", "additive"):
            raise ValueError(
                f"damage_link must be 'coverage' or 'additive', got {config['damage_link']!r}"
            )
        if config["solver"] not in ("lp", "dp"):
            raise ValueError(f"solver must be 'lp' or 'dp', got {config['solver']!r}")
        bound = config["max_predicted_damage"]
        if bound is not None and (
            not isinstance(bound, (int, float))
            or isinstance(bound, bool)
            or not math.isfinite(bound)
            or bound <= 0
        ):
            raise ValueError(
                f"max_predicted_damage must be a finite positive number, got {bound!r}"
            )
        return config

    def validate_search_input(self, constraints, config) -> None:
        """Reject ambiguous target combinations (runs before any model mutation)."""
        if (
            config.get("max_predicted_damage") is not None
            and (constraints or {}).get("effective_bits") is not None
        ):
            raise ValueError(
                "Provide either constraints['effective_bits'] or "
                "method_options['max_predicted_damage'], not both: the damage-bound mode "
                "solves for the minimum effective bits itself."
            )

    def before_search(self) -> None:
        """Prepare the model for search; damage-bound mode supplies the bit budget itself."""
        if self.config["max_predicted_damage"] is not None:
            self.validate_search_input(self.constraints, self.config)
            self.constraints = {"effective_bits": 16.0, **self.constraints}
        # Stored scores are only reusable when their meaning is unchanged (see
        # _current_scoring_signature); solver/SLA re-solves are allowed on resume.
        current_signature = self._current_scoring_signature()
        restored_signature = getattr(self, "scoring_signature", None)
        if self.candidate_stats and restored_signature not in (None, current_signature):
            raise ValueError(
                f"Checkpoint scoring signature {restored_signature} does not match the "
                f"current search config {current_signature}. Use a different checkpoint path."
            )
        self.scoring_signature = current_signature
        super().before_search()

    def _configurable_hparams(self) -> list[QuantRecipeHparam]:
        return [
            hparam
            for _name, hparam in named_hparams(self.model, unique=True)
            if isinstance(hparam, QuantRecipeHparam) and hparam.is_configurable
        ]

    @torch.enable_grad()
    def _estimate_auto_quantize_scores(self, is_param_grad_enabled):
        model = self.model
        no_quant = QuantRecipe(quant_cfg=None)
        lm_head = _get_lm_head(model)
        if self._loss_is_vocab_sharded():
            # The score passes backprop through the KL loss; the vocab-sharded log-softmax
            # uses in-place collectives that autograd cannot differentiate through.
            raise NotImplementedError(
                "aumann_shapley scoring does not support vocab-sharded (Megatron "
                "tensor-parallel) losses yet. Use method='gradient' with a Megatron loss_func."
            )
        num_nodes = int(self.config["num_path_nodes"])

        hparams = self._configurable_hparams()
        recipes = sorted({r for h in hparams for r in h.choices if r != no_quant})

        self._as_recipe: QuantRecipe | None = None
        self._as_t: float = 0.0
        self._corner_kl_sum: torch.Tensor | None = None
        self._score_tokens: int = 0

        def set_all_hparams(recipe_of) -> None:
            for hparam in hparams:
                hparam.active = recipe_of(hparam)

        def score_estimate_forward(module, input, *args, **kwargs):
            recipe = self._as_recipe
            if recipe is None:
                # Reference/corner passes manage hparam state in the outer loop.
                return module._forward_original(input, *args, **kwargs)

            module._as_diffs = None
            for hparam in module._hparams_for_scoring:
                if hparam.is_configurable:
                    hparam.active = no_quant
            output = module._forward_original(input, *args, **kwargs)
            base = output[0] if isinstance(output, tuple) else output

            diffs: dict[QuantRecipeHparam, torch.Tensor] = {}
            diff_total = None
            with torch.no_grad():
                for hparam in module._hparams_for_scoring:
                    if not hparam.is_configurable or recipe not in hparam.choices:
                        continue
                    hparam.active = recipe
                    quant_output = module._forward_original(input, *args, **kwargs)
                    hparam.active = no_quant
                    quant_output = (
                        quant_output[0] if isinstance(quant_output, tuple) else quant_output
                    )
                    diff = (quant_output - base).detach()
                    diffs[hparam] = diff
                    diff_total = diff if diff_total is None else diff_total + diff

            if diff_total is None:
                return output
            if torch.is_grad_enabled() and base.requires_grad:
                module._as_diffs = diffs
            # The shift must run in BOTH reentrant-checkpointing passes (identical streams);
            # only the diff caching above gates on grad being enabled.
            shifted = base + self._as_t * diff_total
            if isinstance(output, tuple):
                return (shifted, *output[1:])
            return shifted

        def backward_hook(module, grad_input, grad_output):
            # Consume-then-clear keeps modules shared across checkpoint segments correct; a
            # module invoked twice in ONE forward keeps only its last invocation's diffs.
            diffs = getattr(module, "_as_diffs", None)
            module._as_diffs = None
            recipe = self._as_recipe
            if not diffs or recipe is None or grad_output[0] is None:
                return
            with torch.no_grad():
                grad = grad_output[0].float()
                for hparam, diff in diffs.items():
                    contribution = (grad * diff.float()).sum() / num_nodes
                    if hparam._importance_dict[recipe][module] is None:
                        hparam._importance_dict[recipe][module] = contribution
                    else:
                        hparam._importance_dict[recipe][module] += contribution

        def setup_params_for_score_estimation(name, param, params_metadata, enable_grad=True):
            params_metadata[name] = {"requires_grad": param.requires_grad}
            param.requires_grad = enable_grad
            if not enable_grad:
                return
            accum_grad, handle = create_param_grad_clear_hook(param)
            params_metadata[name]["accum_grad"] = accum_grad
            params_metadata[name]["handle"] = handle

        def setup_module_for_score_estimation(module):
            module._forward_original = module.forward
            module.forward = types.MethodType(score_estimate_forward, module)
            module._backward_hook_handle = module.register_full_backward_hook(backward_hook)

        def cleanup_module_after_score_estimation(module):
            module.forward = module._forward_original
            del module._forward_original
            module._backward_hook_handle.remove()
            if hasattr(module, "_as_diffs"):
                del module._as_diffs

        def cleanup_params_after_score_estimation(name, param, params_metadata):
            param.requires_grad = params_metadata[name]["requires_grad"]
            handle = params_metadata[name].get("handle")
            if handle is not None:
                handle.remove()

        score_modules = []
        seen: set[int] = set()
        for _name, module in model.named_modules():
            if (
                hasattr(module, "_hparams_for_scoring")
                and any(h.is_configurable for h in module._hparams_for_scoring)
                and id(module) not in seen
            ):
                setup_module_for_score_estimation(module)
                score_modules.append(module)
                seen.add(id(module))

        params_metadata: dict = {}
        for name, param in model.named_parameters():
            setup_params_for_score_estimation(
                name, param, params_metadata, is_param_grad_enabled(name, model)
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_memory("AutoQuantize(aumann_shapley): starting score estimation, ")

        def score_step(model_, data):
            self._as_recipe = None
            set_all_hparams(lambda _h: no_quant)
            with torch.no_grad():
                ref_logits = self.config["forward_step"](model_, data)
                ref_logprob = _get_log_prob(ref_logits, lm_head=lm_head).detach()
                self._score_tokens += int(ref_logprob.numel() // ref_logprob.shape[-1])
                del ref_logits

                set_all_hparams(lambda h: h.choices[0])
                corner_logits = self.config["forward_step"](model_, data)
                corner_loss = _get_kl_div_loss(ref_logprob, corner_logits, lm_head).detach()
                self._corner_kl_sum = (
                    corner_loss
                    if self._corner_kl_sum is None
                    else self._corner_kl_sum + corner_loss
                )
                set_all_hparams(lambda _h: no_quant)
                del corner_logits

            for recipe in recipes:
                for node in range(num_nodes):
                    self._as_t = (node + 0.5) / num_nodes
                    self._as_recipe = recipe
                    logits = self.config["forward_step"](model_, data)
                    loss = _get_kl_div_loss(ref_logprob, logits, lm_head)
                    loss.backward()
                    del logits, loss
            self._as_recipe = None

        try:
            self._run_func(
                score_step,
                num_iters=self.config["num_score_steps"],
                desc="Estimating aumann_shapley scores",
            )
        finally:
            for module in score_modules:
                cleanup_module_after_score_estimation(module)
            for name, param in model.named_parameters():
                cleanup_params_after_score_estimation(name, param, params_metadata)
            del params_metadata
            gc.collect()

        if torch.cuda.is_available():
            report_memory("AutoQuantize(aumann_shapley): after score estimation")

    def _loss_is_vocab_sharded(self) -> bool:
        lm_head = _get_lm_head(self.model)
        parallel_state = getattr(lm_head, "parallel_state", None) if lm_head is not None else None
        return parallel_state is not None and parallel_state.tensor_parallel_group.is_initialized()

    def _reduce_loss_scalar(self, value: float) -> float:
        # A loss scalar is sharded over DP (disjoint batches) and, only when the loss itself
        # is vocab-sharded, over TP; it is REPLICATED across EP ranks (unlike per-module
        # importances, which get_score sums over all three groups).
        module = self._any_score_parallel_module()
        if module is None:
            return value
        parallel_state = module.parallel_state
        sum_groups = [parallel_state.data_parallel_group]
        if self._loss_is_vocab_sharded():
            sum_groups.append(parallel_state.tensor_parallel_group)
        value = DistributedProcessGroup.get_dist_syncd_obj(value, sum_groups, sum)
        return DistributedProcessGroup.get_dist_syncd_obj(
            value, [parallel_state.expert_model_parallel_group], lambda a: a[0]
        )

    def _reduce_token_count(self, count: int) -> int:
        module = self._any_score_parallel_module()
        if module is None:
            return count
        return DistributedProcessGroup.get_dist_syncd_obj(
            count, [module.parallel_state.data_parallel_group], sum
        )

    def _any_score_parallel_module(self):
        for hparam in self._configurable_hparams():
            for module in hparam.score_modules:
                if getattr(module, "parallel_state", None) is not None:
                    return module
        return None

    def _exclude_non_finite_candidates(
        self, no_quant
    ) -> tuple[dict[str, list[str]], dict[str, str], bool]:
        """Drop candidates whose measured attribution is non-finite.

        A non-finite score is a measurement of a format that destroys the reference output
        (or of a numerically broken pass); zeroing or clamping it would make that candidate
        look cheap to the solver, so it is removed from its group's ladder instead. When a
        constraint is only reachable through removed candidates, the solve reports
        ``is_satisfied=False``. Returns ``(excluded, forced, corner_removed)``:

        - ``excluded``: the removed format labels per group.
        - ``forced``: groups with neither a finite candidate nor a no-quant fallback, mapped
          to their retained format -- the least aggressive entry, kept as the forced choice
          with a neutral solver score (the non-finite raw measurement is preserved). The
          caller reports such searches unsatisfied and invalidates the damage model.
        - ``corner_removed``: True when some group lost its most aggressive format. The
          measured corner ran with that format active, so the anchor no longer corresponds
          to the candidate corner and the caller invalidates the coverage fit.
        """
        excluded: dict[str, list[str]] = {}
        forced: dict[str, str] = {}
        corner_removed = False
        for name, stat in self.candidate_stats.items():
            if stat.get("is_fixed", False) or len(stat["formats"]) <= 1:
                continue
            keep = [
                index
                for index, (recipe, raw) in enumerate(
                    zip(stat["formats"], stat["raw_scores"], strict=True)
                )
                if recipe == no_quant or math.isfinite(raw)
            ]
            if len(keep) == len(stat["formats"]):
                continue
            if not keep:
                keep = [len(stat["formats"]) - 1]
                stat["scores"][keep[0]] = 0.0
                forced[name] = str(stat["formats"][keep[0]])
            if keep[0] != 0:
                corner_removed = True
            excluded[name] = [
                str(stat["formats"][index])
                for index in range(len(stat["formats"]))
                if index not in keep
            ]
            for field in ("formats", "scores", "raw_scores", "costs"):
                stat[field] = [stat[field][index] for index in keep]
        if excluded:
            warn_rank_0(
                "aumann_shapley: excluding candidates with non-finite damage measurements "
                f"from the search space: {excluded}"
            )
        return excluded, forced, corner_removed

    def initialize_candidate_stats(self):
        """Initialize candidate stats, then convert raw attributions through the damage link.

        The base implementation performs the distributed score reduction; the nonlinear
        coverage inversion runs after it, on rank-identical values, and overwrites the
        per-choice scores with log-headroom ``b`` so that minimizing their sum under the
        standard solve is the coverage-optimal allocation.
        """
        super().initialize_candidate_stats()

        no_quant = QuantRecipe(quant_cfg=None)
        # Scoring-time semantics must be captured before pruning rewrites the ladders: the
        # reference baseline, and which groups were configurable when scores were measured.
        damage_reference = self._damage_reference(no_quant)
        eligible = [
            name
            for name, stat in self.candidate_stats.items()
            if not stat.get("is_fixed", False)
            and len(stat["formats"]) > 1
            and any(r != no_quant for r in stat["formats"])
        ]
        excluded, forced_candidates, corner_removed = self._exclude_non_finite_candidates(no_quant)
        tokens = self._reduce_token_count(int(getattr(self, "_score_tokens", 0)))
        if tokens <= 0:
            warn_rank_0("aumann_shapley: no scored tokens; leaving raw scores in place.")
            return
        corner_kl_sum = getattr(self, "_corner_kl_sum", None)
        corner_kl_sum = 0.0 if corner_kl_sum is None else float(corner_kl_sum.item())
        f_corner = self._reduce_loss_scalar(corner_kl_sum) / tokens

        # A group that became a singleton through pruning stays in the fit (its remaining
        # candidate is still a normalized decision the quote must account for); groups whose
        # only remaining measurement is non-finite are forced and cannot be fitted.
        names = [
            name
            for name in eligible
            if name not in forced_candidates
            and any(r != no_quant for r in self.candidate_stats[name]["formats"])
        ]
        # Equal formats can carry different auto-generated display names, so all tables are
        # keyed by the canonical config signature; one representative provides the label.
        recipe_by_key: dict[str, QuantRecipe] = {}
        for name in names:
            for recipe in self.candidate_stats[name]["formats"]:
                if recipe != no_quant:
                    recipe_by_key.setdefault(recipe.checkpoint_signature, recipe)
        keys = sorted(recipe_by_key)
        labels = {key: str(recipe_by_key[key]) for key in keys}

        signed_by_key = {key: np.zeros(len(names)) for key in keys}
        ladders = set()
        for i, name in enumerate(names):
            stat = self.candidate_stats[name]
            ladder = []
            for recipe, raw_score in zip(stat["formats"], stat["raw_scores"], strict=True):
                if recipe == no_quant:
                    continue
                key = recipe.checkpoint_signature
                # Raw (unclamped) values: the base's monotonicity clamp must not leak into
                # the fit or the diagnostics.
                signed_by_key[key][i] = raw_score / tokens
                ladder.append(key)
            ladders.add(tuple(ladder))
        heterogeneous = len(ladders) > 1
        if heterogeneous:
            warn_rank_0(
                "aumann_shapley: groups have differing candidate ladders; the joint coverage "
                "interpretation is approximate for the formats not shared by all groups."
            )
        as_by_key = {key: np.maximum(vector, 0.0) for key, vector in signed_by_key.items()}
        # Candidate-level non-finite measurements were excluded from the ladders above; a
        # non-finite corner cannot be localized to one candidate, so it invalidates the fit.
        finite = math.isfinite(f_corner)
        signed_total = sum(float(np.abs(v).sum()) for v in signed_by_key.values())
        negative_mass = sum(
            float(np.abs(np.minimum(v, 0.0)).sum()) for v in signed_by_key.values()
        ) / max(signed_total, 1e-12)

        corner_mask_by_key = {key: np.zeros(len(names), dtype=bool) for key in keys}
        for i, name in enumerate(names):
            corner = self.candidate_stats[name]["formats"][0]
            corner_mask_by_key[corner.checkpoint_signature][i] = True
        # Mathematical completeness concerns the SIGNED attribution sum; the clamp applied
        # for the solver is reported separately as negative_attribution_mass.
        corner_mass = sum(
            float(signed_by_key[key][mask].sum()) for key, mask in corner_mask_by_key.items()
        )

        link = self.config["damage_link"]
        valid = True
        flags: list[str] = []
        if not finite:
            flags.append("non_finite_measurements")
            valid = False
        if excluded:
            flags.append("non_finite_scores_excluded")
        if corner_removed:
            # f_corner was measured with the original most-aggressive formats active; the
            # anchor no longer describes the corner of the pruned candidate space.
            flags.append("corner_format_excluded")
            valid = False
        if forced_candidates:
            # Some group had neither a finite candidate nor a no-quant fallback: its damage
            # is real but unquantifiable, so no quote derived from these scores is reliable.
            flags.append("non_finite_candidate_forced")
            valid = False
        if heterogeneous:
            flags.append("heterogeneous_ladders")
        if negative_mass > 1e-3:
            flags.append("negative_attribution_mass")
        damage_model: dict = {
            "link": link,
            "f_corner": f_corner,
            "n_score_tokens": tokens,
            "damage_reference": damage_reference,
            "negative_attribution_mass": negative_mass,
            "as_scores": {
                labels[key]: dict(zip(names, vector.tolist()))
                for key, vector in signed_by_key.items()
            },
        }
        if excluded:
            damage_model["excluded_candidates"] = excluded
        if forced_candidates:
            damage_model["forced_candidates"] = forced_candidates

        zero_tolerance = 1e-12
        positive_mass = sum(float(vector.sum()) for vector in as_by_key.values())
        coverage = link == "coverage" and bool(names) and bool(keys)
        if coverage:
            if not finite or corner_removed:
                # Without a usable corner (non-finite, or measured with a since-pruned
                # format) the inversion would anchor to an unrelated measurement; the
                # normalized attributions themselves are the honest solver objective.
                c, kappa, inflation, converged = 0.0, 1.0, 1.0, False
                score_by_key = as_by_key
                b_by_key = {key: np.zeros(len(names)) for key in keys}
            elif f_corner <= zero_tolerance and positive_mass <= zero_tolerance:
                # Quantization is measurably free here: an exact zero-damage model.
                flags.append("zero_damage")
                c, kappa, inflation, converged = 0.0, 1.0, 1.0, True
                score_by_key = {key: np.zeros(len(names)) for key in keys}
                b_by_key = score_by_key
            elif f_corner <= zero_tolerance:
                # Attributions claim damage the corner measurement does not show: the
                # coverage fit is not meaningful.
                flags.append("zero_corner_with_attribution_mass")
                valid = False
                c, kappa, inflation, converged = 0.0, 1.0, 1.0, False
                score_by_key = as_by_key
                b_by_key = {key: np.zeros(len(names)) for key in keys}
            else:
                c, b_by_key, kappa, inflation, converged = _anchor_ceiling(
                    as_by_key, f_corner, corner_mask_by_key
                )
                valid = valid and bool(converged) and math.isfinite(c)
                score_by_key = b_by_key
            if not valid:
                warn_rank_0(
                    "aumann_shapley: the coverage damage fit is not valid "
                    f"(flags={flags or ['inversion_not_converged']}); damage quotes are "
                    "unreliable and damage-bound searches will report is_satisfied=False."
                )
        else:
            score_by_key = as_by_key

        # Project solver scores onto the monotone ladder (more aggressive => at least as
        # much damage) by raising the more aggressive entries: conservative, and never
        # erases a less-aggressive format's real damage.
        projected_by_key = {key: np.zeros(len(names)) for key in keys}
        for i, name in enumerate(names):
            stat = self.candidate_stats[name]
            scores = [
                0.0 if recipe == no_quant else float(score_by_key[recipe.checkpoint_signature][i])
                for recipe in stat["formats"]
            ]
            for k in range(len(scores) - 2, -1, -1):
                scores[k] = max(scores[k], scores[k + 1])
            stat["scores"] = scores
            for recipe, score in zip(stat["formats"], scores, strict=True):
                if recipe != no_quant:
                    projected_by_key[recipe.checkpoint_signature][i] = score

        unprojected_total = sum(float(vector.sum()) for vector in score_by_key.values())
        projected_total = sum(float(vector.sum()) for vector in projected_by_key.values())
        adjustment = (projected_total - unprojected_total) / max(unprojected_total, 1e-12)
        if adjustment > 1e-9:
            flags.append("monotonicity_projection")
            damage_model["monotonicity_adjustment"] = adjustment

        if coverage:
            projected_corner_b = sum(
                float(projected_by_key[key][mask].sum()) for key, mask in corner_mask_by_key.items()
            )
            damage_model.update(
                {
                    "c": c,
                    "kappa": kappa,
                    "ceiling_inflation": inflation,
                    "inversion_converged": bool(converged),
                    # The quote-operative link values (what the solver and predicted_damage
                    # use); the unprojected inversion output stays exactly corner-anchored.
                    "b": {
                        labels[key]: dict(zip(names, b.tolist()))
                        for key, b in projected_by_key.items()
                    },
                    "b_unprojected": {
                        labels[key]: dict(zip(names, b.tolist())) for key, b in b_by_key.items()
                    },
                    "projected_corner_damage": _predict_damage(c, projected_corner_b),
                }
            )

        damage_model["valid"] = valid
        damage_model["approximation_flags"] = flags
        damage_model["completeness"] = corner_mass / max(f_corner, 1e-12)
        self.damage_model = damage_model

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Dispatch to the LP (default), the exact DP, or the SLA search."""
        max_predicted_damage = self.config.get("max_predicted_damage")
        if max_predicted_damage is not None:
            recipes, is_satisfied = self._run_damage_bound_search(
                float(max_predicted_damage), verbose
            )
        elif self.config.get("solver", "lp") == "dp":
            recipes, is_satisfied = self._run_dp_budget_search(max_weight_size, verbose)
        else:
            recipes, is_satisfied = super().run_search_with_stats(max_weight_size, verbose)
        flags = (getattr(self, "damage_model", None) or {}).get("approximation_flags", [])
        if is_satisfied and "non_finite_candidate_forced" in flags:
            warn_rank_0(
                "AutoQuantize FAILED to find a valid solution! The selection includes a "
                "forced candidate whose damage measurement was non-finite. "
            )
            is_satisfied = False
        return recipes, is_satisfied

    def run_search(self):
        """Run the inherited search and attach the predicted-damage quote."""
        super().run_search()
        self._attach_predicted_damage()
        # The base flow only saves before solving; re-save so the checkpoint file carries the
        # chosen recipe and can be re-solved offline.
        self.save_search_checkpoint(verbose=self.config.get("verbose", False))

    def _attach_predicted_damage(self) -> None:
        damage_model = getattr(self, "damage_model", None)
        if not damage_model or not self.best.get("recipe"):
            return
        no_quant = QuantRecipe(quant_cfg=None)
        total_score = 0.0
        for name, recipe in self.best["recipe"].items():
            stat = self.candidate_stats[name]
            if stat.get("is_fixed", False) or recipe == no_quant:
                continue
            total_score += stat["scores"][stat["formats"].index(recipe)]
        if damage_model["link"] == "coverage" and "c" in damage_model:
            predicted = _predict_damage(damage_model["c"], total_score)
        else:
            predicted = float(total_score)
        self.best["predicted_damage"] = predicted
        self.best["predicted_damage_valid"] = bool(damage_model.get("valid", True))
        if self.config.get("verbose"):
            print_rank_0(
                f"AutoQuantize(aumann_shapley) predicted damage: {predicted:.4e} "
                "(mean per-token KL, calibration units)"
            )

    def _stat_tables(self):
        stats = self.candidate_stats
        names = list(stats)
        n = len(names)
        num_choices = max(len(stats[name]["formats"]) for name in names)
        scores = np.full((n, num_choices), np.inf)
        costs = np.full((n, num_choices), np.inf)
        lengths = np.zeros(n, dtype=int)
        uncompressed = np.zeros(n)
        for i, name in enumerate(names):
            stat = stats[name]
            k = len(stat["formats"])
            lengths[i] = k
            scores[i, :k] = stat["scores"]
            costs[i, :k] = stat["costs"]
            uncompressed[i] = stat.get("uncompressed_cost", max(stat["costs"]))
        return names, scores, costs, lengths, uncompressed

    def _best_recipes_from_choice(self, names, choice_idx):
        best_recipes = {}
        for name, k in zip(names, choice_idx, strict=True):
            stat = self.candidate_stats[name]
            best_recipes[name] = {
                "format": stat["formats"][int(k)],
                "costs": stat["costs"][int(k)],
                "scores": stat["scores"][int(k)],
            }
        return best_recipes

    @staticmethod
    def _min_score_choices(scores, lengths):
        # Ties resolve toward the least aggressive choice (scores are non-increasing along
        # ascending compression, so scan from the end).
        return np.array(
            [
                int(lengths[i]) - 1 - int(np.argmin(scores[i, : lengths[i]][::-1]))
                for i in range(len(lengths))
            ]
        )

    @staticmethod
    def _minimize_within_budget(objective, constraint, budget):
        """Per-row choices minimizing ``sum(objective)`` s.t. ``sum(constraint) <= budget``.

        Each row's true minimum constraint is subtracted before discretizing, so only the
        increments above the mandatory baseline consume grid resolution (rows never overflow
        the grid regardless of their count) and ceil rounding keeps the returned selection
        feasible on the true budget. Returns None when even the per-row minima exceed it.
        """
        n = len(objective)
        row_min = np.where(np.isfinite(constraint), constraint, np.inf).min(axis=1)
        remaining = float(budget - row_min.sum())
        if remaining < -1e-9 * max(abs(budget), 1.0):
            return None
        if remaining <= 0:
            # No slack above the mandatory baseline: take each row's minimum-constraint
            # column, breaking ties toward the lower objective.
            at_minimum = constraint <= row_min[:, None]
            return np.where(at_minimum, objective, np.inf).argmin(axis=1)

        quantum = remaining / _DP_GRID
        increments = np.where(
            np.isfinite(constraint),
            np.ceil((constraint - row_min[:, None]) / quantum),
            _DP_GRID + 1,
        ).astype(np.int64)
        # Reorder each row so its zero-increment column is first (DP precondition).
        order = np.argsort(increments, axis=1, kind="stable")
        rows = np.arange(n)[:, None]
        values = np.where(np.isfinite(objective[rows, order]), -objective[rows, order], -np.inf)
        selection, _ = _mckp_max_value(values, increments[rows, order], _DP_GRID)
        return order[np.arange(n), selection]

    def _run_dp_budget_search(self, max_weight_size, verbose=False):
        """Deterministic grid-approximate knapsack solve for budget mode (LP alternative)."""
        names, scores, costs, lengths, _uncompressed = self._stat_tables()
        n = len(names)

        choice_idx = self._minimize_within_budget(scores, costs, float(max_weight_size))
        if choice_idx is None:
            warn_rank_0(
                "AutoQuantize FAILED to find a solution! The searched model might not meet "
                "all constraints. "
            )
            # Best effort: each row's minimum true cost, ties toward the lower score.
            at_minimum = costs <= np.where(np.isfinite(costs), costs, np.inf).min(
                axis=1, keepdims=True
            )
            choice_idx = np.where(at_minimum, scores, np.inf).argmin(axis=1)
        realized = float(costs[np.arange(n), choice_idx].sum())
        is_satisfied = realized <= max_weight_size * (1 + 1e-12) + 1e-9
        if verbose:
            print_rank_0(
                f"AutoQuantize(dp): realized weight size {realized:.2f} "
                f"(target {max_weight_size:.2f}), satisfied={is_satisfied}"
            )
        return self._best_recipes_from_choice(names, choice_idx), is_satisfied

    def _run_damage_bound_search(self, max_predicted_damage, verbose=False):
        """Minimize total weight cost subject to predicted damage <= ``max_predicted_damage``.

        Under the coverage link the bound maps to a score budget ``-log(1 - eps/c)``; score
        costs are rounded UP on the constraint axis so the quote is certified.
        """
        damage_model = getattr(self, "damage_model", None) or {}
        link = damage_model.get("link", self.config.get("damage_link", "additive"))
        names, scores, costs, lengths, _uncompressed = self._stat_tables()
        n = len(names)

        if link == "coverage" and not damage_model.get("valid", False):
            warn_rank_0(
                "AutoQuantize FAILED to find a solution! The coverage damage fit is invalid "
                f"(flags={damage_model.get('approximation_flags')}), so the damage bound "
                "cannot be certified. Returning the minimum-damage configuration. "
            )
            choice_idx = self._min_score_choices(scores, lengths)
            return self._best_recipes_from_choice(names, choice_idx), False

        if link == "coverage" and "c" in damage_model:
            c = float(damage_model["c"])
            budget = (
                np.inf if max_predicted_damage >= c else -math.log(1.0 - max_predicted_damage / c)
            )
        else:
            budget = float(max_predicted_damage)

        if not np.isfinite(budget):
            choice_idx = np.zeros(n, dtype=int)
        else:
            choice_idx = self._minimize_within_budget(costs, scores, budget) if budget > 0 else None
            if choice_idx is None:
                warn_rank_0(
                    "AutoQuantize FAILED to find a solution! Even the least aggressive "
                    "choices exceed the damage budget. "
                )
                choice_idx = self._min_score_choices(scores, lengths)

        total_score = float(scores[np.arange(n), choice_idx].sum())
        is_satisfied = (not np.isfinite(budget)) or bool(total_score <= budget + 1e-12)
        if verbose:
            realized = float(costs[np.arange(n), choice_idx].sum())
            print_rank_0(
                f"AutoQuantize(sla): score total {total_score:.4e} (budget {budget:.4e}), "
                f"weight size {realized:.2f}, satisfied={is_satisfied}"
            )
        return self._best_recipes_from_choice(names, choice_idx), is_satisfied


AUTO_QUANTIZE_SEARCHERS[AutoQuantizeAumannShapleySearcher.method_name] = (
    AutoQuantizeAumannShapleySearcher
)
