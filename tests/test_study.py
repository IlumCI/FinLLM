"""The statistics behind the repo's claims.

`lamb/study.py` decides which differences this project is allowed to assert, so its
arithmetic is worth testing directly. None of these tests train anything.
"""

from __future__ import annotations

from lamb.study import (ALL_ARMS, ARMS, COMPUTE_MATCH, TASKS, _perm_p_paired,
                        _perm_p_unpaired, min_detectable, seeds_needed, summarise)


def _rows(arm, vals):
    return [{"arm": arm, "task": "d2g1", "seed": i, "acc": v} for i, v in enumerate(vals)]


def test_complete_separation_is_the_strongest_claim_five_seeds_allow():
    """Disjoint seed ranges at 5+5 give the exact permutation floor, 2/C(10,5)."""
    a, b = [0.75, 0.78, 0.79, 0.91, 0.90], [0.48, 0.53, 0.70, 0.49, 0.32]
    assert min(a) > max(b)
    assert abs(_perm_p_unpaired(a, b) - 2 / 252) < 1e-9


def test_sign_flip_test_has_a_floor_that_five_seeds_cannot_beat():
    """Whatever the effect size, 5 paired seeds cannot go below 2/2**5."""
    huge = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert _perm_p_paired(huge) == 2 / 2 ** 5 == 0.0625
    # ...and the floor halves per added seed, so 6 clears 0.05
    assert _perm_p_paired(huge + [15.0]) == 2 / 2 ** 6 < 0.05


def test_permutation_tests_do_not_reject_when_the_arms_are_interleaved():
    a, b = [0.50, 0.60, 0.70], [0.55, 0.65, 0.45]
    assert _perm_p_unpaired(a, b) > 0.3
    assert _perm_p_paired([x - y for x, y in zip(a, b)]) > 0.3


def test_power_statement_matches_the_observed_spread():
    """The number that reframes every single-run claim in this repo."""
    sd = 0.138                       # the measured Coconut seed-to-seed spread
    assert 0.17 < min_detectable(sd, 5) < 0.18      # n=5 resolves ~18 points, no better
    assert seeds_needed(sd, 0.05) > 50              # a 5-point claim needs ~61 seeds
    assert seeds_needed(sd, 0.30) < 10              # a 30-point one needs very few


def test_summarise_pairs_by_seed_and_flags_separation():
    rows = _rows("coconut", [0.48, 0.53, 0.70, 0.49, 0.32]) + \
           _rows("lotus-answer", [0.75, 0.78, 0.79, 0.91, 0.90])
    s = summarise(rows)
    assert s["arms"]["coconut"]["n"] == s["arms"]["lotus-answer"]["n"] == 5
    pair = s["pairs"]["lotus-answer - coconut"]
    assert pair["separated"] is True
    assert pair["mean"] > 0.3 and not pair["crosses_zero"]
    assert pair["p_perm_unpaired"] < 0.01


def test_summarise_uses_only_seeds_both_arms_ran():
    """Extending one arm to more seeds must not silently unpair the comparison."""
    rows = _rows("coconut", [0.40, 0.50]) + _rows("lotus-trace", [0.80, 0.90, 0.95])
    s = summarise(rows)
    assert s["arms"]["lotus-trace"]["n"] == 3
    assert s["pairs"]["lotus-trace - coconut"]["n"] == 2    # the shared seeds only


def test_task_specs_give_each_space_enough_latents_for_its_trace():
    """n_latent must cover the gold trace or the trace arm is handicapped."""
    from lamb import ArithmeticTokenizer
    from lamb.lotus import trace_targets
    from lamb.selfplay.grammar import Descriptor, TaskGrammar

    g, tok = TaskGrammar(), ArithmeticTokenizer()
    for key, spec in TASKS.items():
        d = Descriptor(spec.depth, spec.digits, spec.ops_key)
        worst = max(len(trace_targets(tok, g.sample_with_trace(d, s)[2])) for s in range(400))
        assert worst <= spec.n_latent, (key, worst, spec.n_latent)


def test_compute_match_factor_is_greater_than_one_and_declared():
    """The control exists because the arms are not wall-clock matched."""
    assert COMPUTE_MATCH > 1.0
    assert "coconut-long" in ARMS
    assert "lotus-space" in ALL_ARMS and "lotus-space" not in ARMS
