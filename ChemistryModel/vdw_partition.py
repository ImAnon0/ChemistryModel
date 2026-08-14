"""Standalone vdW energy-partition experiments; never imported by production."""

from __future__ import annotations

import numpy as np

import vdw_reference as V


ARCHITECTURES = (
    "whole_suppressed",
    "all_pairs_shielded",
    "split_wca_suppressed_repulsion",
    "split_wca_dispersion_only",
    "shared_contact_linear_split",
)


def _long_range_product(energy, force, distance):
    switch = V.cutoff_switch(distance)
    derivative = V.cutoff_switch_derivative(distance)
    return switch * energy, switch * force - derivative * energy


def reaxff_wca_components(distance, first, second):
    """Split the shielded curve at its minimum in the WCA manner.

    This is an exact algebraic partition: ``repulsive + attractive == raw``.
    The attractive part is constant below the minimum, while the repulsive
    remainder carries the short-range force. It is a diagnostic division of
    responsibilities, not a ReaxFF prescription.
    """
    distance = np.asarray(distance, dtype=float)
    minimum, depth = V.pair_parameters(first, second)
    raw_energy = V.raw_reaxff_energy(distance, first, second)
    raw_force = V.raw_reaxff_force(distance, first, second)
    inside = distance < minimum
    repulsive_energy = np.where(inside, raw_energy + depth, 0.0)
    repulsive_force = np.where(inside, raw_force, 0.0)
    attractive_energy = np.where(inside, -depth, raw_energy)
    attractive_force = np.where(inside, 0.0, raw_force)
    return {
        "repulsive_energy": repulsive_energy,
        "repulsive_force": repulsive_force,
        "attractive_energy": attractive_energy,
        "attractive_force": attractive_force,
    }


def partition_energy_force(distance, first, second, architecture):
    """Return standalone vdW contribution and radial force for a partition."""
    distance = np.asarray(distance, dtype=float)
    if architecture == "whole_suppressed":
        values = V.suppressed_vdw_components(
            distance, first, second, model="reaxff"
        )
        return values["energy"], values["force"]

    raw_energy = V.raw_reaxff_energy(distance, first, second)
    raw_force = V.raw_reaxff_force(distance, first, second)
    if architecture == "all_pairs_shielded":
        return _long_range_product(raw_energy, raw_force, distance)

    components = reaxff_wca_components(distance, first, second)
    attractive_energy, attractive_force = _long_range_product(
        components["attractive_energy"],
        components["attractive_force"], distance,
    )
    if architecture == "split_wca_dispersion_only":
        return attractive_energy, attractive_force

    if architecture not in {
        "split_wca_suppressed_repulsion", "shared_contact_linear_split"
    }:
        raise ValueError(f"unknown vdW partition architecture: {architecture!r}")

    if architecture == "split_wca_suppressed_repulsion":
        weight = V.suppression_weight(distance, first, second)
        weight_derivative = V.suppression_weight_derivative(
            distance, first, second
        )
    else:
        # Deliberately test the simplest shared-contact responsibility. This is
        # not proposed as sufficient merely because it reuses chemical state.
        taper = V.reactive_contact_taper(distance, first, second)
        weight = 1.0 - taper
        weight_derivative = -V.reactive_contact_taper_derivative(
            distance, first, second
        )

    repulsive_energy = weight * components["repulsive_energy"]
    repulsive_force = (
        weight * components["repulsive_force"]
        - weight_derivative * components["repulsive_energy"]
    )
    repulsive_energy, repulsive_force = _long_range_product(
        repulsive_energy, repulsive_force, distance
    )
    return attractive_energy + repulsive_energy, attractive_force + repulsive_force
