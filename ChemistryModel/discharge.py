import numpy as np

import reactive as R


# ============================================================
# A discharge channel
# ============================================================
#
# Two things happen inside a spark, and only one of them is heat.
#
# The channel is hot, so atoms in it get thermal velocities drawn
# at the channel temperature. Lightning runs at 20,000 to 30,000
# kelvin, which is a measured quantity rather than an invented
# one.
#
# But heat alone barely breaks anything, and that is not a flaw
# in the code. A bond only breaks if energy arrives in the
# relative motion along it, and random thermal velocities put
# most of their energy into the pair moving off together instead.
# Measured on this model, a 30,000 K channel delivers about 3.9
# eV per atom but only 1.3 eV into the bond coordinate, so under
# a tenth of bonds break. Run it and you get fifty bonds formed
# and one broken.
#
# Real discharges do not depend on that. They dissociate by
# electron impact: a free electron dumps energy directly into a
# bond's antibonding orbital and the bond comes apart. That is
# the dominant channel in any real spark, and it is exactly what
# a model with no electrons cannot reproduce by heating.
#
# So it is put in explicitly. Bonds inside the channel are
# selected at random and given opposing kicks along their own
# axis, carrying more than their dissociation energy. Momentum is
# conserved, the energy goes where it would really go, and the
# fraction dissociated is the one honest free parameter.


def find_bonds(positions, types, box_size, threshold=0.35):
    count = len(positions)

    first, second = np.triu_indices(count, k=1)

    offsets = positions[second] - positions[first]
    offsets -= box_size * np.round(offsets / box_size)

    distances = np.linalg.norm(offsets, axis=1)

    inner = R.CUTOFF_INNER[types[first], types[second]]
    outer = R.CUTOFF_OUTER[types[first], types[second]]

    taper = R.smooth_cutoff(distances, inner, outer)

    keep = taper > threshold

    return first[keep], second[keep], offsets[keep], distances[keep]


def strike(positions, velocities, masses, types, box_size,
           generator, radius=2.2, temperature=25000.0,
           dissociation=0.35, excess=1.4):
    # Returns new velocities plus a short report. Works on plain
    # numpy so the viewer and the batch runner can share it.

    count = len(positions)

    origin = generator.uniform(0.0, box_size, size=3)

    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)

    offsets = positions - origin
    offsets -= box_size * np.round(offsets / box_size)

    along = offsets @ axis

    perpendicular = offsets - along[:, None] * axis

    distance = np.linalg.norm(perpendicular, axis=1)

    inside = distance < radius

    struck = int(np.count_nonzero(inside))

    if struck == 0:
        return velocities, {"struck": 0, "dissociated": 0,
                            "deposited": 0.0}

    # How completely each atom is caught by the channel.

    weight = np.zeros(count)

    weight[inside] = 0.5 * (
        1.0 + np.cos(np.pi * distance[inside] / radius)
    )

    # ---- the thermal part ----

    scale = np.sqrt(
        8.617333e-5 * temperature / (masses * 103.642)
    )

    hot = generator.normal(size=positions.shape) * scale[:, None]

    # Blended on the square root so energy, not speed,
    # interpolates between cold and hot.

    updated = (
        np.sqrt(1.0 - weight)[:, None] * velocities
        + np.sqrt(weight)[:, None] * hot
    )

    # ---- the electron-impact part ----

    first, second, bond_offsets, distances = find_bonds(
        positions, types, box_size
    )

    dissociated = 0

    if len(first) > 0 and dissociation > 0.0:
        # A bond is a candidate in proportion to how deep in the
        # channel it sits.

        pair_weight = np.minimum(weight[first], weight[second])

        chance = pair_weight * dissociation

        chosen = generator.random(len(first)) < chance

        depth = R.BOND_DEPTH[types[first], types[second]]

        for index in np.where(chosen)[0]:
            a = int(first[index])
            b = int(second[index])

            direction = bond_offsets[index] / max(
                distances[index], 1e-9
            )

            reduced = (
                masses[a] * masses[b] / (masses[a] + masses[b])
            )

            # Enough relative energy to clear the well, with a
            # margin so the fragments actually separate rather
            # than falling straight back together.

            energy = excess * depth[index]

            relative_speed = np.sqrt(
                2.0 * energy / (reduced * 103.642)
            )

            # Opposing kicks, split by mass, so the pair flies
            # apart with the centre of mass unchanged.

            total = masses[a] + masses[b]

            updated[a] -= direction * relative_speed * masses[b] / total
            updated[b] += direction * relative_speed * masses[a] / total

            dissociated += 1

    # Remove whatever net momentum the strike introduced, or the
    # whole box slowly drifts after repeated strikes.

    momentum = np.sum(masses[:, None] * updated, axis=0)

    updated = updated - momentum / np.sum(masses)

    deposited = float(
        0.5 * np.sum(
            masses[:, None] * (updated ** 2 - velocities ** 2)
        ) * 103.642
    )

    return updated, {
        "struck": struck,
        "bonds_in_channel": int(np.count_nonzero(
            np.minimum(weight[first], weight[second]) > 0.05
        )) if len(first) else 0,
        "dissociated": dissociated,
        "deposited": deposited,
    }


def apply_to(simulation, generator, radius=2.2,
             temperature=25000.0, dissociation=0.35):
    # Convenience wrapper for a torch simulation.

    import torch

    positions = simulation.positions_numpy
    velocities = simulation.velocities.detach().cpu().numpy()
    masses = simulation.masses.detach().cpu().numpy()

    updated, report = strike(
        positions,
        velocities,
        masses,
        simulation.types_numpy,
        simulation.box_size,
        generator,
        radius=radius,
        temperature=temperature,
        dissociation=dissociation,
    )

    simulation.velocities = torch.tensor(
        updated,
        device=simulation.device,
        dtype=simulation.dtype
    )

    return report
