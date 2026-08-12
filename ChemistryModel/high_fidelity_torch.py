"""Characterisation-only higher-fidelity reactive physics.

This module deliberately does not replace the discovery engine. It subclasses
BatchedReactiveSimulation so controlled molecule experiments can A/B test a
more expensive local electronic approximation while ordinary soup runs keep
using the existing potential unchanged.

Revision 3 keeps the competitive valence-state model from revision 2, but
delays that replacement until the second contact has meaningful bond character.
V1 proved that the H-transfer barrier was controlled by the hydrogen
coordination term, but it also allowed the transferring hydrogen to keep two
full Morse bonds at once, producing an unphysical C-H-H three-centre product.

V3 treats a valence-one hydrogen with a strong heavy-atom donor bond and a
second approaching partner as two competing bonding states once the second
contact is chemically significant:

    donor-H   + nonbonded partner
    donor     + H-partner

The lower mixed state is used in place of the base model's two simultaneous
full bonds plus hydrogen over-coordination penalty. Before that region, the
base Morse tail is left untouched so it can attract the approaching partner
instead of V2 prematurely replacing that weak entrance interaction. This is an empirical
valence-state-mixing approximation: it does not name molecules, products or
reactions, and it is exactly zero when hydrogen has only one partner. Core
Morse repulsion remains present in both states, so the correction cannot make
atoms collapse through each other.

The state coupling is scaled from the geometric mean of the two measured bond
depths and only exists while both contacts overlap. A 0.45 dimensionless
mixing fraction is an experimental first value, not a fitted universal
constant. It must be validated across independent H-transfer reactions before
promotion beyond characterisation.
"""

import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation


HF_MODEL_NAME = "reactive_v1+h_transfer_competition_v6"
HF_MODEL_REVISION = 6

# A transfer correction needs both a heavy-atom donor contact and a second
# partner contact. There is deliberately no hard taper threshold: the energy
# correction itself tends continuously to zero as either contact fades. H2,
# H + H2, and ordinary one-bond hydrogens therefore stay exactly on the base
# potential without introducing a force discontinuity at an arbitrary cutoff.
H_TRANSFER_DONOR_TAPER_MIN = 0.0
H_TRANSFER_COMPETITOR_TAPER_MIN = 0.0

# Dimensionless coupling between the two alternative valence states. The
# dimensional coupling is this fraction times sqrt(D_donor * D_partner), then
# smoothly gated by simultaneous contact and by how balanced the two contacts
# are. 0.45 is intentionally a first experimental value rather than a tune-to-
# outcome knob.
H_TRANSFER_STATE_MIXING_FRACTION = 0.63

# V2 switched to the valence-state surface as soon as the second contact had
# any non-zero cutoff taper. That was mathematically smooth but physically too
# early: it removed the ordinary attractive Morse tail while the incoming atom
# was still only a weak encounter, creating a new entrance barrier around
# H-H = 1.14 A. V3 centres a smooth engagement window on the same 0.35 taper
# used everywhere else to call a contact bonded. Below 0.20 the ordinary
# potential is exact; above 0.50 the competitive one-valence surface is exact.
# Cubic smoothstep has zero slope at both ends, so the force stays continuous.
H_TRANSFER_GATE_START = 0.20
H_TRANSFER_GATE_FULL = 0.50

# In each valence state one bond is occupied and the other is not. V4 gave the
# unoccupied partner only the repulsive half of its Morse curve, which is
# positive but far too weak: with both contacts strong the mixed state can
# still sink into a bound three-centre well, which is exactly what V1 was
# built to prevent and what a 10 ps trapped trajectory showed it does not.
#
# LEPS handles this with a separate anti-Morse curve for the unoccupied bond,
# genuinely repulsive where the occupied one is attractive. This constant
# blends between the two:
#
#     0.0  exactly V4 (unoccupied bond keeps bare Morse repulsion)
#     1.0  full anti-Morse
#
# It is deliberately shared by every element pair for now. Real LEPS uses a
# per-pair Sato parameter, and this probably needs to become a table: one
# value has to serve C-H and H-H at once, and their depths differ by enough
# that a single number may not suit both. Measure before assuming.
H_TRANSFER_SATO = 0.0

# The state coupling is scaled by a geometric overlap that peaks when both
# contacts are strong -- which is exactly the shared-hydrogen configuration
# the correction exists to forbid. Measured at mixing 0.63, the avoided-
# crossing stabilisation is 2.13 eV at the three-centre trap and 0.60 eV at
# the transition state: inverted, and enough to turn what should be a saddle
# into a bound minimum.
#
# In LEPS the coupling varies slowly across the surface. A roughly constant
# coupling at a state crossing rounds the corner and leaves a barrier; a
# coupling that peaks at the crossing digs a well there instead.
#
# This constant blends the peaked overlap toward a flat engagement function
# that saturates once the weaker contact is bonded, rather than continuing to
# grow as both bonds strengthen:
#
#     0.0  exactly V4 (peaked overlap)
#     1.0  flat once the weaker contact passes the usual 0.35 bond threshold
#
# Both forms still vanish when either contact fades, so isolated reactants
# and products recover the base potential exactly either way.
H_TRANSFER_COUPLING_FLATTEN = 0.0

# Ceiling, in eV, on how far the avoided crossing may pull the mixed state
# below the lower of the two valence states.
#
# The three-centre trap and the collision barrier both sit where every cutoff
# taper is saturated, so no knob keyed on geometry can tell them apart: sato
# and the coupling flatten both move them together, which is what the knob
# map showed. What does separate them is the crossing stabilisation itself,
# measured at mixing 0.63:
#
#     three-centre trap      2.13 eV
#     collision barrier top  1.70 eV
#     transition state       0.60 eV
#
# A ceiling therefore bites hardest exactly where the well is deepest, leaves
# the transition state untouched while it stays above 0.6, and keys on the
# diabatic gap rather than on any distance.
#
# None disables the ceiling and reproduces V4 exactly. The softening below
# keeps the value and its slope continuous, since a hard clamp would put a
# force discontinuity along the whole contour where the cap begins to bind.
# Width of the smoothing on the two-contact minimum in the gate, as a squared
# taper value. A bare min(a, b) is continuous but not differentiable where the
# two contacts are equal, which is exactly where the donor and competitor
# labels swap, so it trades an energy step for a force flip. The softened form
# below is differentiable everywhere and agrees with min to within the
# smoothing width. sqrt(1e-4) = 0.01 taper units.
H_TRANSFER_SMOOTH_MIN_EPSILON_SQUARED = 1e-4

H_TRANSFER_LOWERING_CAP = None
H_TRANSFER_LOWERING_CAP_SOFTNESS = 0.25


class HighFidelityBatchedReactiveSimulation(BatchedReactiveSimulation):
    """Batched reactive simulation with competitive valence-one H bonding."""

    physics_model_name = HF_MODEL_NAME
    physics_model_revision = HF_MODEL_REVISION

    def __init__(self, *args, **kwargs):

        # Owned per instance rather than read from module scope on every call.
        # The diagnostic tools rebind these globals to sweep a parameter, and
        # reading them lazily meant a simulation built earlier in the same
        # process silently picked up a later experiment's settings. Copying
        # here means every constructed model permanently owns what it was
        # built with.
        self.h_transfer_state_mixing_fraction = float(
            H_TRANSFER_STATE_MIXING_FRACTION
        )
        self.h_transfer_gate_start = float(H_TRANSFER_GATE_START)
        self.h_transfer_gate_full = float(H_TRANSFER_GATE_FULL)
        self.h_transfer_sato = float(H_TRANSFER_SATO)
        self.h_transfer_coupling_flatten = float(H_TRANSFER_COUPLING_FLATTEN)
        self.h_transfer_lowering_cap = (
            None if H_TRANSFER_LOWERING_CAP is None
            else float(H_TRANSFER_LOWERING_CAP)
        )
        self.h_transfer_lowering_cap_softness = float(
            H_TRANSFER_LOWERING_CAP_SOFTNESS
        )
        self.h_transfer_smooth_min_epsilon_squared = float(
            H_TRANSFER_SMOOTH_MIN_EPSILON_SQUARED
        )

        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        base = super().energy_per_atom(positions)
        correction = self._hydrogen_transfer_competition(positions)
        return base + correction

    def _hydrogen_transfer_competition(self, positions):
        """Return one local correction per atom.

        The base model gives every contact a complete Morse single-bond term,
        then uses a quadratic over-coordination energy to discourage a
        valence-one H from keeping two partners. That creates a barrier, but it
        cannot actually *transfer* the one bond from donor to acceptor.

        For an H that is already strongly attached to a heavy atom and has a
        second contact, V2 replaces the local double-bonded picture with a
        smooth two-state energy. The rest of the potential -- carbonyl bond
        order, angles, cutoffs, thermostat, integration, etc. -- is untouched.
        """

        neighbours = self.neighbours
        mask = self.neighbour_mask.to(self.dtype)

        offsets = positions[neighbours] - positions[:, None, :]
        offsets = offsets - self.box_size * torch.round(
            offsets / self.box_size
        )

        distances = torch.sqrt(
            torch.clamp(torch.sum(offsets ** 2, dim=2), min=1e-12)
        )

        centre_types = self.types[:, None].expand_as(neighbours)
        other_types = self.types[neighbours]

        pair_inner = self.cutoff_inner[centre_types, other_types]
        pair_outer = self.cutoff_outer[centre_types, other_types]
        span = torch.clamp(pair_outer - pair_inner, min=1e-9)

        fraction = torch.clamp(
            (distances - pair_inner) / span, 0.0, 1.0
        )
        taper = (
            0.5 * (1.0 + torch.cos(torch.pi * fraction)) * mask
        )

        coordination = torch.sum(taper, dim=1)
        valence = self.valence[self.types]

        # Reproduce the base bond-order calculation so the state energies use
        # exactly the same distance/depth/width parameters as reactive_v1.
        spare = torch.clamp(valence - coordination, min=0.0)
        spare_other = spare[neighbours]

        weighted = taper * spare_other
        totals = torch.sum(weighted, dim=1)
        totals_other = totals[neighbours]

        share_out = torch.where(
            totals[:, None] > 1e-9,
            spare[:, None] * weighted
            / torch.clamp(totals[:, None], min=1e-9),
            torch.zeros_like(weighted),
        )

        share_back = torch.where(
            totals_other > 1e-9,
            spare_other * (taper * spare[:, None])
            / torch.clamp(totals_other, min=1e-9),
            torch.zeros_like(weighted),
        )

        order = torch.clamp(
            1.0 + torch.minimum(share_out, share_back), 0.0, 3.0
        ) * mask

        lower = torch.clamp(order - 1.0, 0.0, 1.0)
        upper = torch.clamp(order - 2.0, 0.0, 1.0)

        def blend(single_table, double_table, triple_table):
            single = single_table[centre_types, other_types]
            double = double_table[centre_types, other_types]
            triple = triple_table[centre_types, other_types]
            first = single + (double - single) * lower
            return first + (triple - first) * upper

        pair_length = blend(
            self.bond_length, self.double_length, self.triple_length
        )
        pair_depth = blend(
            self.bond_depth, self.double_depth, self.triple_depth
        )
        pair_width = blend(
            self.bond_width, self.double_width, self.triple_width
        )

        shift = distances - pair_length
        repulsive = pair_depth * torch.exp(-2.0 * pair_width * shift)
        attractive = 2.0 * pair_depth * torch.exp(-pair_width * shift)

        # These are full undirected-pair energy values. The base engine stores
        # half of each directed copy on each atom, so the summed base energy
        # contains each value once. A correction placed on the transferring H
        # can therefore replace the two complete pair energies directly.
        pair_morse = taper * (repulsive - attractive)
        pair_core = taper * repulsive

        # Anti-Morse: the triplet-like curve for a bond that is not occupied
        # in this state. It reuses the same depth, length and width entries as
        # the bonding curve, so no new tables are introduced; only the sign
        # structure differs. It is repulsive at every separation and decays to
        # zero at long range.
        pair_anti = 0.5 * pair_depth * taper * (
            torch.exp(-2.0 * pair_width * shift)
            + 2.0 * torch.exp(-pair_width * shift)
        )

        # At sato 0 this is exactly pair_core, so the default path reproduces
        # V4 bit for bit and this whole addition is inert.
        pair_unoccupied = (
            (1.0 - float(self.h_transfer_sato)) * pair_core
            + float(self.h_transfer_sato) * pair_anti
        )

        hydrogen_index = int(R.ELEMENT_INDEX["H"])
        is_hydrogen = self.types == hydrogen_index
        heavy = other_types != hydrogen_index

        # Pick the strongest currently bonded heavy-atom donor for each H.
        # `argmax` only selects which local state is active; gradients still
        # flow through all gathered energies/distances for that selected state.
        donor_score = taper * pair_depth * heavy.to(self.dtype)
        donor_score = donor_score * mask
        donor_slot = torch.argmax(donor_score, dim=1)
        row = torch.arange(
            neighbours.shape[0], device=self.device, dtype=torch.long
        )

        donor_taper = taper[row, donor_slot]
        donor_depth = pair_depth[row, donor_slot]
        donor_morse = pair_morse[row, donor_slot]
        donor_core = pair_core[row, donor_slot]
        donor_unoccupied = pair_unoccupied[row, donor_slot]

        donor_strength = donor_score[row, donor_slot]
        donor_valid = (
            is_hydrogen
            & (donor_strength > float(H_TRANSFER_DONOR_TAPER_MIN))
            & heavy[row, donor_slot]
        )

        # Strongest second contact, excluding the selected donor slot. The
        # partner may itself be H, C, N or O; no reaction identity is encoded.
        slot_numbers = torch.arange(
            neighbours.shape[1], device=self.device, dtype=torch.long
        )[None, :]
        not_donor = slot_numbers != donor_slot[:, None]

        competitor_score = taper * pair_depth * mask * not_donor.to(self.dtype)
        competitor_slot = torch.argmax(competitor_score, dim=1)

        competitor_strength = competitor_score[row, competitor_slot]
        competitor_taper = taper[row, competitor_slot]
        competitor_depth = pair_depth[row, competitor_slot]
        competitor_morse = pair_morse[row, competitor_slot]
        competitor_core = pair_core[row, competitor_slot]
        competitor_unoccupied = pair_unoccupied[row, competitor_slot]

        # If every non-donor slot is padded/absent, argmax still returns slot
        # zero. Gate on the *excluded score* rather than the gathered taper so
        # that a single ordinary bond remains exactly the base potential.
        competitor_valid = (
            competitor_strength > float(H_TRANSFER_COMPETITOR_TAPER_MIN)
        )
        active = donor_valid & competitor_valid

        # State D: donor-H is the occupied bond; the competing partner keeps
        # only its short-range core repulsion.
        donor_state = donor_morse + competitor_unoccupied

        # State P: the new H-partner bond is occupied; the donor keeps only its
        # core repulsion. This prevents V1's two-full-bonds C-H-H minimum.
        partner_state = donor_unoccupied + competitor_morse

        # Electronic/valence mixing exists only while both contacts coexist.
        # The balance term is one for equal contacts and tends smoothly to zero
        # when either side dominates, so isolated reactants/products recover
        # reactive_v1 exactly.
        contact_sum = donor_taper + competitor_taper
        balance = (
            4.0 * donor_taper * competitor_taper
            / torch.clamp(contact_sum * contact_sum, min=1e-12)
        )
        peaked_overlap = torch.sqrt(
            torch.clamp(donor_taper * competitor_taper, min=1e-12)
        ) * balance

        # Flat alternative: engaged or not, keyed on the weaker of the two
        # contacts, so the coupling stops growing once both are bonded. Same
        # 0.35 threshold and same cubic smoothstep used elsewhere, so it is
        # continuous in value and slope and still reaches zero whenever
        # either contact does.
        weaker = torch.minimum(donor_taper, competitor_taper)
        flat_fraction = torch.clamp(weaker / 0.35, 0.0, 1.0)
        flat_overlap = (
            flat_fraction * flat_fraction * (3.0 - 2.0 * flat_fraction)
        )

        overlap = (
            (1.0 - float(self.h_transfer_coupling_flatten)) * peaked_overlap
            + float(self.h_transfer_coupling_flatten) * flat_overlap
        )

        coupling = (
            float(self.h_transfer_state_mixing_fraction)
            * torch.sqrt(torch.clamp(donor_depth * competitor_depth, min=0.0))
            * overlap
        )

        half_difference = 0.5 * (donor_state - partner_state)

        # How far below the lower valence state the avoided crossing pulls.
        lowering = torch.sqrt(
            half_difference * half_difference + coupling * coupling + 1e-12
        ) - torch.abs(half_difference)

        if self.h_transfer_lowering_cap is not None:
            # Softplus rather than a hard clamp: the cap binds over a finite
            # window instead of switching on along a contour, so the force
            # stays continuous where it starts to take effect. Softness is in
            # the same units as the cap.
            ceiling = float(self.h_transfer_lowering_cap)
            softness = max(float(self.h_transfer_lowering_cap_softness), 1e-6)
            lowering = ceiling - softness * torch.nn.functional.softplus(
                (ceiling - lowering) / softness
            )

        mixed_state = (
            0.5 * (donor_state + partner_state)
            - torch.abs(half_difference)
            - lowering
        )

        # Remove exactly the local picture already present in the base energy:
        # two complete Morse bonds plus this H atom's over-coordination term.
        # The coefficient itself is not retuned; the invalid two-bond state is
        # replaced by the one-valence mixed state instead.
        # Only the share of the over-coordination penalty that the two
        # selected contacts are responsible for. The penalty is computed from
        # the total coordination over every neighbour, but the correction
        # replaces exactly two of them, so subtracting all of it handed any
        # third contact a free unpenalised bond and opened a bound
        # three-centre well. Removing the difference between the full penalty
        # and the penalty that would remain without these two contacts leaves
        # a third partner's restraint intact.
        excess = torch.clamp(coordination - valence, min=0.0)
        # What the hydrogen still carries once these two contacts are handed
        # to the mixed state: one valence's worth from that state, plus every
        # unselected contact. So the residual excess over valence is just the
        # unselected coordination, and subtracting valence again here would
        # wrongly cancel it. With only two contacts this is zero and the
        # expression reduces to the old one exactly.
        excess_rest = torch.clamp(
            coordination - donor_taper - competitor_taper, min=0.0
        )
        base_h_over = self.over_penalty * (excess ** 2 - excess_rest ** 2)

        local_base = donor_morse + competitor_morse + base_h_over
        delta = mixed_state - local_base

        # Keep the ordinary weak-contact Morse attraction intact until the
        # competitor has substantial bond character. Then engage the one-
        # valence surface smoothly around the same 0.35 taper used by the
        # bond detector. This fixes V2's premature entrance repulsion without
        # changing its treatment of the actual transfer region.
        # Keyed on the weaker of the two contacts, not on the competitor
        # alone. Which neighbour is called donor and which competitor comes
        # from an argmax, and those labels swap as the hydrogen crosses over.
        # Everything else here is symmetric under that swap; reading
        # competitor_taper by itself was not, so an unsaturated gate put a
        # step in the energy exactly at the crossing. Measured at 0.055 eV
        # over 0.0002 A for an O...H...C pair 2.95 A apart, which is ordinary
        # hydrogen bond geometry.
        #
        # When the donor is the stronger contact, as it usually is, the
        # minimum is the competitor and this is identical to the old
        # behaviour. It differs only where the two cross, which is the only
        # place the old form was wrong.
        gap = donor_taper - competitor_taper
        weaker_contact = 0.5 * (
            donor_taper + competitor_taper
            - torch.sqrt(
                gap * gap
                + float(self.h_transfer_smooth_min_epsilon_squared)
            )
        )
        gate_fraction = torch.clamp(
            (weaker_contact - float(self.h_transfer_gate_start))
            / float(self.h_transfer_gate_full - self.h_transfer_gate_start),
            0.0,
            1.0,
        )
        gate = gate_fraction * gate_fraction * (3.0 - 2.0 * gate_fraction)

        return torch.where(active, gate * delta, torch.zeros_like(delta))