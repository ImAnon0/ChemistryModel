# Short-range van der Waals reference comparison

## Scope and sources

This is a standalone diagnostic. It does not alter ChemistryModel forces, MD,
Torch code, or production parameters.

- ReaxFF form: van Duin et al., *J. Phys. Chem. A* **105**, 9396 (2001),
  [doi:10.1021/jp004368u](https://doi.org/10.1021/jp004368u), as documented in
  the [ReaxFF manual](https://www.scm.com/doc.2022/ReaxFF/_downloads/bd665ad53e67ab052370fcce3d526423/ReaxFF.pdf).
- AIREBO-M form: O'Connor, Andzelm, and Robbins, *J. Chem. Phys.* **142**,
  024903 (2015), [doi:10.1063/1.4905549](https://doi.org/10.1063/1.4905549).

No ReaxFF or AIREBO-M fitted H/C/N/O parameter set is copied into this work.

## Published functional forms

ReaxFF uses a shielded Morse-like pair term

    E_vdw(r) = D [exp(alpha(1-rho/r_vdw))
                  - 2 exp(0.5 alpha(1-rho/r_vdw))]
    rho(r) = [r^p + gamma_w^(-p)]^(1/p)

(before its long-range taper). The published common exponent is `p = 1.69`.
As `r -> 0`, `rho -> 1/gamma_w`: both energy and force remain finite, and the
force tends to zero for `p > 1`. ReaxFF normally evaluates this interaction for
all atom pairs rather than suppressing it merely because a pair is bonded.

AIREBO-M replaces the divergent Lennard-Jones core with

    E_M(r) = D [z^2 - 2z],  z = exp[-alpha(r-r_min)].

At `r = 0` this is finite. The additional width `alpha` controls how quickly
the repulsive wall rises while `D` and `r_min` independently retain the outer
well depth and location.

## Diagnostic parameter mapping

Each H/C/N/O pair preserves the already audited UFF `r_min` and `D`. To avoid
choosing the Morse width by eye, each alternative also matches the UFF energy
at `1.5 r_min`.

For the ReaxFF diagnostic, `p = 1.69` is literature-backed. The shield core is
declared as `rho(0) = 0.5 r_min`; `r_vdw` is then chosen so the minimum in the
physical distance remains at `r_min`, and `alpha` is solved from the same outer
tail match. The `0.5` core fraction is a transparent diagnostic convention,
not a published ChemistryModel parameter and not a fit.

## Combined reactive + vdW handoff result

The existing smooth chemical-contact complement and long-range cutoff were
kept unchanged so this is the same handoff test as the UFF prototype. All
energies below are the maximum combined energy in the handoff window; forces
are the maximum absolute force there.

| pair | UFF barrier (eV) | AIREBO-M (eV) | ReaxFF (eV) | UFF max force (eV/A) | AIREBO-M | ReaxFF |
|---|---:|---:|---:|---:|---:|---:|
| H-H | 176.066 | 0.529 | 0.268 | 2218.109 | 30.481 | 27.103 |
| C-H | 14.571 | 0.255 | 0.160 | 147.202 | 17.806 | 16.829 |
| O-H | 28.292 | 0.289 | 0.170 | 300.708 | 22.300 | 20.989 |
| C-C | 1.617 | 0.110 | 0.079 | 23.610 | 11.069 | 10.837 |
| C-O | 1.513 | 0.095 | 0.068 | 25.443 | 12.584 | 12.359 |
| N-N | 1.133 | 0.074 | 0.053 | 21.989 | 13.044 | 12.878 |
| O-O | 0.390 | 0.038 | 0.029 | 11.202 | 7.881 | 7.793 |

For every tested pair:

- the accepted chemical minimum is unchanged to scan resolution;
- the raw short-range energy and force are finite;
- analytic forces agree with numerical energy derivatives;
- the handoff and cutoff are continuous in energy and force;
- one sensible outer dispersion minimum remains;
- no extra local minimum appears inside the handoff;
- **a positive local maximum (an artificial barrier) remains inside the
  handoff**.

## Decision

Neither candidate passes all selection criteria when inserted behind the
current chemical-contact suppression. ReaxFF shielding is the better of the
two numerical shapes, but adopting it now would retain a smaller artificial
barrier and would misrepresent the way published ReaxFF uses its vdW term.
AIREBO-M is similarly unsuitable with this handoff.

The next handoff design should therefore investigate an all-pairs,
chemistry-aware energy partition (starting from the ReaxFF architectural idea)
rather than multiplying a repulsive nonbonded curve by the current complement.
That is deliberately left for a separate reviewed design; no production vdW
implementation is recommended by this milestone.
