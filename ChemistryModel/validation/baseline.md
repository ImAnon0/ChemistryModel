# ChemistryModel independent validation report

Revision: `ed34f6a1fd589945f92d548b33198fda6526d19f`
Mode: `full`
Force-field parameters changed by this report: **no**

Fit targets are labelled and are not counted as independent validation.

## Golden whole-model validation

**FINAL GOLDEN RESULT: PASS**

- **main runtime imports**: PASS (2.44 s)
- **heavy overcoordination guard**: PASS (2.41 s)
- **runner physics selection**: PASS (2.28 s)
- **heavy valence density matrix**: PASS (2.02 s)
- **optimised valence integration**: PASS (61.17 s)
- **batched heavy valence**: PASS (216.85 s)
- **large heavy valence states**: PASS (2.71 s)
- **cached H topology**: PASS (21.72 s)
- **factorised H grouped execution**: PASS (41.04 s)
- **H-state components**: PASS (4.47 s)
- **H-state factorised**: PASS (5.74 s)
- **H-state factorised NVE**: PASS (30.02 s)
- **index-select gather**: PASS (71.18 s)
- **smooth valence NVE**: PASS (50.41 s)
- **valence-state factorised fixed**: PASS (28.21 s)
- **valence-state promotion**: PASS (18.69 s)
- **molecule library**: PASS (0.15 s)
- **heavy state pressure diagnostics**: PASS (2.45 s)
- **smooth valence force probe**: PASS (5.45 s)
- **pytest suite**: PASS (221.33 s)
- **Dense optimised-valence soup stress**: PASS
  - seed 0: final C/N over-valent 0/0; max C/N coordination 4/4
  - seed 1: final C/N over-valent 0/0; max C/N coordination 4/3

## Baseline summary

**Strong**
- deterministic core, high-fidelity, recorder, and replay regressions
- preserved focused N-N and O-O recombination ensembles
- H2 effective depth now matches the thermochemical convention while preserving equilibrium length and harmonic curvature
- whole-model H2-forming abstraction energies now have the expected sign and are within 0.10 eV of their BDE298-derived comparisons
- most fitted X-H geometry and curvature diagnostics

**Weak**
- the relaxed methane abstraction barrier remains below its reference range
- N-N/N-O heavy-atom curvature transfer is weak or uncertain

**Uncertain**
- reaction-energy references are calibration-linked BDE298 differences, not independent like-for-like electronic energies
- ammonia lacks a bundled independent abstraction-barrier reference and broader collision ensembles are still needed
- the full H2 Morse curve is convention-dependent; spectroscopy constrains its equilibrium length and local curvature here
- double/triple bond transfer and other high-level pointwise potential curves remain sparsely validated

## 1. Parameter consistency

- NumPy/Torch: **GOOD**
- Maximum table difference: 0.0
- Single bonds are accepted calibration values; double/triple rows are inherited.
- Stored D is an effective classical Morse depth before bond-order/environment terms.

## 2. Geometry

| molecule | bond | model A | reference A | error % | classification/status |
| --- | --- | ---: | ---: | ---: | --- |
| H2 | H-H | 0.741 | 0.741 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| CH4 | C-H | 1.086 | 1.086 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| NH3 | N-H | 1.011 | 1.011 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| H2O | O-H | 0.960 | 0.960 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| C2H6 | C-C | 1.525 | 1.525 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| CH3NH2 | C-N | 1.470 | 1.471 | 0.07 | FIT TARGET - NOT INDEPENDENT |
| CH3OH | C-O | 1.427 | 1.427 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| N2H4 | N-N | 1.446 | 1.446 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| NH2OH | N-O | 1.453 | 1.453 | 0.00 | FIT TARGET - NOT INDEPENDENT |
| H2O2 | O-O | 1.475 | 1.475 | 0.00 | FIT TARGET - NOT INDEPENDENT |

## 3. Hold-out geometry

These are generic-pair transfer diagnostics, not fully relaxed molecular structures.

| molecule | bond | model A | reference A | error % | status |
| --- | --- | ---: | ---: | ---: | --- |
| ethanol | C-C | 1.525 | 1.512 | 0.86 | GOOD |
| ethanol | C-O | 1.427 | 1.431 | 0.28 | GOOD |
| dimethyl ether | C-O | 1.427 | 1.411 | 1.13 | GOOD |
| ethylamine | C-N | 1.470 | 1.469 | 0.07 | GOOD |

## 4. Harmonic curvature

| pair | model cm-1 | reference cm-1 | error % | comparison | status |
| --- | ---: | ---: | ---: | --- | --- |
| H-H | 4401.2 | 4401.2 | 0.0 | clean/strong comparison | FIT TARGET - NOT INDEPENDENT |
| C-H | 2936.2 | 2917.0 | 0.7 | approximate polyatomic normal mode | FIT TARGET - NOT INDEPENDENT |
| N-H | 3199.1 | 3337.0 | 4.1 | approximate polyatomic normal mode | FIT TARGET - NOT INDEPENDENT |
| O-H | 3750.8 | 3657.0 | 2.6 | approximate polyatomic normal mode | FIT TARGET - NOT INDEPENDENT |
| C-C | 1057.3 | 993.0 | 6.5 | mixed polyatomic normal mode | ACCEPTABLE |
| C-N | 1058.4 | 1044.8 | 1.3 | perturbed polyatomic normal mode | GOOD |
| C-O | 1057.6 | 1033.0 | 2.4 | polyatomic normal mode | GOOD |
| N-N | 733.2 | 1077.2 | 31.9 | polyatomic normal mode; rejected width fit | FAIL |
| N-O | 779.0 | 955.0 | 18.4 | polyatomic normal mode | WEAK |
| O-O | 877.2 | 877.0 | 0.0 | polyatomic mode used to select width | FIT TARGET - NOT INDEPENDENT |

## 5. Potential curves

- **H2 H-H**: GOOD; minimum 0.7409 A, dissociation coordinate 4.517 eV. External pointwise comparison: CONVENTION-DEPENDENT DIAGNOSTIC.
- **CH4 C-H**: GOOD; minimum 1.0890 A, dissociation coordinate 4.572 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **C2H6 C-C**: GOOD; minimum 1.5250 A, dissociation coordinate 3.651 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **CH3NH2 C-N**: GOOD; minimum 1.4675 A, dissociation coordinate 3.708 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **CH3OH C-O**: GOOD; minimum 1.4290 A, dissociation coordinate 3.719 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **N2H4 N-N**: GOOD; minimum 1.4455 A, dissociation coordinate 1.768 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **NH2OH N-O**: GOOD; minimum 1.4510 A, dissociation coordinate 2.098 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.
- **H2O2 O-O**: GOOD; minimum 1.4730 A, dissociation coordinate 1.439 eV. External pointwise comparison: INSUFFICIENT REFERENCE DATA.

## 6. Dissociation energies

| reaction | model kJ/mol | reference kJ/mol | error % | status |
| --- | ---: | ---: | ---: | --- |
| H2 -> H + H | 435.8 | 435.8 | 0.0 | GOOD |
| CH4 -> CH3 + H | 439.0 | 439.0 | 0.0 | FIT TARGET - NOT INDEPENDENT |
| NH3 -> NH2 + H | 449.0 | 449.0 | 0.0 | FIT TARGET - NOT INDEPENDENT |
| H2O -> OH + H | 498.0 | 498.0 | 0.0 | FIT TARGET - NOT INDEPENDENT |
| C2H6 -> CH3 + CH3 | 348.0 | 377.0 | 7.7 | ACCEPTABLE |
| CH3NH2 -> CH3 + NH2 | 356.0 | 356.0 | 0.0 | FIT TARGET - NOT INDEPENDENT |
| CH3OH -> CH3 + OH | 358.0 | 384.6 | 6.9 | ACCEPTABLE |
| H2O2 -> OH + OH | 146.0 | 210.4 | 30.6 | FAIL |

## 7. Bond-depth diagnostics

These pair-depth differences are diagnostics only. They are not ChemistryModel reaction thermochemistry.

- **H + CH4 -> H2 + CH3**: GOOD; model 0.033 eV, reference 0.033 eV.
- **H + H2O -> H2 + OH**: GOOD; model 0.645 eV, reference 0.645 eV.
- **H + NH3 -> H2 + NH2**: GOOD; model 0.137 eV, reference 0.137 eV.
- **CH3 + CH3 -> C2H6**: INSUFFICIENT REFERENCE DATA.
- **CH3 + OH -> CH3OH**: INSUFFICIENT REFERENCE DATA.
- **NH2 + NH2 -> N2H4**: INSUFFICIENT REFERENCE DATA.
- **OH + OH -> H2O2**: INSUFFICIENT REFERENCE DATA.

## 8. Whole-model reaction energies

These use relaxed complete reactant and product species through the production Torch energy function. The reference is a BDE298-derived thermochemical difference, so it is not a like-for-like zero-temperature electronic-energy observable.

- **H + CH4 -> H2 + CH3**: GOOD; model Delta E 0.092 eV, BDE298-derived reference 0.033 eV; relaxation converged: True.
- **H + H2O -> H2 + OH**: GOOD; model Delta E 0.644 eV, BDE298-derived reference 0.645 eV; relaxation converged: True.
- **H + NH3 -> H2 + NH2**: GOOD; model Delta E 0.137 eV, BDE298-derived reference 0.137 eV; relaxation converged: True.

## 9. Reaction barriers

Frozen scans are screening diagnostics; relaxed full-mode scans are the stronger result.

- **formaldehyde**: WEAK; model 0.126 eV (relaxed scan).
- **water**: WEAK; model 0.000 eV (relaxed scan).
- **methane**: WEAK; model 0.165 eV (relaxed scan).
- **ammonia**: INSUFFICIENT REFERENCE DATA; model 0.168 eV (relaxed scan).

## 10. Dynamic reactions

- **NH2 + NH2 -> N2H4**: NOT RUN.
- **OH + OH -> H2O2**: NOT RUN.
- **H + H -> H2**: INSUFFICIENT REFERENCE DATA.
- **H + CH3 -> CH4**: INSUFFICIENT REFERENCE DATA.
- **CH3 + CH3 -> C2H6**: INSUFFICIENT REFERENCE DATA.
- **CH3 + OH -> CH3OH**: INSUFFICIENT REFERENCE DATA.

## 11. Mixture behaviour

- Large matched mixture: **INSUFFICIENT REFERENCE DATA**

## 12. Numerical stability

- **GOOD**
- All reported NVE probes and preserved focused batches are checked separately in JSON.

## 13. Performance

- CPU probe: 432.4 steps/s (0.231 s for 100 steps).

## 14. Transferability

- **Geometry**: ACCEPTABLE: ethanol/ethylamine transfer is close; dimethyl-ether C-O exposes environment dependence
- **X H Curvature**: STRONGEST AREA, but most X-H molecular bands remain calibration-linked
- **Heavy Heavy Curvature**: WEAK/UNCERTAIN: mixed normal modes; N-N width fit was rejected by capture dynamics
- **Dissociation**: MODERATE: effective pair depths are diagnostics, not molecular thermochemistry, and several hold-outs differ materially
- **Reaction Energies**: UNCERTAIN: complete-species engine energies are now reported, but their experimental comparators are BDE298-derived and calibration-linked
- **Barriers**: reported separately; no parameter tuning performed here
- **Dynamics**: GOOD for preserved focused N-N and O-O recombination ensembles; other reactions need baselines
- **Numerical Stability**: GOOD across deterministic and NVE probes

Important gaps:

- no bundled high-level pointwise potential curves
- collision accessibility is static-barrier based; no trajectory probability ensemble
- limited preserved focused ensembles for H-H, C-H, C-C and C-O capture
- double/triple bond tables remain inherited and unvalidated in this stage

No overall accuracy percentage is reported.
