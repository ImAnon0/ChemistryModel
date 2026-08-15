"""
Reconstruct approximate diabatic state energies from the EXISTING minimax-path
open-shell SAPT run. No new Psi4 calculations are needed.

It combines, for each SAPT state calculation:

    E_state_proxy =
        E_HF(monomer A, dimer basis)
      + E_HF(monomer B, dimer basis)
      + E_SAPT0(interaction)

and also forms an exchange-only counterpart:

    E_state_exchange_proxy =
        E_HF(monomer A, dimer basis)
      + E_HF(monomer B, dimer basis)
      + E_EXCH10(interaction)

The monomer HF energies are parsed directly from Psi4's verbose output, where
each SAPT0 job runs in this fixed order:

    Dimer HF
    Monomer A HF
    Monomer B HF
    SAPT0

The CSV supplies the job ordering and SAPT component energies.

Inputs:
    research_data/sapt/sapt_open_shell_minimax_path_probe.csv
    research_data/sapt/sapt_open_shell_minimax_path_probe_psi4.out

Output:
    research_data/sapt/sapt_reconstructed_state_gaps.csv

Run from either Python environment:

    py sapt_reconstruct_state_gaps.py

No fitting is performed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HARTREE_TO_EV = 27.211386245988

DEFAULT_CSV = Path(
    "research_data/sapt/sapt_open_shell_minimax_path_probe.csv"
)

DEFAULT_PSI4_OUT = Path(
    "research_data/sapt/sapt_open_shell_minimax_path_probe_psi4.out"
)

DEFAULT_OUTPUT = Path(
    "research_data/sapt/sapt_reconstructed_state_gaps.csv"
)

FINAL_ENERGY_RE = re.compile(
    r"@DF-UHF Final Energy:\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)"
)


def read_success_rows(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("status", "").strip().lower() == "ok"
        ]

    if not rows:
        raise RuntimeError(
            f"No successful rows in {path}"
        )

    return rows


def parse_hf_triplets(path):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    values = [
        float(match.group(1))
        for match in FINAL_ENERGY_RE.finditer(text)
    ]

    if len(values) % 3 != 0:
        raise RuntimeError(
            "Expected Psi4 output to contain 3 UHF final energies per SAPT job "
            f"(dimer, monomer A, monomer B), but found {len(values)} energies."
        )

    return [
        {
            "dimer_hf_H": values[index],
            "monomer_a_hf_H": values[index + 1],
            "monomer_b_hf_H": values[index + 2],
        }
        for index in range(0, len(values), 3)
    ]


def to_float(row, field):
    return float(row[field])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )

    parser.add_argument(
        "--psi4-output",
        type=Path,
        default=DEFAULT_PSI4_OUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    rows = read_success_rows(
        args.csv
    )

    triplets = parse_hf_triplets(
        args.psi4_output
    )

    if len(triplets) != len(rows):
        raise RuntimeError(
            "Job-count mismatch: "
            f"{len(rows)} successful CSV rows but {len(triplets)} SAPT HF triplets "
            "in the Psi4 output. If the Psi4 output contains multiple runs, "
            "rerun the minimax probe with a fresh output file."
        )

    enriched = []

    for row, hf in zip(rows, triplets):
        monomer_sum_eV = (
            hf["monomer_a_hf_H"]
            + hf["monomer_b_hf_H"]
        ) * HARTREE_TO_EV

        dimer_hf_eV = (
            hf["dimer_hf_H"]
            * HARTREE_TO_EV
        )

        sapt_total = to_float(
            row,
            "sapt_total_eV",
        )

        exch10 = to_float(
            row,
            "exch10_eV",
        )

        enriched.append(
            {
                **row,
                **hf,
                "dimer_hf_eV": dimer_hf_eV,
                "monomer_sum_hf_eV": monomer_sum_eV,
                "state_total_proxy_eV": (
                    monomer_sum_eV
                    + sapt_total
                ),
                "state_exchange_proxy_eV": (
                    monomer_sum_eV
                    + exch10
                ),
            }
        )

    by_point = {}

    for row in enriched:
        key = (
            row["system"],
            row["point_label"],
        )

        by_point.setdefault(
            key,
            {},
        )[row["state"]] = row

    summary_rows = []

    print(
        "RECONSTRUCTED DIABATIC STATE-GAP PROXY"
    )
    print(
        "======================================="
    )
    print(
        "HF monomer reference energies + SAPT interaction; no new calculation."
    )
    print()
    print(
        "system        point          idx  pathE    model_gap   "
        "HFfrag_gap   +EXCH_gap   +SAPT_gap   dimerHF_diff"
    )

    for key, states in by_point.items():
        if not (
            "reactant" in states
            and "product" in states
        ):
            continue

        reactant = states["reactant"]
        product = states["product"]

        hf_frag_gap = (
            product["monomer_sum_hf_eV"]
            - reactant["monomer_sum_hf_eV"]
        )

        exch_interaction_gap = (
            to_float(product, "exch10_eV")
            - to_float(reactant, "exch10_eV")
        )

        total_interaction_gap = (
            to_float(product, "sapt_total_eV")
            - to_float(reactant, "sapt_total_eV")
        )

        exchange_proxy_gap = (
            product["state_exchange_proxy_eV"]
            - reactant["state_exchange_proxy_eV"]
        )

        total_proxy_gap = (
            product["state_total_proxy_eV"]
            - reactant["state_total_proxy_eV"]
        )

        dimer_hf_diff = (
            product["dimer_hf_eV"]
            - reactant["dimer_hf_eV"]
        )

        model_gap = to_float(
            reactant,
            "model_signed_gap_product_minus_reactant_eV",
        )

        summary = {
            "system": reactant["system"],
            "point_label": reactant["point_label"],
            "path_index": int(
                float(
                    reactant["path_index"]
                )
            ),
            "donor_A": to_float(
                reactant,
                "donor_A",
            ),
            "transfer_A": to_float(
                reactant,
                "transfer_A",
            ),
            "relative_path_energy_eV": to_float(
                reactant,
                "relative_path_energy_eV",
            ),
            "model_gap_eV": model_gap,
            "hf_fragment_internal_gap_eV": hf_frag_gap,
            "exch10_interaction_gap_eV": exch_interaction_gap,
            "sapt_total_interaction_gap_eV": total_interaction_gap,
            "exchange_proxy_gap_eV": exchange_proxy_gap,
            "sapt_total_proxy_gap_eV": total_proxy_gap,
            "dimer_hf_state_difference_eV": dimer_hf_diff,
        }

        summary_rows.append(
            summary
        )

        print(
            f"{summary['system']:<12} "
            f"{summary['point_label']:<13} "
            f"{summary['path_index']:3d}  "
            f"{summary['relative_path_energy_eV']:+7.3f}  "
            f"{model_gap:+10.3f}  "
            f"{hf_frag_gap:+10.3f}  "
            f"{exchange_proxy_gap:+10.3f}  "
            f"{total_proxy_gap:+10.3f}  "
            f"{dimer_hf_diff:+12.6f}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "system",
        "point_label",
        "path_index",
        "donor_A",
        "transfer_A",
        "relative_path_energy_eV",
        "model_gap_eV",
        "hf_fragment_internal_gap_eV",
        "exch10_interaction_gap_eV",
        "sapt_total_interaction_gap_eV",
        "exchange_proxy_gap_eV",
        "sapt_total_proxy_gap_eV",
        "dimer_hf_state_difference_eV",
    ]

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  HFfrag_gap is the missing internal fragment-reference contribution."
    )
    print(
        "  +EXCH_gap = HF fragment gap + open-shell EXCH10 interaction gap."
    )
    print(
        "  +SAPT_gap = HF fragment gap + full SAPT0 interaction gap."
    )
    print(
        "  The dimerHF_diff audit should be ~0 because both fragmentations use "
        "the same total geometry/electron count; a large value means the parser "
        "mapping is wrong."
    )
    print()
    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
