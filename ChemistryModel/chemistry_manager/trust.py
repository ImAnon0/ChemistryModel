"""Trusted-reactant policy derived from existing molecule/QM records."""

from enum import Enum
from pathlib import Path

import molecule_library
import qm_structure_validator


class MoleculeTrust(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    CM_VALIDATED = "CM_VALIDATED"
    QM_VALIDATED = "QM_VALIDATED"
    REJECTED = "REJECTED"


TRUSTED_LEVELS = {MoleculeTrust.CM_VALIDATED, MoleculeTrust.QM_VALIDATED}


def _explicit_level(molecule):
    value = molecule.get("trust_status")
    if value is None:
        return None
    try:
        return MoleculeTrust(str(value))
    except ValueError as problem:
        raise ValueError(
            f"unknown trust_status for {molecule.get('id')}: {value!r}"
        ) from problem


def _successful_qm_validation(record):
    if record.get("status") != "complete":
        return False
    comparison = record.get("comparison") or {}
    return bool(
        comparison.get("connectivity_preserved") is True
        and comparison.get("fragmented") is not True
        and comparison.get("rearranged") is not True
    )


def trust_level(molecule, qm_root=None):
    explicit = _explicit_level(molecule)
    if explicit is not None:
        return explicit

    root = (
        Path(qm_root)
        if qm_root is not None
        else qm_structure_validator.DEFAULT_ROOT
    )
    validations = qm_structure_validator.list_validations(
        molecule["id"], root=root
    )
    if any(_successful_qm_validation(record) for record in validations):
        return MoleculeTrust.QM_VALIDATED
    return MoleculeTrust.UNVALIDATED


def trusted_molecules(molecule_root=molecule_library.DEFAULT_ROOT, qm_root=None):
    trusted = []
    for molecule in molecule_library.list_molecules(root=molecule_root):
        level = trust_level(molecule, qm_root=qm_root)
        if level in TRUSTED_LEVELS:
            row = dict(molecule)
            row["trust_status"] = level.value
            trusted.append(row)
    return trusted

