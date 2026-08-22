from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm


def test_qeq_solver_preserves_neutral_charge():
    class Dummy:
        pass

    import torch
    from chemistry_engine.context import InteractionContext

    ctx = InteractionContext(
        positions=torch.tensor([[0.0, 0.0, 0.0],
                                [0.74, 0.0, 0.0]], dtype=torch.float64),
        element_types=None,
        atomic_numbers=(1, 1),
        neighbours=None,
        neighbour_mask=None,
        box_size=10.0,
        box_count=1,
        atoms_per_box=2,
        batch_assignment=(0, 0),
    )

    charges = ElectrostaticEnergyTerm().solve_charges(ctx)

    assert torch.allclose(charges.sum(), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
