from live import LiveSimulation, run_live_window


# Drag the slider to heat or cool the system while it runs.
# Turn the thermostat off to watch it coast under pure
# Newtonian dynamics with energy conserved.

simulation = LiveSimulation(
    unit_cells_per_side=3,
    number_density=0.85,
    time_step=0.002,
    cutoff_distance=2.5,
    target_temperature_kelvin=120.0,
    thermostat_friction=2.0
)

run_live_window(
    simulation=simulation,
    steps_per_frame=8
)
