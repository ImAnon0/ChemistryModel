from hf_surface_scan import (apply_system, measure_barrier,
                             describe_spectators)

apply_system("water")
found = measure_barrier("high_fidelity", "water", mixing=0.52, relax=True)

print(f"barrier   {found['barrier']:+.4f} eV")
print(f"saddle    r(donor) {found['donor']:.3f}  "
      f"r(transfer) {found['transfer']:.3f}")
lines, pinned = describe_spectators("water", found["spectators"])
for line in lines:
    print(line)
print("pinned:", pinned or "none")