import tarfile

ARCHIVE = r"C:\Users\Mikey\Documents\reactiveengine dataset\wb97xd3.tar.gz"

with tarfile.open(ARCHIVE, "r:gz") as tar:
    member = next(m for m in tar if m.name.endswith(".log")
                  and "/ts" in m.name)
    print("FILE:", member.name)
    lines = tar.extractfile(member).read().decode("utf-8", "replace").splitlines()

hits = [(i, l) for i, l in enumerate(lines) if "Frequency" in l]
print("frequency lines:", len(hits))
for i, l in hits[:4]:
    print(repr(l))

for i, l in enumerate(lines):
    if "VIBRATIONAL ANALYSIS" in l:
        print("\n--- vib section ---")
        print("\n".join(lines[i:i + 20]))
        break