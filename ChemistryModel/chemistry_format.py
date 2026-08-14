"""Canonical chemistry display formatting, independent of atom storage order."""


def molecular_formula(symbols):
    """Return a Hill-system molecular formula for an iterable of symbols."""
    counts = {}
    for raw in symbols:
        symbol = str(raw)
        counts[symbol] = counts.get(symbol, 0) + 1

    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    else:
        order = sorted(counts)

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] != 1 else "")
        for symbol in order
    )
