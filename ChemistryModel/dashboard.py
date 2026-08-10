import argparse
import json
import os
import webbrowser


# ============================================================
# One page holding everything
# ============================================================
#
# The browser shows one run at a time, which is right for reading
# a report and wrong for finding patterns. This writes a single
# HTML file containing every run from every batch: a sortable,
# filterable table, per-batch averages, and a scatter plot for
# looking at one measure against another.
#
# No server and no dependencies. The data is embedded in the
# file, so it opens by double-clicking and keeps working after
# the recordings are deleted or moved.
#
#   py dashboard.py runs
#   py dashboard.py runs --open


COLUMNS = [
    ("batch", "batch", "text"),
    ("run", "run", "number"),
    ("seed", "seed", "number"),
    ("stable", "ok", "bool"),
    ("mixture", "mixture", "text"),
    ("atoms", "atoms", "number"),
    ("box", "box", "number"),
    ("picoseconds", "ps", "number"),
    ("strikes", "strikes", "number"),
    ("cool_temperature", "trap K", "number"),
    ("heavy_bonds_formed", "bonds", "number"),
    ("late_formed", "late+", "number"),
    ("late_broke", "late-", "number"),
    ("turnovers", "turnover", "number"),
    ("largest_closed", "closed", "number"),
    ("largest_any", "any", "number"),
    ("most_carbon", "carbons", "number"),
    ("best_chain", "chain", "number"),
    ("best_tail", "tail", "number"),
    ("species_count", "species", "number"),
    ("final_potential", "PE", "number"),
    ("headline", "biggest product", "text"),
]

# The ones worth averaging in the summary at the top.

SUMMARY = [
    "heavy_bonds_formed",
    "late_formed",
    "turnovers",
    "largest_closed",
    "largest_any",
    "most_carbon",
    "best_tail",
    "species_count",
]


def find_batches(root):
    found = []

    if os.path.exists(os.path.join(root, "index.json")):
        found.append((os.path.basename(os.path.abspath(root)), root))

    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)

            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "index.json")
            ):
                found.append((name, path))

    return found


def collect(root):
    rows = []

    for label, path in find_batches(root):
        with open(os.path.join(path, "index.json")) as handle:
            for entry in json.load(handle):
                row = {"batch": label}

                for key, title, kind in COLUMNS:
                    if key == "batch":
                        continue

                    value = entry.get(key)

                    if value is None:
                        value = "" if kind == "text" else None

                    row[key] = value

                row["closed_shell"] = entry.get("closed_shell", [])
                row["species"] = entry.get("species_seen", [])

                rows.append(row)

    return rows


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chemistry runs</title>
<style>
 :root {
   --ink: #1b1b1a; --dim: #6b6b66; --line: #d8d6cf;
   --panel: #faf9f5; --accent: #2f6f9f; --warn: #b4471f;
 }
 * { box-sizing: border-box; }
 body {
   margin: 0; padding: 24px; background: #f1efe8; color: var(--ink);
   font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
 }
 h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
 .sub { color: var(--dim); margin-bottom: 20px; }
 .panel {
   background: var(--panel); border: 1px solid var(--line);
   border-radius: 8px; padding: 16px; margin-bottom: 18px;
 }
 table { border-collapse: collapse; width: 100%; font-size: 13px; }
 th, td {
   padding: 6px 9px; text-align: right;
   border-bottom: 1px solid var(--line); white-space: nowrap;
 }
 th {
   position: sticky; top: 0; background: var(--panel);
   cursor: pointer; user-select: none; font-weight: 600;
 }
 th:hover { color: var(--accent); }
 th.sorted::after { content: " \\2193"; color: var(--accent); }
 th.sorted.up::after { content: " \\2191"; }
 td.text, th.text { text-align: left; }
 tbody tr:hover { background: #efece3; }
 tr.unstable { color: var(--warn); }
 tr.unstable td:first-child::before { content: "! "; }
 .controls { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
 input[type=search], select {
   font: inherit; padding: 5px 8px; border: 1px solid var(--line);
   border-radius: 5px; background: white;
 }
 label { color: var(--dim); }
 .scroll { max-height: 60vh; overflow: auto; }
 .count { color: var(--dim); margin-top: 8px; }
 canvas { background: white; border: 1px solid var(--line); border-radius: 5px; }
 .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }
 .swatch { width: 11px; height: 11px; border-radius: 2px; display: inline-block; margin-right: 5px; }
</style>
</head>
<body>

<h1>Chemistry runs</h1>
<div class="sub" id="headline"></div>

<div class="panel">
  <div id="summary"></div>
</div>

<div class="panel">
  <div class="controls">
    <input type="search" id="filter" placeholder="filter, any text">
    <label><input type="checkbox" id="onlyStable" checked> hide unstable runs</label>
    <label>batch
      <select id="batchPick"><option value="">all</option></select>
    </label>
  </div>
  <div class="count" id="count"></div>
  <div class="scroll"><table id="table"></table></div>
</div>

<div class="panel">
  <div class="controls">
    <label>x <select id="xPick"></select></label>
    <label>y <select id="yPick"></select></label>
  </div>
  <canvas id="plot" width="900" height="380"></canvas>
  <div class="legend" id="legend"></div>
</div>

<script>
const ROWS = __ROWS__;
const COLUMNS = __COLUMNS__;
const SUMMARY = __SUMMARY__;

const COLOURS = ["#2f6f9f","#b4471f","#1d7a55","#7a5bb0","#a8871f",
                 "#3f8fa0","#9f3f6f","#5f7a2f"];

const batches = [...new Set(ROWS.map(r => r.batch))].sort();
const colourOf = {};
batches.forEach((b, i) => colourOf[b] = COLOURS[i % COLOURS.length]);

let sortKey = "batch", sortUp = true;

function visible() {
  const text = document.getElementById("filter").value.toLowerCase();
  const onlyStable = document.getElementById("onlyStable").checked;
  const batch = document.getElementById("batchPick").value;

  return ROWS.filter(r => {
    if (onlyStable && r.stable === false) return false;
    if (batch && r.batch !== batch) return false;
    if (!text) return true;
    return JSON.stringify(r).toLowerCase().includes(text);
  });
}

function mean(values) {
  const good = values.filter(v => typeof v === "number" && isFinite(v));
  if (!good.length) return null;
  return good.reduce((a, b) => a + b, 0) / good.length;
}

function drawSummary() {
  const rows = visible();
  const table = ["<table><tr><th class='text'>batch</th><th>n</th>"];

  SUMMARY.forEach(key => {
    const col = COLUMNS.find(c => c[0] === key);
    table.push("<th>" + (col ? col[1] : key) + "</th>");
  });

  table.push("</tr>");

  batches.forEach(b => {
    const subset = rows.filter(r => r.batch === b);
    if (!subset.length) return;

    table.push("<tr><td class='text'><span class='swatch' style='background:"
      + colourOf[b] + "'></span>" + b + "</td><td>" + subset.length + "</td>");

    SUMMARY.forEach(key => {
      const value = mean(subset.map(r => r[key]));
      table.push("<td>" + (value === null ? "-" : value.toFixed(1)) + "</td>");
    });

    table.push("</tr>");
  });

  table.push("</table>");
  document.getElementById("summary").innerHTML = table.join("");
}

function drawTable() {
  const rows = visible().slice().sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x === y) return 0;
    if (x === null || x === undefined || x === "") return 1;
    if (y === null || y === undefined || y === "") return -1;
    const order = (typeof x === "string") ? x.localeCompare(y) : x - y;
    return sortUp ? order : -order;
  });

  const head = COLUMNS.map(([key, title, kind]) => {
    const classes = [kind === "text" ? "text" : ""];
    if (key === sortKey) { classes.push("sorted"); if (sortUp) classes.push("up"); }
    return "<th class='" + classes.join(" ") + "' data-key='" + key + "'>" + title + "</th>";
  }).join("");

  const body = rows.map(r => {
    const cells = COLUMNS.map(([key, title, kind]) => {
      let value = r[key];
      if (kind === "bool") value = (value === false) ? "no" : "yes";
      else if (value === null || value === undefined) value = "";
      else if (typeof value === "number" && !Number.isInteger(value))
        value = value.toFixed(1);
      return "<td class='" + (kind === "text" ? "text" : "") + "'>" + value + "</td>";
    }).join("");
    return "<tr class='" + (r.stable === false ? "unstable" : "") + "'>" + cells + "</tr>";
  }).join("");

  document.getElementById("table").innerHTML =
    "<thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody>";

  document.getElementById("count").textContent =
    rows.length + " runs shown of " + ROWS.length;

  document.querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const key = th.dataset.key;
      if (key === sortKey) sortUp = !sortUp; else { sortKey = key; sortUp = true; }
      drawTable();
    };
  });
}

function drawPlot() {
  const canvas = document.getElementById("plot");
  const c = canvas.getContext("2d");
  const xKey = document.getElementById("xPick").value;
  const yKey = document.getElementById("yPick").value;

  c.clearRect(0, 0, canvas.width, canvas.height);

  const rows = visible().filter(r =>
    typeof r[xKey] === "number" && typeof r[yKey] === "number");

  if (!rows.length) return;

  const xs = rows.map(r => r[xKey]), ys = rows.map(r => r[yKey]);
  const pad = 52;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = v => pad + (v - x0) / ((x1 - x0) || 1) * (canvas.width - pad - 20);
  const sy = v => canvas.height - pad - (v - y0) / ((y1 - y0) || 1) * (canvas.height - pad - 20);

  c.strokeStyle = "#d8d6cf"; c.fillStyle = "#6b6b66"; c.font = "11px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const gy = y0 + (y1 - y0) * i / 4;
    c.beginPath(); c.moveTo(pad, sy(gy)); c.lineTo(canvas.width - 20, sy(gy)); c.stroke();
    c.textAlign = "right"; c.fillText(gy.toFixed(1), pad - 6, sy(gy) + 4);
    const gx = x0 + (x1 - x0) * i / 4;
    c.textAlign = "center"; c.fillText(gx.toFixed(1), sx(gx), canvas.height - pad + 16);
  }

  c.fillStyle = "#1b1b1a"; c.textAlign = "center";
  c.fillText(xKey, canvas.width / 2, canvas.height - 12);
  c.save(); c.translate(14, canvas.height / 2); c.rotate(-Math.PI / 2);
  c.fillText(yKey, 0, 0); c.restore();

  rows.forEach(r => {
    c.fillStyle = colourOf[r.batch];
    c.globalAlpha = 0.8;
    c.beginPath(); c.arc(sx(r[xKey]), sy(r[yKey]), 4.5, 0, 6.283); c.fill();
  });
  c.globalAlpha = 1;

  document.getElementById("legend").innerHTML = batches.map(b =>
    "<span><span class='swatch' style='background:" + colourOf[b] + "'></span>" + b + "</span>"
  ).join("");
}

function redraw() { drawSummary(); drawTable(); drawPlot(); }

const numeric = COLUMNS.filter(c => c[2] === "number");
["xPick", "yPick"].forEach((id, i) => {
  const select = document.getElementById(id);
  numeric.forEach(([key, title]) => {
    const option = document.createElement("option");
    option.value = key; option.textContent = title;
    select.appendChild(option);
  });
  select.value = i === 0 ? "most_carbon" : "largest_any";
  select.onchange = drawPlot;
});

const picker = document.getElementById("batchPick");
batches.forEach(b => {
  const option = document.createElement("option");
  option.value = b; option.textContent = b;
  picker.appendChild(option);
});

document.getElementById("filter").oninput = redraw;
document.getElementById("onlyStable").onchange = redraw;
picker.onchange = redraw;

document.getElementById("headline").textContent =
  ROWS.length + " runs across " + batches.length + " batches"
  + "  -  " + ROWS.filter(r => r.stable === false).length + " unstable";

redraw();
</script>
</body>
</html>
"""


def build(root, output):
    rows = collect(root)

    if not rows:
        raise SystemExit(f"no batches found under {root}")

    page = PAGE
    page = page.replace("__ROWS__", json.dumps(rows))
    page = page.replace("__COLUMNS__", json.dumps(COLUMNS))
    page = page.replace("__SUMMARY__", json.dumps(SUMMARY))

    with open(output, "w", encoding="utf-8") as handle:
        handle.write(page)

    return rows, output


def main():
    parser = argparse.ArgumentParser(
        description="Write a dashboard of every run to one HTML file."
    )

    parser.add_argument("directory", nargs="?", default="runs")
    parser.add_argument("--out", default="dashboard.html")
    parser.add_argument(
        "--open", action="store_true",
        help="open it in a browser once written"
    )

    options = parser.parse_args()

    rows, path = build(options.directory, options.out)

    batches = len({row["batch"] for row in rows})
    unstable = sum(1 for row in rows if row.get("stable") is False)

    print(
        f"{len(rows)} runs from {batches} batches "
        f"({unstable} unstable) -> {path}"
    )

    if options.open:
        webbrowser.open("file://" + os.path.abspath(path))


if __name__ == "__main__":
    main()
