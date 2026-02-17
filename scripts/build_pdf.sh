#!/usr/bin/env bash
# Build a nicely formatted PDF of the Phase 0 report using pandoc + xelatex.
# Usage: bash scripts/build_pdf.sh [output_path]
#
# Preprocesses the markdown to extract title/author/date into YAML front matter,
# then converts via pandoc. Figures resolve relative to the research/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESEARCH_DIR="$PROJECT_DIR/research"
OUTPUT="${1:-$PROJECT_DIR/anamnesis.pdf}"
TMPFILE="$(mktemp /tmp/report_XXXXXX.md)"
trap 'rm -f "$TMPFILE"' EXIT

cd "$RESEARCH_DIR"

# Preprocess: inject YAML front matter from the H1/H2 title lines and metadata block,
# then strip those original lines so pandoc uses the YAML metadata for the title page.
# Also rebalance two table separator lines where the first column is too narrow.
python3 - phase0_report.md "$TMPFILE" <<'PYEOF'
import sys, re

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    lines = f.readlines()

# Extract title (H1), subtitle (H2), then find author/date in the blockquote
title = ""
subtitle = ""
author = ""
date = ""
skip_until = 0

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("# ") and not stripped.startswith("## ") and not title:
        title = stripped[2:].strip()
        skip_until = i + 1
        continue
    if stripped.startswith("## ") and not subtitle and skip_until == i:
        subtitle = stripped[3:].strip()
        skip_until = i + 1
        continue
    if stripped == "" and skip_until == i:
        skip_until = i + 1
        continue
    # Parse the blockquote metadata lines
    if stripped.startswith("> **Author:**"):
        m = re.search(r'\*\*Author:\*\*\s*(.*)', stripped[2:])
        if m:
            # Preserve markdown links so pandoc renders them as hyperlinks
            author = m.group(1).strip()
        skip_until = i + 1
        continue
    if stripped.startswith("> **Date:**"):
        m = re.search(r'\*\*Date:\*\*\s*(.*)', stripped[2:])
        if m:
            date = m.group(1).strip()
        skip_until = i + 1
        continue
    if stripped.startswith("> **Status:**") or stripped.startswith("> **Data"):
        skip_until = i + 1
        continue
    if stripped == "---" and i <= skip_until + 1:
        skip_until = i + 1
        continue
    if skip_until > 0 and stripped == "":
        skip_until = i + 1
        continue
    break

# Build YAML front matter
yaml = "---\n"
yaml += f'title: "{title}"\n'
if subtitle:
    yaml += f'subtitle: "{subtitle}"\n'
if author:
    # author field needs to allow inline markdown for links
    yaml += f'author: "{author}"\n'
if date:
    yaml += f'date: "{date} — [github.com/LuxiaSL/anamnesis](https://github.com/LuxiaSL/anamnesis)"\n'
yaml += "---\n\n"

# Write preprocessed file: YAML header + remaining content (skip extracted lines)
remaining = lines[skip_until:]
# Strip the leading --- separator if present
while remaining and remaining[0].strip() in ("", "---"):
    remaining = remaining[1:]

# Rebalance specific pipe-table separators where the first column is too narrow.
# Pandoc infers column widths from relative separator dash lengths.
TABLE_FIXES = {
    # §4.1 Progressive Confound Removal: give Metric column ~35% width
    "|--------|-------------------|------------------------|---------------------------|":
    "|-------------------------------|--------------|------------------|------------------|",
    # §4.3 Double Dissociation: give Tier column ~30% width
    "|------|----------------------------------------|----------------------------|":
    "|--------------------------|-------------------------------|----------------------|",
    # D.5 Cross-Run Tier Ablation: give Tier column ~35% width
    "|------|-------------------|------------------------|---------------------------|":
    "|-------------------------------|--------------|------------------|------------------|",
}

output_lines = []
for line in remaining:
    stripped = line.strip()
    if stripped in TABLE_FIXES:
        output_lines.append(TABLE_FIXES[stripped] + "\n")
    else:
        output_lines.append(line)

with open(dst, "w") as f:
    f.write(yaml)
    f.write("".join(output_lines))
PYEOF

pandoc "$TMPFILE" \
  -o "$OUTPUT" \
  --pdf-engine=xelatex \
  --resource-path=".:$PROJECT_DIR/outputs/paper_figures:$PROJECT_DIR" \
  --toc \
  --toc-depth=3 \
  -V documentclass=article \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V toccolor=black \
  -V papersize=letter \
  -V header-includes='\usepackage{booktabs}' \
  -V header-includes='\usepackage{float}' \
  -V header-includes='\usepackage{caption}' \
  -V header-includes='\captionsetup{font=small,labelfont=bf}' \
  -V header-includes='\usepackage{fancyhdr}' \
  -V header-includes='\pagestyle{fancy}' \
  -V header-includes='\fancyhead[L]{\small Computational Signatures in Transformer Internal States}' \
  -V header-includes='\fancyhead[R]{\small Luxia, 2026}' \
  -V header-includes='\fancyfoot[C]{\thepage}' \
  -V header-includes='\usepackage{microtype}' \
  -V header-includes='\usepackage{parskip}' \
  -V mainfont='Libertinus Serif' \
  -V sansfont='Noto Sans' \
  -V monofont='Source Code Pro'

echo "PDF written to: $OUTPUT"
