#!/usr/bin/env python3
"""Post-process pandoc LaTeX output for arXiv submission."""

import re
import sys
from pathlib import Path

def load_cite_mapping(path: str) -> dict[str, str]:
    """Load numeric ref → citation key mapping."""
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

def build_author_cite_map() -> list[tuple[re.Pattern, str]]:
    """Build regex patterns mapping author-year text to \\cite commands.

    Returns list of (pattern, replacement) tuples, ordered from most
    specific to least specific to avoid partial matches.
    """
    # Each entry: (text_pattern, cite_key, is_textual)
    # We handle both parenthetical and textual citations

    # Unique single-author patterns (unambiguous)
    single_authors = {
        "Afzal et al.": "afzal2025",
        "Behrouz et al.": "behrouz2025",
        "Boxo et al.": "boxo2025",
        "Brandon et al.": "brandon2024",
        "Cai et al.": "cai2024",
        "Chen et al.": "chen2025",
        "Chua et al.": "chua2025",
        "Cloud et al.": "cloud2025",
        "Do et al.": "do2025",
        "Dong et al.": "dong2025",
        "Heo et al.": "heo2025",
        "Huang et al.": "huang2025",
        "Lewis et al.": "lewis2020",
        "Liang et al.": "liang2025",
        "Liu et al.": "liu2024",
        "Packer et al.": "packer2023",
        "Pochinkov et al.": "pochinkov2025",
        "Qi et al.": "qi2025",
        "Ramachandran et al.": "ramachandran2025",
        "Schrodi et al.": "schrodi2025",
        "Servedio et al.": "servedio2025",
        "Shelmanov et al.": "shelmanov2025",
        "Shi et al.": "shi2026",
        "Turner et al.": "turner2023",
        "Wei et al.": "wei2025",
        "Xiao et al.": "xiao2024",
        "Xing et al.": "xing2025",
        "Zou et al.": "zou2023",
        "Hooper et al.": "hooper2024",
        "Zur et al.": "zur2025",
        "Ni et al.": "ni2025",
        "Zhong et al.": "zhong2024",
        "Ali et al.": "ali2025",
        "Gurnee et al.": "gurnee2025",
    }

    # Two-author patterns
    two_authors = {
        r"Hewitt\s*\\?&\s*Liang": ("hewitt2019", "1919"),
        r"Wang\s*\\?&\s*Xu": ("wang2025c", "2025"),
        r"Frising\s*\\?&\s*Balcells": ("frising2025", "2025"),
        r"Vardhan\s*\\?&\s*Teja": ("vardhan2026", "2026"),
        r"Reimers\s*\\?&\s*Gurevych": ("reimers2019", "1919"),
    }

    # Ambiguous authors that need year disambiguation
    # Wang: wang2023 (LongMem), wang2025a (CoE), wang2025b (TARG), wang2025c (ThoughtProbe)
    # Li: li2025a (MemOS), li2023 (ITI), li2024 (SnapKV), li2026 (steering vectors)
    # Zhang: zhang2025a (MHAD), zhang2023 (H2O), zhang2025b (reasoning/COLM)
    # Betley: betley2025a (Emergent Misalignment), betley2025b (Weird Gen)

    replacements = []

    # Two-author patterns first (more specific)
    for pattern, (key, year_suffix) in two_authors.items():
        # Textual: "Hewitt & Liang (2019)"
        replacements.append((
            re.compile(rf'({pattern})\s*\(({year_suffix[:4]})\)'),
            rf'\\citet{{{key}}}'
        ))
        # Also handle ", YEAR" form
        replacements.append((
            re.compile(rf'({pattern}),?\s*({year_suffix[:4]})'),
            rf'\\citet{{{key}}}'
        ))

    # Single-author patterns
    for author, key in single_authors.items():
        escaped = re.escape(author).replace(r'\ ', r'\s*')
        year = key[-4:]

        # Textual: "Author et al. (2025)"
        replacements.append((
            re.compile(rf'{escaped}\s*\({year}\)'),
            rf'\\citet{{{key}}}'
        ))
        # Parenthetical component: "Author et al., 2025" (inside parens)
        replacements.append((
            re.compile(rf'{escaped},?\s*{year}'),
            rf'CITEKEY:{key}'  # placeholder, handle parens separately
        ))

    # Ambiguous: handle specific known patterns from the paper
    # Betley et al., 2025 appears in two forms — handle by context
    # For now, map the general "Betley et al." to betley2025a (more common reference)
    replacements.append((
        re.compile(r'Betley\s+et\s+al\.\s*\(2025\)'),
        r'\\citet{betley2025a}'
    ))
    replacements.append((
        re.compile(r'Betley\s+et\s+al\.,?\s*2025'),
        r'CITEKEY:betley2025a'
    ))

    # Wollschlager (special char)
    replacements.append((
        re.compile(r'Wollschlager\s+et\s+al\.\s*\(2025\)'),
        r'\\citet{wollschlager2025}'
    ))
    replacements.append((
        re.compile(r'W[oö]llschl[aä]ger\s+et\s+al\.,?\s*2025'),
        r'CITEKEY:wollschlager2025'
    ))

    # Li disambiguation by year
    for year, key in [("2023", "li2023"), ("2024", "li2024"), ("2025", "li2025a"), ("2026", "li2026")]:
        replacements.append((
            re.compile(rf'Li\s+et\s+al\.\s*\({year}\)'),
            rf'\\citet{{{key}}}'
        ))
        replacements.append((
            re.compile(rf'Li\s+et\s+al\.,?\s*{year}'),
            rf'CITEKEY:{key}'
        ))

    # Wang disambiguation by year
    for year, key in [("2023", "wang2023"), ("2025", "wang2025a")]:
        replacements.append((
            re.compile(rf'Wang\s+et\s+al\.\s*\({year}\)'),
            rf'\\citet{{{key}}}'
        ))
        replacements.append((
            re.compile(rf'Wang\s+et\s+al\.,?\s*{year}'),
            rf'CITEKEY:{key}'
        ))

    # Zhang disambiguation by year
    for year, key in [("2023", "zhang2023"), ("2025", "zhang2025b")]:
        replacements.append((
            re.compile(rf'Zhang\s+et\s+al\.\s*\({year}\)'),
            rf'\\citet{{{key}}}'
        ))
        replacements.append((
            re.compile(rf'Zhang\s+et\s+al\.,?\s*{year}'),
            rf'CITEKEY:{key}'
        ))

    # Zhang et al.~(2025; COLM 2025) — special pattern in the text
    replacements.append((
        re.compile(r'Zhang\s+et\s+al\.~?\s*\(2025;\s*COLM\s*2025\)'),
        r'\\citet{zhang2025b}'
    ))

    return replacements


def convert_citations(text: str) -> str:
    """Convert author-year citations to \\cite commands."""
    replacements = build_author_cite_map()

    for pattern, repl in replacements:
        text = pattern.sub(repl, text)

    # Now handle parenthetical groups: convert (CITEKEY:a; CITEKEY:b) to \citep{a, b}
    # First, find parenthetical groups containing CITEKEY markers
    def convert_paren_group(match: re.Match) -> str:
        content = match.group(1)
        # Extract all CITEKEY:xxx tokens
        keys = re.findall(r'CITEKEY:(\S+)', content)
        if keys:
            # Check if there's non-CITEKEY text remaining
            remaining = re.sub(r'CITEKEY:\S+', '', content)
            remaining = re.sub(r'[;,\s]+', '', remaining)
            # Filter out venue annotations like "ICML 2025", "ICLR 2026", etc.
            remaining = re.sub(r'(?:ICML|ICLR|NeurIPS|EMNLP|ACL|COLM|AAAI|IJCAI)\s*\d{4}', '', remaining)
            remaining = remaining.strip()
            if not remaining:
                return r'\citep{' + ', '.join(keys) + '}'
            else:
                # Mixed content, keep as-is but replace CITEKEY markers
                result = content
                for key in keys:
                    result = result.replace(f'CITEKEY:{key}', rf'\citet{{{key}}}', 1)
                return '(' + result + ')'
        return match.group(0)

    text = re.sub(r'\(([^)]*CITEKEY:[^)]*)\)', convert_paren_group, text)

    # Clean up any remaining CITEKEY markers (shouldn't happen but safety)
    text = re.sub(r'CITEKEY:(\S+)', lambda m: rf'\citet{{{m.group(1)}}}', text)

    # Clean up semicolons inside cite commands: \citep{key1;, key2} → \citep{key1, key2}
    text = re.sub(r';,', ',', text)
    # Also: \citep{key;} → \citep{key}
    text = re.sub(r';(\})', r'\1', text)

    # Fix specific unconverted citations that have non-standard formatting
    manual_fixes = [
        ('(Hewitt \\& Liang, 2019)', '\\citep{hewitt2019}'),
        ('\\textbf{Vardhan \\& Teja} (2026)', '\\citet{vardhan2026}'),
        ('Servedio et al.~(ACL 2025)', '\\citet{servedio2025}'),
        ('(Chen, Arditi, Evans et al., 2025)', '\\citep{chen2025}'),
        ('Ali et al.~{[}52{]}', '\\citet{ali2025}'),
        ('Gurnee et al.~{[}53{]}', '\\citet{gurnee2025}'),
        # Fix wrong-key bugs: disambiguation for multi-paper authors
        ('\\textbf{Weird Generalization} \\citep{betley2025a}', '\\textbf{Weird Generalization} \\citep{betley2025b}'),
        # TARG cite fix — handle both possible patterns
        ('TARG; \\citet{wang2025a}', 'TARG; \\citet{wang2025b}'),
        ('(TARG; \\citet{wang2025b})', '\\citep[TARG;][]{wang2025b}'),
        # Uncited papers — add \cite where they're mentioned by name
        ('\\textbf{MHAD} (IJCAI 2025)', '\\textbf{MHAD} \\citep{zhang2025a}'),
        ('(yunoshev, 2026)', '\\citep{yunoshev2026}'),
        ('(Reimers \\& Gurevych, 2019)', '\\citep{reimers2019}'),
        ('Reimers \\& Gurevych, 2019', '\\citet{reimers2019}'),
        # DSEM and MLP Memory — mentioned by name without \cite
        ('\\textbf{Dynamic Steering with Episodic Memory (DSEM)} (ACL 2025 Findings)',
         '\\textbf{Dynamic Steering with Episodic Memory (DSEM)} \\citep{do2025}'),
        ('\\textbf{MLP Memory} (2025)', '\\textbf{MLP Memory} \\citep{wei2025}'),
    ]
    for old, new in manual_fixes:
        text = text.replace(old, new)

    return text


def fix_section_levels(text: str) -> str:
    """Promote section levels and strip embedded numbering.

    Pandoc output hierarchy (from H2/H3/H4 markdown):
      \\subsection{1. Introduction}     → \\section{Introduction}
      \\subsubsection{2.1 Probing...}   → \\subsection{Probing...}
      \\paragraph{4.8.1 Method}         → \\subsubsection{Method}

    Must strip embedded numbers so LaTeX auto-numbering doesn't double up.
    Appendix headings (A., B., etc.) are preserved differently.
    """
    # Step 1: Handle #### headings (\paragraph) FIRST — promote and strip "X.Y.Z " prefix
    text = re.sub(
        r'\\paragraph\{(\d+\.\d+\.\d+)\s+([^}]*)\}',
        r'\\subsubsection{\2}',
        text
    )
    # Any remaining \paragraph without numbered prefix
    text = re.sub(r'\\paragraph\{', r'\\subsubsection{', text)

    # Step 2: Handle ### headings (\subsubsection) — promote and strip "X.Y " prefix
    # But preserve appendix sub-headings (A.1, B.2, etc.)
    text = re.sub(
        r'\\subsubsection\{(\d+\.\d+)\s+([^}]*)\}',
        r'\\subsection{\2}',
        text
    )
    # Appendix sub-headings: \subsubsection{A.1 Run 1...} → \subsection{...}
    # Keep the appendix letter prefix for these
    text = re.sub(
        r'\\subsubsection\{([A-E]\.\d+)\s+([^}]*)\}',
        r'\\subsection{\2}',
        text
    )
    # Appendix main headings: \subsubsection{A. Full Mode Prompts} → \section{...}
    text = re.sub(
        r'\\subsubsection\{([A-E])\.\s+([^}]*)\}',
        r'\\section{\2}',
        text
    )

    # Step 3: Handle ## headings (\subsection) — promote and strip "X." prefix
    text = re.sub(
        r'\\subsection\{(\d+)\.\s+([^}]*)\}',
        r'\\section{\2}',
        text
    )
    # Catch any \subsection without numbered prefix (e.g., Abstract)
    text = re.sub(r'\\subsection\{Abstract\}', r'\\section{Abstract}', text)

    # Step 4: Handle appendix D sub-sub-headings that use \paragraph
    # e.g., \paragraph{D.5 Cross-Run...} — already handled by step 1 catchall
    # but D.X patterns need letter prefix stripped
    text = re.sub(
        r'\\subsubsection\{([A-E]\.\d+)\s+([^}]*)\}',
        r'\\subsection{\2}',
        text
    )

    return text


def fix_abstract(text: str) -> str:
    """Convert Abstract section to proper \\begin{abstract} environment."""
    # Find the abstract section and its content
    pattern = re.compile(
        r'\\section\{Abstract\}\\label\{abstract\}\s*\n(.*?)(?=\\section\{)',
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        abstract_content = match.group(1).strip()
        # Remove the horizontal rule if present
        abstract_content = re.sub(
            r'\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}',
            '',
            abstract_content
        ).strip()
        abstract_block = f'\\begin{{abstract}}\n{abstract_content}\n\\end{{abstract}}\n\n'
        # Use string replacement instead of regex sub to avoid backslash issues
        text = text[:match.start()] + abstract_block + text[match.end():]
    return text


def fix_figures(text: str) -> str:
    """Convert pandoc figure output to proper LaTeX figure environments."""
    # Pandoc generates: \begin{figure}\n\centering\n\includegraphics{...}\n\caption{...}\n\end{figure}
    # Or inline: \includegraphics{...}
    # We want: \begin{figure}[htbp]\n\centering\n\includegraphics[width=\textwidth]{figures/...}\n\caption{...}\n\label{...}\n\end{figure}

    # Fix figure paths — strip directory prefixes, point to figures/ subdir
    text = re.sub(
        r'\\includegraphics(\[.*?\])?\{(?:\.\./outputs/paper_figures/|outputs/paper_figures/)?([^}]+)\}',
        r'\\includegraphics[width=\\textwidth]{figures/\2}',
        text
    )

    # Add [htbp] to figure environments
    text = text.replace(r'\begin{figure}', r'\begin{figure}[htbp]')

    # Generate labels from figure filenames
    def add_label(match: re.Match) -> str:
        full = match.group(0)
        filename_match = re.search(r'figures/([^}]+)\.png', full)
        if filename_match:
            # Strip 'fig_' prefix from filename to avoid 'fig:fig_...' labels
            label_base = filename_match.group(1)
            if label_base.startswith('fig_'):
                label_base = label_base[4:]
            label = label_base.replace('_', ':')
            if r'\label{' not in full:
                full = full.replace(r'\end{figure}', f'\\label{{fig:{label}}}\n\\end{{figure}}')
        return full

    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', add_label, text, flags=re.DOTALL)

    return text


def remove_horizontal_rules(text: str) -> str:
    """Remove pandoc horizontal rules."""
    text = re.sub(
        r'\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}\s*',
        '\n',
        text
    )
    return text


def remove_references_section(text: str) -> str:
    """Remove the inline References section (replaced by \\bibliography).

    Preserves appendix content that follows the references.
    After section level fixes, appendix headings are \\section{Full Mode Prompts} etc.
    """
    # References may be \section or \subsection depending on promotion order
    ref_start = -1
    for cmd in [r'\section{References}', r'\subsection{References}']:
        idx = text.find(cmd)
        if idx != -1:
            ref_start = idx
            break
    if ref_start == -1:
        return text

    # Find where appendices start — look for known appendix section titles
    # These are the titles AFTER number stripping: "Full Mode Prompts", "Topic Lists", etc.
    appendix_titles = [
        'Full Mode Prompts', 'Topic Lists', 'Feature Definitions',
        'Per-Run Detailed Results', 'Reproduction Instructions',
    ]
    appendix_start = len(text)
    for title in appendix_titles:
        for cmd in [f'\\section{{{title}}}', f'\\subsection{{{title}}}']:
            idx = text.find(cmd, ref_start + 1)
            if idx != -1 and idx < appendix_start:
                appendix_start = idx

    if appendix_start < len(text):
        appendix_content = text[appendix_start:]
        # Remove the "Appendices" heading if present (it's just a label, \appendix handles it)
        appendix_content = re.sub(
            r'\\(?:sub)?section\{Appendices\}\\label\{appendices\}\s*',
            '',
            appendix_content
        )
        # Insert \appendix command before the first appendix section
        text = text[:ref_start] + '\n\\appendix\n\n' + appendix_content
    else:
        # No appendices found — just remove references to end
        text = text[:ref_start]

    return text


def fix_tables(text: str) -> str:
    """Fix table column widths where pandoc allocates too little to the first column."""

    # Strategy: find longtable environments with p{} column specs where the first
    # column gets less than 15% of columnwidth, and rebalance.
    def rebalance_table(match: re.Match) -> str:
        table = match.group(0)

        # Extract all \real{X.XXXX} values
        reals = re.findall(r'\\real\{([0-9.]+)\}', table)
        if not reals:
            return table  # simple l/r/c columns, leave alone

        widths = [float(r) for r in reals]

        # Only rebalance if first column is under 15% and there are 3+ columns
        if len(widths) >= 3 and widths[0] < 0.15:
            # Set first column to 25%, distribute the rest proportionally
            old_first = widths[0]
            new_first = 0.25
            remaining_old = sum(widths[1:])
            scale = (1.0 - new_first) / remaining_old if remaining_old > 0 else 1.0
            new_widths = [new_first] + [w * scale for w in widths[1:]]

            # Replace in the table text
            for old_val, new_val in zip(reals, new_widths):
                table = table.replace(
                    f'\\real{{{old_val}}}',
                    f'\\real{{{new_val:.4f}}}',
                    1  # replace one at a time to handle duplicates
                )

        return table

    text = re.sub(
        r'\\begin\{longtable\}.*?\\end\{longtable\}',
        rebalance_table,
        text,
        flags=re.DOTALL
    )

    return text


def fix_acknowledgments(text: str) -> str:
    """Convert Acknowledgments section to unnumbered."""
    text = text.replace(
        r'\section{Acknowledgments}',
        r'\section*{Acknowledgments}'
    )
    text = text.replace(
        r'\subsection{Acknowledgments}',
        r'\section*{Acknowledgments}'
    )
    return text


def main():
    body_path = Path(__file__).parent / "body.tex"
    output_path = Path(__file__).parent / "content.tex"

    text = body_path.read_text()

    # Apply transformations in order
    text = fix_section_levels(text)
    text = fix_abstract(text)
    text = remove_horizontal_rules(text)
    text = fix_figures(text)
    text = remove_references_section(text)
    text = fix_acknowledgments(text)
    text = convert_citations(text)
    text = fix_tables(text)

    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    output_path.write_text(text)
    print(f"Processed body written to {output_path}")
    print(f"Lines: {len(text.splitlines())}")


if __name__ == "__main__":
    main()
