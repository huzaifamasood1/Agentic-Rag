"""
reasoning.py - Turns Chain of Thought / Tree of Thoughts answers into structures the UI can draw.

The prompts in prompts.py ask for a strict line format; these parsers read it. If the model
ignores the format, the parsers return None and the UI shows the plain text instead.
"""
import re

SUPPORT_COLORS = {"strong": "#2e7d32", "weak": "#f9a825", "none": "#c62828"}


def _clean(text: str) -> str:
    return text.strip().strip("*").strip()


def split_final(answer: str):
    """Return (reasoning_part, final_answer). final_answer is '' if there is no marker."""
    match = list(re.finditer(r"final answer\s*:", answer, flags=re.I))
    if not match:
        return answer, ""
    cut = match[-1]
    return answer[:cut.start()], answer[cut.end():].strip()


def parse_chain(answer: str):
    """-> {"steps": [(title, text), ...], "final": str} or None."""
    reasoning, final = split_final(answer)
    steps = re.findall(r"^\W*Step\s*\d+\s*[-:–—]\s*([^:\n]+?)\s*:\s*(.+)$", reasoning, flags=re.M | re.I)
    if not steps or not final:
        return None
    return {"steps": [(_clean(t), _clean(x)) for t, x in steps], "final": final}


def parse_tree(answer: str):
    """-> {"branches": {"A": {"text", "support", "why"}}, "best", "best_why", "final"} or None."""
    reasoning, final = split_final(answer)
    branches = {}
    for letter, text in re.findall(r"^\W*Branch\s*([A-C])\s*:\s*(.+)$", reasoning, flags=re.M | re.I):
        branches[letter.upper()] = {"text": _clean(text), "support": "none", "why": ""}
    for letter, level, why in re.findall(
            r"^\W*Support\s*([A-C])\s*:\s*(strong|weak|none)\b\W*(.*)$", reasoning, flags=re.M | re.I):
        if letter.upper() in branches:
            branches[letter.upper()].update(support=level.lower(), why=_clean(why))
    best = re.search(r"^\W*Best\s*:\s*([A-C])\b\W*(.*)$", reasoning, flags=re.M | re.I)
    if len(branches) < 2 or not best or not final:
        return None
    return {"branches": branches, "best": best.group(1).upper(),
            "best_why": _clean(best.group(2)), "final": final}


def _wrap(text: str, width: int = 38, max_lines: int = 5) -> str:
    words, lines, line = text.replace('"', "'").split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 3] + "..."
    return "\\n".join(lines)


def tree_to_dot(question: str, tree: dict) -> str:
    """Graphviz source: question -> branches (coloured by support) -> final answer."""
    out = ['digraph T {', 'rankdir=LR; bgcolor="transparent";',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fontcolor=white];',
           f'q [label="{_wrap(question, 30, 4)}", fillcolor="#455a64"];']
    for letter, b in tree["branches"].items():
        color = SUPPORT_COLORS.get(b["support"], "#757575")
        chosen = letter == tree["best"]
        label = f'{letter} ({b["support"]})\\n{_wrap(b["text"])}'
        pen = ', penwidth=4, color="#1565c0"' if chosen else ""
        out.append(f'{letter} [label="{label}", fillcolor="{color}"{pen}];')
        out.append(f'q -> {letter};')
    out.append(f'ans [label="Final answer\\n{_wrap(tree["final"], 38, 6)}", fillcolor="#1565c0"];')
    out.append(f'{tree["best"]} -> ans [penwidth=3, color="#1565c0"];')
    out.append("}")
    return "\n".join(out)
