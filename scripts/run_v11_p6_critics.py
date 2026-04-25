"""v11 P6 — five critic agents review the v11 results via the Anthropic API.

Each critic gets:
  - The PLANNING.md (frozen scientific spec)
  - STATUS.md, FINDINGS.md tail (v9/v10 context)
  - The v11 results JSON summaries and figure file list
  - A focused prompt for that critic's lens

Outputs:
  results/v11/critic_<name>.md    per critic
  results/v11/critics_summary.md   short aggregator that lists each critic's verdict and top-3 concerns

Critics:
  methodology      — data leakage, circular reasoning, threshold gaming, CV folds
  alternative      — cheaper explanations for each positive finding
  statistical      — effect sizes, multi-comp correction, CI, sample-size adequacy
  novelty          — is this new vs Anthropic manifold work, Goodfire, Park, Nanda IOI
  narrative        — internal consistency across v9/v10/v11; flag contradictions
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CRITIC_PROMPTS: dict[str, str] = {
    "methodology": (
        "You are an independent methodology critic for a mech-interp study. "
        "The team has just finished v11 (cross-model dense extraction + analyses). "
        "Your job is to find data-leakage, circular reasoning, threshold-gaming, "
        "selection-bias, or CV-fold issues that would undermine the claims. "
        "Specifically scrutinize:\n"
        "  - The orthogonalized increment R² (P3c) — is the residualization on "
        "    ẑ_{L-1} actually removing all of L-1's contribution, or only the linear part?\n"
        "  - The z-vs-lexical disentanglement (P3d) — does the W_U-based lexical "
        "    direction give a fair test, or is it confounded by output-class frequency?\n"
        "  - The 9B head taxonomy — were thresholds picked top-quartile of *what* and "
        "    is that data-leak across the same heads being later ablated?\n"
        "  - Cross-pair transfer (P3e) — is the seed-0 subset enough or is variance large?\n"
        "Return a punch list (≤5 items): each item is (a) concrete concern, (b) which result "
        "JSON it points to, (c) recommended fix or whether to retract a claim."
    ),
    "alternative": (
        "You are an alternative-explanation critic. Your job: for each positive v11 finding, "
        "propose the simplest cheaper explanation that could produce the same result. "
        "Do not be polite — assume the team is overclaiming and your job is to find the "
        "cheap explanation they missed. Examples of cheap explanations to consider:\n"
        "  - cross-pair transfer at 40% could be shared token embeddings for number words, "
        "    not a domain-general z-code\n"
        "  - SAE z-features could be tracking output-token frequency, not z\n"
        "  - low Jaccard across pairs could be SAE training instability, not lack of shared structure\n"
        "  - 9B head taxonomy alignment with 2B could be coincidence given the threshold gates\n"
        "Return ≤5 alternative explanations, each tied to a specific v11 result file."
    ),
    "statistical": (
        "You are a statistical critic. Scrutinize effect sizes, multiple-comparisons, "
        "and confidence intervals. v11 ran ~30k cells × 8 pairs × 2 models × multiple "
        "metrics. Specifically check:\n"
        "  - Are R² values reported with bootstrap or analytical CIs? Did the team correct for "
        "    multi-pair / multi-layer testing?\n"
        "  - The cross-pair transfer 8×8 matrix has 56 off-diagonal cells — what's the "
        "    Bonferroni-corrected significance threshold for non-zero slope?\n"
        "  - The N=400 cells (per pair, after seed-collapse) — is that enough to claim "
        "    a 7-D intrinsic manifold? Stability under bootstrap resample?\n"
        "  - The 'top-quartile' thresholds in head taxonomy — is the resulting tag count "
        "    significant vs a permutation null?\n"
        "Return ≤5 specific statistical concerns and the smallest fix to address each."
    ),
    "novelty": (
        "You are a novelty critic comparing v11 against existing mech-interp literature: "
        "Anthropic's manifold geometry work (Toy Models of Superposition, Scaling Monosemanticity), "
        "Goodfire's SAE-feature decomposition, Kiho Park et al.'s causal inner product on the "
        "unembedding, Neel Nanda's IOI circuit, and the broader probing literature. "
        "For each headline v11 claim, score (1-5) how novel it is and identify the closest "
        "prior result. Specifically address:\n"
        "  - Is 'encode-vs-use as a layer-depth phenomenon' novel beyond Geva et al.'s "
        "    'transformer feed-forward layers are key-value memories'?\n"
        "  - Is the W_U-based lexical-vs-z disentanglement (P3d-fixed) novel vs Park's "
        "    causal inner product framework?\n"
        "  - Is the 9B replication a workshop-paper 'replication contribution' or appendix?\n"
        "Return ≤5 items, each with a (claim, novelty 1-5, closest prior, recommendation)."
    ),
    "narrative": (
        "You are a narrative critic checking internal consistency across the v9 / v10 / v11 "
        "story. Read FINDINGS.md §13 (v9 layer sweep), §14 (v10 dense-height), and the v11 "
        "result JSONs. Identify any place where v11 contradicts v9 or v10 without "
        "acknowledging it, or where v11 silently re-confirms a refuted hypothesis. "
        "Specifically check:\n"
        "  - v9 §10.3 refuted Park causal inner product. Does v11 P3d/P3e re-introduce a "
        "    similar claim?\n"
        "  - v10 §14.6 has a 2B head taxonomy. Does v11 9B taxonomy structurally agree, "
        "    contradict, or simply differ in detail?\n"
        "  - v9 SAE Jaccard was 0.06 cross-pair. Does v11's higher-N replication change that?\n"
        "Return ≤5 items: each is a contradiction or unacknowledged tension, with a "
        "specific FINDINGS reference."
    ),
}


def load_results_summary() -> str:
    """Concatenate the v11 result JSONs into a single context blob (truncated)."""
    out_dir = REPO / "results" / "v11"
    if not out_dir.exists():
        return "<no v11 results found>"
    bits: list[str] = []
    # Top-level model summaries first
    for model in ("gemma2-2b", "gemma2-9b"):
        mdir = out_dir / model
        if not mdir.exists(): continue
        for p in sorted(mdir.glob("*.json")):
            content = p.read_text()
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"
            bits.append(f"--- {p.relative_to(REPO)} ---\n{content}")
        # Per-pair summaries
        for pair_dir in sorted(mdir.iterdir()):
            if not pair_dir.is_dir(): continue
            for p in sorted(pair_dir.glob("*.json")):
                if "meta" in p.name: continue   # the meta is already verbose; skip
                content = p.read_text()
                if len(content) > 4000:
                    content = content[:4000] + "\n... [truncated]"
                bits.append(f"--- {p.relative_to(REPO)} ---\n{content}")
    return "\n\n".join(bits)[:120000]   # cap at 120K chars to fit in context


def list_figures() -> str:
    fig_dir = REPO / "figures" / "v11"
    if not fig_dir.exists(): return "<no figures yet>"
    return "\n".join(sorted(str(p.relative_to(REPO)) for p in fig_dir.rglob("*.png")))


def call_critic(name: str, prompt: str, results_blob: str, figures_list: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    project_context = ""
    for f in ("PLANNING.md", "STATUS.md"):
        p = REPO / f
        if p.exists():
            project_context += f"\n\n=== {f} ===\n" + p.read_text()[:6000]
    findings_path = REPO / "FINDINGS.md"
    findings_tail = ""
    if findings_path.exists():
        text = findings_path.read_text()
        findings_tail = "\n\n=== FINDINGS.md (last 12000 chars: §13/§14) ===\n" + text[-12000:]
    sys_prompt = (
        "You are an independent research critic. Be specific, terse, and concrete — "
        "name file paths, layer numbers, and metric values from the supplied JSONs. "
        "Cap your reply at ~600 words.\n\n"
        f"{prompt}"
    )
    user = (
        f"PROJECT CONTEXT:\n{project_context}{findings_tail}\n\n"
        f"V11 RESULT JSONS (truncated):\n{results_blob}\n\n"
        f"V11 FIGURE FILES:\n{figures_list}\n\n"
        f"Now write your critique."
    )
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=sys_prompt,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def main():
    out_dir = REPO / "results" / "v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = load_results_summary()
    figs = list_figures()
    print(f"[P6] results blob: {len(blob)} chars, figures: {figs.count(chr(10)) + 1}")

    summary_lines = ["# v11 P6 critic round\n"]
    for name, prompt in CRITIC_PROMPTS.items():
        print(f"[P6] running critic: {name}...", flush=True)
        try:
            text = call_critic(name, prompt, blob, figs)
        except Exception as e:
            text = f"<critic {name} failed: {e}>"
            print(f"[P6]   FAILED: {e}", file=sys.stderr)
        out_path = out_dir / f"critic_{name}.md"
        out_path.write_text(f"# {name} critic\n\n{text}\n")
        print(f"[P6]   wrote {out_path.relative_to(REPO)}")
        summary_lines.append(f"## {name}\n\n{text[:1500]}\n\n*(see {out_path.name} for full)*")

    (out_dir / "critics_summary.md").write_text("\n".join(summary_lines))
    print(f"\n[P6] wrote {out_dir / 'critics_summary.md'}")


if __name__ == "__main__":
    main()
