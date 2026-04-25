# BUILDING.md — What to run RIGHT NOW

## Active task — none

v10 dense-height deep dive shipped (completion-promise word: **DENSE-MANGO**).
See FINDINGS §14 and `docs/NEXT_GPU_SESSION_v10.md` for the original plan.

## Next candidate tasks (pull from TODO.md when ready)

- **v11 GPU session — go wide AND deep on all 8 pairs**
    See `docs/NEXT_GPU_SESSION_v11.md` for the full plan. Blocked on
    re-extracting the v10 NPZs first (the Vast.ai instance died before
    upload). Key experiments: z-vs-lexical disentanglement (P3d),
    cross-pair transfer on dense grid (P3e), orthogonalized increment
    R² (P3c), causal head ablation (P5).
- **One-shot reproducibility upload (≤1 min, requires write HF token)**
    ```
    python scripts/upload_v10_to_hf.py
    ```
    The v10 NPZs (~826 MB) are not yet on
    `xrong1729/mech-interp-relativity-activations`. Until they are,
    `python scripts/fetch_from_hf.py --only v10` will hit an empty
    folder and a fresh `git pull` cannot re-run the v10 analyses
    without re-doing the GPU extraction. The token used by the rest
    of v10 was read-only.
- Paper draft (ICML MI Workshop, May 8 deadline)
