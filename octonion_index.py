"""Octonion / Fano project -- one index over all the modules.

Everything in this project is built on ONE substrate: the octonion algebra, whose
multiplication table is the Fano plane (7 imaginary units = 7 points, 7 lines).
`octonion_lm.py` defines that algebra (octo_mul, fano_lines); every other module
imports it.  This file is the single entry point: it groups the ~18 modules into
the storyline, prints the map, and can run any of them.

    python3 octonion_index.py            # print the project map
    python3 octonion_index.py <module>   # run one module's demo (e.g. attention)

The one coherent finding across the project (each claim base64-verified in its own
module): the octonion structure is *load-bearing* exactly where its ALGEBRA is
exploited -- bind/unbind recall (100% vs 3% for elementwise) and Fano-specific
transport (0.949 vs 0.628 for random generators) -- and interchangeable with plain
vectors where you only bundle-and-classify.  Gradient-free training helps exactly
where the encoding has NOT already done the work (colliding codes, interactions).
"""
import sys, runpy

# (module, one-line description) grouped by theme -- the project's storyline
GROUPS = [
    ("Foundation: the octonion / Fano substrate", [
        ("octonion_lm",        "octonion algebra + Fano-path char LM (no backprop); the base everything imports"),
        ("octonion_attention", "linear attention = fast-weight memory; bind/unbind, 100% associative recall"),
    ]),
    ("A transformer mechanism, gradient-free and O(n)", [
        ("octonion_induction", "causal induction head; matches softmax attention, softmax-free"),
        ("octonion_delta",     "outer-product block memory; matches softmax capacity at O(n)"),
        ("octonion_speed",     "O(n) vs softmax O(n^2): 33x faster, constant memory at 8k tokens"),
        ("octonion_deep",      "depth (composition) + gating, gradient-free, from the algebra"),
        ("octonion_reason",    "stacked memories -> multi-step in-context reasoning"),
        ("octonion_seq",       "integrated multi-head model; in-context learning on real text"),
    ]),
    ("Retrieval applications (where recall shines)", [
        ("octonion_symptom",   "symptom -> disease ranker; 100% top-1, coherent top-7 differential"),
        ("octonion_chat",      "retrieval QA bot over ~37k medical Q&A; 96% paraphrase recall"),
        ("octonion_answer",    "long structured answers (495k sentences, LSH + symbolic plan)"),
    ]),
    ("Gradient-free training: where it helps, and where it doesn't", [
        ("octonion_hebb",      "competitive Hebbian (LVQ/STDP) -- honest negative on bundled codes"),
        ("octonion_nonlinear", "nonlinear features + Hebbian -- why a random HDC lift has no headroom"),
        ("octonion_select",    "conjunction selection: 9->98% on interactions, fails on linear data"),
        ("octonion_codebook",  "adaptive codebook: +3 pts on real data when codes collide"),
    ]),
    ("Fano paths as transport maps (object -> object)", [
        ("octonion_transport", "Fano rotations as smooth transports; learns functions, Fano matters"),
        ("octonion_analogy",   "a:b::c:d on word analogies; +6-10 pts on morphology"),
        ("octonion_compose",   "chaining transports -> zero-shot multi-step reasoning"),
    ]),
    ("Toward fluent gradient-free generation (root attempts)", [
        ("octonion_semantic",  "PMI-SVD meaning + n-gram fluency + evolving discourse state; prompt-faithful"),
        ("octonion_proposition","octonion-BIND discourse state (role(x)filler, Fano roles); matches real topic-drift"),
        ("octonion_logic",     "classical connectives on octonions; sequential inference exact (chain 1.000)"),
        ("octonion_infer",     "multi-hop modus-ponens SEARCH; proves 6-hop chains (fidelity 1.0), rejects unreachable"),
        ("octonion_gpt",       "END-TO-END: all components in one gradient-free O(n) generator on a large corpus"),
        ("octonion_code",      "structure-aware Python CODE generation; idioms/signatures, not execution"),
    ]),
]

def print_map():
    print(__doc__.strip().split("\n\n")[0])
    print()
    for title, mods in GROUPS:
        print(f"  {title}")
        for name, desc in mods:
            print(f"    {name:22s} {desc}")
        print()
    print("  run one:  python3 octonion_index.py <module>   (e.g. attention, transport, codebook)")

def main():
    if len(sys.argv) < 2:
        print_map(); return
    name = sys.argv[1]
    if not name.startswith("octonion_"):
        name = "octonion_" + name
    known = {m for _, mods in GROUPS for m, _ in mods}
    if name not in known:
        print(f"unknown module '{name}'.\n"); print_map(); return
    sys.argv = sys.argv[1:]                      # forward remaining args to the module
    runpy.run_module(name, run_name="__main__")

if __name__ == "__main__":
    main()
