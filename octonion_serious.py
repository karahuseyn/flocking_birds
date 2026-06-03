# octonion_serious.py -- serious medical prompts answered in COMPOSITION mode (compose_logic:
# fuzzy scan -> cluster into topic-headings -> inferential role order -> octonion IMPLIES chain
# -> realise). Big pool (patient + 80k mined). Shows the full composed answer, role chain, and
# the IMPLIES-chain fidelity. Honest -- whatever the system produces.
import json, base64
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_composer import compose_logic, compose

PROMPTS = [
    "what is hypertension and how is it treated?",
    "what causes a stroke?",
    "what is asthma?",
    "what is rheumatoid arthritis?",
    "how is pneumonia treated?",
    "what causes kidney stones?",
    "what is depression?",
    "what are the risk factors for heart disease?",
    "what is osteoporosis?",
    "what is a migraine?",
]

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_big.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=30000, verbose=True)
    out = []
    for q in PROMPTS:
        txt, roles, fid = compose_logic(bot, q, return_meta=True)
        out += ["Q: " + q,
                "  roles: " + " -> ".join(roles) + ("   [implies-chain fidelity=%.3f]" % fid),
                "  COMPOSITION: " + txt[:460], ""]
    print("B64S:" + base64.b64encode("\n".join(out).encode()).decode())
