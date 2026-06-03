import json, base64
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_composer import compose_logic

P = ["what is hypertension and how is it treated?", "what is type 2 diabetes?",
     "what causes a stroke?", "what is a migraine?"]

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_big.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=30000, verbose=True)
    out = []
    for q in P:
        out += ["PROMPT: " + q, "ANSWER: " + compose_logic(bot, q), ""]
    print("B64FX:" + base64.b64encode("\n".join(out).encode()).decode())
