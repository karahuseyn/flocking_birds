# octonion_qa_show.py -- clean, FULL prompt -> answer pairs (composition mode, no truncation).
import json, base64
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_composer import compose_logic

PROMPTS = [
    "what is hypertension and how is it treated?",
    "what causes a stroke?",
    "what is asthma?",
    "how is pneumonia treated?",
    "what causes kidney stones?",
    "what is osteoporosis?",
    "what is a migraine?",
    "what is type 2 diabetes?",
]

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_big.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=30000, verbose=True)
    out = []
    for q in PROMPTS:
        ans = compose_logic(bot, q)
        out += ["PROMPT: " + q, "ANSWER: " + ans, ""]
    print("B64QA:" + base64.b64encode("\n".join(out).encode()).decode())
