# octonion_qa_clean.py -- same prompts as octonion_qa_show but on the CLEAN pool (patient +
# 15.5k clean mined, no noisy 80k research dump) to show reduced jargon intrusion.
import json, base64
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_composer import compose_logic
from octonion_qa_show import PROMPTS

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_qa.json"))]   # clean 15.5k
    bot = OctonionPRBot().fit(pool, vocab_size=24000, verbose=True)
    out = []
    for q in PROMPTS:
        out += ["PROMPT: " + q, "ANSWER: " + compose_logic(bot, q), ""]
    print("B64QC:" + base64.b64encode("\n".join(out).encode()).decode())
