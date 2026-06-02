# octonion_demo.py -- qualitative demo of the full augmented bot (patient + mined pool).
import json, base64
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_qa.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=24000, verbose=True)
    qs = ["i have a headache and a fever, what should i do?",
          "what causes leg cramps at night?",
          "i can't sleep at night, what can help?",
          "is it safe to exercise when i have a cold?",
          "can stress cause chest pain?",
          "what are the symptoms of dehydration?",
          "how is high blood pressure treated?",
          "what are the symptoms of a heart attack?",
          "what is type 2 diabetes?",
          "what is misoprostol?",
          "what is metformin used for?",
          "what is gout?"]
    out = []
    for q in qs:
        ans, match = bot.answer(q, with_match=True)
        out += ["Q: " + q, "A: " + ans[:300], "   (nearest stored Q: " + (match or "-")[:60] + ")", ""]
    print("B64DEMO:" + base64.b64encode("\n".join(out).encode()).decode())
