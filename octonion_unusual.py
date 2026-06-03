# octonion_unusual.py -- big pool (patient + 80k mined ~116k) with all current improvements
# (fano-gram on, entity-weighted match, rerank) answering MANY unusual / out-of-distribution
# prompts. Each answer shows a match-confidence (the bot's honesty signal).
import json, base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full

UNUSUAL = [
    "can i take ibuprofen with alcohol?",
    "why do i feel dizzy when i stand up quickly?",
    "what helps with a hangover?",
    "can anxiety cause stomach problems?",
    "why do i get a headache when i drink coffee?",
    "is intermittent fasting safe?",
    "can lack of sleep make you gain weight?",
    "how do i know if a mole is dangerous?",
    "is it normal to feel tired all the time?",
    "can stress cause hair loss?",
    "what should i eat before a workout?",
    "is it bad to crack my knuckles?",
    "why are my hands always cold?",
    "can i drink coffee while pregnant?",
    "what causes bad breath in the morning?",
    "is it safe to hold in a sneeze?",
    "why do i bruise so easily?",
    "can spicy food cause ulcers?",
    "how much water should i drink a day?",
    "why do my ears ring after a concert?",
    "is cracking my back dangerous?",
    "can dehydration cause headaches?",
    "what makes you yawn?",
    "is it bad to read in the dark?",
]

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_big.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=30000, verbose=True)
    out = []
    for q in UNUSUAL:
        pe, nn = bot._match(q, k=1, lex=0.6, w_fano=3.0); conf = float(bot.EPru[int(nn[0])] @ pe)
        ans, match = bot.answer(q, with_match=True)
        out += ["Q: " + q + ("   [conf=%.2f]" % conf),
                "A: " + ans[:240],
                "   (nearest: " + (match or "-")[:55] + ")", ""]
    print("B64U:" + base64.b64encode("\n".join(out).encode()).decode())
