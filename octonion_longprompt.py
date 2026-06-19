# octonion_longprompt.py -- consolidated bot on LONG, multi-part prompts: does it capture the
# different parts of a long prompt, route each correctly, and answer each? Shows segmentation.
import json
import octonion_transport as T
T.GEN_IDX = list(range(7))                               # fast transport for this routing demo
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full

PROMPTS = [
    "i've had a really bad headache and a fever for two days, and i'm also worried about my high blood pressure, plus i've been having trouble sleeping at night. what should i do?",
    "my elderly father has type 2 diabetes and his memory has been getting worse, and he also complains of joint pain in his knees. how can we help him?",
    "i'm always tired and i think i might be anemic, i also get frequent migraines, and my skin has been really dry lately. any advice?",
    "i have a sore throat and a cough, but i'm also stressed at work and it's causing chest pain, and i can't stop my hands from shaking.",
    "what causes kidney stones, how are they treated, and can i prevent them from coming back?",
]

if __name__ == "__main__":
    pool = load_full()                                   # patient pool (lighter; best for conversational)
    bot = OctonionPRBot().fit(pool, vocab_size=14000, verbose=True)
    out = []
    for q in PROMPTS:
        segs = bot._segments(q)
        out.append("PROMPT: " + q)
        out.append("  captured parts: " + " | ".join(" ".join(bot.M["vocab"][t] for t in ct[:2]) for _, ct in segs))
        out.append("  RESPONSE:")
        out.append(bot.respond(q))
        out.append("")
    open("lp_out.txt", "w").write("\n".join(out))
    print("WROTE lp_out.txt")
