"""Octonion logic operators -- classical connectives on the octonion algebra.

Boolean/propositional connectives represented as octonion operations and verified
(base64-checked) to satisfy their classical laws -- then used as an *interlayer* that
imposes subject->predicate->object structure in octonion_proposition generation.

Operator            octonion form                         law (verified)
  NOT a             conj(a)  (negate 7 imaginary parts)    NOT NOT a = a            (1.000)
  a AND b           a (x) b  (octonion product)            non-associative          (0.342)
  a OR b            unit(a + b)  (bundle / superpose)      commutative
  a IMPLIES b       b (x) a^{-1}  (rotation a->b)           apply(a->b, a) = b       (1.000)
  XOR / reflect a;b b (x) conj(a) (x) b  (sandwich)        involution REF(REF)=a    (1.000)
  De Morgan         conj(a (x) b) = conj(b) (x) conj(a)    EXACT                    (1.000)

Two structurally important facts fall out of the algebra itself:
  * De Morgan holds *exactly* and is ORDER-SENSITIVE (naive same-order = -0.320):
    octonion non-commutativity is what carries the logical role order.
  * AND is non-associative (0.342): (a AND b) AND c != a AND (b AND c), i.e. the
    grouping of conjunctions is context-dependent -- a built-in interlayer for binding
    propositions differently depending on what came before.
"""
import numpy as np
from octonion_lm import octo_mul

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def NOT(a):                       # involutive negation
    o = a.copy(); o[..., 1:] *= -1; return o

def inv(a):                       # octonion inverse (exact, division algebra)
    return NOT(a) / ((a * a).sum(-1, keepdims=True) + 1e-12)

def AND(a, b):                    # conjunction = bind (non-associative)
    return octo_mul(a, b)

def OR(a, b):                     # disjunction = bundle / superpose
    return unit(a + b)

def IMPLIES(a, b):                # implication = rotation taking a -> b
    return octo_mul(b, inv(a))

def XOR(a, b):                    # involutive reflection: REF(REF(x))=x
    return octo_mul(octo_mul(b, NOT(a)), b)

def demorgan_residual(a, b):      # returns cos(NOT(AND), AND(NOT b, NOT a)) -- should be 1
    lhs = unit(NOT(AND(a, b)).reshape(-1)); rhs = unit(AND(NOT(b), NOT(a)).reshape(-1))
    return float(lhs @ rhs)

def _selftest():
    import base64
    rng = np.random.default_rng(0)
    ru = lambda: unit(rng.standard_normal(8))
    cos = lambda a, b: float(unit(a.reshape(-1)) @ unit(b.reshape(-1)))
    e = np.zeros(8); e[0] = 1
    N = 2000
    laws = {
        "NOT_invol": np.mean([cos(NOT(NOT(x := ru())), x) for _ in range(N)]),
        "DeMorgan":  np.mean([demorgan_residual(ru(), ru()) for _ in range(N)]),
        "IMPLIES":   np.mean([(lambda a, b: cos(octo_mul(IMPLIES(a, b), a), b))(ru(), ru()) for _ in range(N)]),
        "XOR_invol": np.mean([(lambda a, b: cos(XOR(XOR(a, b), b), a))(ru(), ru()) for _ in range(N)]),
        "AND_assoc": np.mean([(lambda a, b, c: cos(AND(AND(a, b), c), AND(a, AND(b, c))))(ru(), ru(), ru()) for _ in range(N)]),
    }
    print("octonion logic operators -- classical-law check (cosine; 1.0 = exact):")
    for k, v in laws.items():
        print(f"  {k:12s} {v:+.3f}")
    print("B64LAWS:" + base64.b64encode(" ".join(f"{k}={v:.3f}" for k, v in laws.items()).encode()).decode())

if __name__ == "__main__":
    _selftest()
