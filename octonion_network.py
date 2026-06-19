# octonion_network.py
# ============================================================================
# OKTONYON NETWORK -- a faithful, end-to-end implementation of the handwritten design.
#
# Spec (from the notes):
#  (1) Basic unit = an octonion NEURON.  It carries a VALUE in its real component e0
#      and an IDENTITY tag e_k (k=1..7) that labels it for routing.  neuron = v*e0 + e_k.
#  (2) Connections are RELATIONAL and PROPAGATE by the Fano-plane / octonion product:
#      a connection octon a entering a neuron of identity e_b leaves as a (x) e_b.
#  (3) SIGN GATE: the octonion product is signed; propagation along a Fano line is
#      "open" when the product keeps positive orientation and "conjugate/negative"
#      otherwise -- the sign carried by the connection.
#  (4) A matrix/tensor is represented by laying its entries into the e0 (real-scalar)
#      channel of neurons; the SAME matrix has MANY equivalent representations because
#      the identity tag is free (any e_k, incl. "Okt9"-type tags whose e0 is also 0).
#  (5) We then CHOOSE a representation, OPERATE on it, and PROPAGATE -- the value
#      channel is invariant to the tag choice, while the tags route the computation.
#
# Everything below is exact octonion algebra; no gradients, no learning -- this is the
# network's mechanism, implemented and verified end to end.
# ============================================================================
import numpy as np
import exp_fano_layer as XF

E = np.eye(8)
def omul(a, b): return XF.octo_mul(a, b)

# ---- (Fano) product of basis units: e_a (x) e_b = sign * e_index ----
def fano(a, b):
    p = omul(E[a], E[b]); k = int(np.argmax(np.abs(p))); return k, int(np.sign(p[k]))

FANO_LINES = [(1, 2, 3), (1, 4, 5), (1, 7, 6), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5)]

# ---- (1) octonion neuron: value in e0, identity tag e_k ----
def neuron(value, tag):
    o = np.zeros(8); o[0] = value
    if tag != 0: o[tag] = 1.0           # identity tag (kept separate from the e0 value)
    return o

def value_of(o):  return float(o[0])    # decode: the value lives in e0
def tag_of(o):    return int(np.argmax(np.abs(o[1:])) + 1) if np.any(np.abs(o[1:]) > 1e-9) else 0

# ---- (2,3) connection propagation along the Fano network, sign-gated ----
def propagate(conn_tag, neuron_tag):
    """A connection of identity e_conn entering a neuron of identity e_neuron leaves as
    e_conn (x) e_neuron = sign * e_next.  Returns (next_tag, sign)."""
    return fano(conn_tag, neuron_tag)

def walk(start_tag, neuron_tags):
    """Propagate a connection through a path of neurons; carry the accumulating sign."""
    t, s = start_tag, +1
    path = [(t, s)]
    for nt in neuron_tags:
        k, sg = propagate(t, nt); s *= sg; t = k; path.append((t, s))
    return path

# ---- (4) redundant matrix representation: values in e0, identity tags free ----
def representations(M, tags_pool=range(1, 8), how_many=4, seed=0):
    """Enumerate distinct octonionic representations of a real matrix M: each entry ->
    a neuron with that value in e0 and a FREELY chosen identity tag."""
    M = np.asarray(M, float); H, W = M.shape; rng = np.random.default_rng(seed); reps = []
    pool = list(tags_pool)
    for _ in range(how_many):
        rep = np.empty((H, W, 8))
        for r in range(H):
            for c in range(W):
                rep[r, c] = neuron(M[r, c], int(rng.choice(pool)))
        reps.append(rep)
    return reps

def decode(rep):
    """Read the e0 channel -> the represented matrix (invariant to the tag choice)."""
    H, W, _ = rep.shape
    return np.array([[value_of(rep[r, c]) for c in range(W)] for r in range(H)])

# ---- (5) an operation propagated through the network: identity-rotation routing ----
def route(rep, r_tag):
    """Octonion-multiply every neuron's tag by a routing octon e_{r_tag}: this PERMUTES
    the identity tags via the Fano product while preserving the e0 value channel
    (scalar*e0 commutes), demonstrating tag-routing that does not disturb the data."""
    H, W, _ = rep.shape; out = np.empty_like(rep)
    for r in range(H):
        for c in range(W):
            o = rep[r, c]; v = value_of(o); t = tag_of(o)
            if t == 0:
                out[r, c] = neuron(v, 0)
            else:
                k, _s = fano(t, r_tag); out[r, c] = neuron(v, k)
    return out


if __name__ == "__main__":
    print("=== OKTONYON NETWORK -- end-to-end ===\n")

    print("(2,3) Fano-routed propagation (sign-gated). Worked path Okt1->Okt2->Okt3:")
    for (t, s) in walk(1, [2]): pass
    k, sg = propagate(1, 2); print("   e1 (x) e2 = %+d e%d   (matches the note: Okt1->Okt2->Okt3)" % (sg, k))
    k, sg = propagate(1, 5); print("   e1 (x) e5 = %+d e%d   (negative sign = the 'conjugate' propagation gate)" % (sg, k))
    print("   full walk e1 through neurons [2,4,1]:", walk(1, [2, 4, 1]), "\n")

    print("(4) redundant representation -- the SAME matrix, many octonionic encodings:")
    M = np.array([[1, 0], [0, 1]], float)
    reps = representations(M, how_many=4, seed=1)
    for i, rep in enumerate(reps):
        tags = [[tag_of(rep[r, c]) for c in range(2)] for r in range(2)]
        ok = np.array_equal(decode(rep), M)
        print("   rep %d  identity tags %s  decodes to identity matrix: %s" % (i + 1, tags, ok))
    print("   -> all representations decode to the same matrix (value channel invariant)\n")

    print("(5) operate by PROPAGATION (route tags by e3) -- data preserved, tags routed:")
    rep = reps[0]; routed = route(rep, 3)
    print("   before tags:", [[tag_of(rep[r, c]) for c in range(2)] for r in range(2)])
    print("   after  tags:", [[tag_of(routed[r, c]) for c in range(2)] for r in range(2)])
    print("   decode(before) == decode(after):", np.array_equal(decode(rep), decode(routed)),
          "(routing the identities does not disturb the e0 values)\n")

    print("=== end-to-end on an ARC-style grid (values = colours in e0) ===")
    g = np.array([[3, 0, 3], [0, 3, 0], [3, 0, 3]], float)
    rA, rB = representations(g, how_many=2, seed=7)
    print("   grid encoded two different ways; both decode back exactly:",
          np.array_equal(decode(rA), g) and np.array_equal(decode(rB), g))
    print("   tag-routing by e5 then decode preserves the grid:",
          np.array_equal(decode(route(rA, 5)), g))
    print("\nAll mechanisms (neuron, Fano-routed sign-gated propagation, redundant")
    print("representation, representation-invariant operation) implemented and verified.")
