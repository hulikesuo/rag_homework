"""Quick offline regression for the dense+bm25 RRF used by T3 optimized strategies."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments import t3_query_opt as T

    def mk(cid):
        return {"chunk_id": cid, "text": "x", "doc": "", "seq": 0}

    class Stub:
        def __init__(self, mp):
            self.mp = mp
        def retrieve(self, query, k, mode, **kw):
            return [(dict(chunk), score)
                    for chunk, score in self.mp.get((query, mode), [])]

    # y appears twice at rank 2, beating the single rank-1 hit.
    mp = {("qA", "dense"): [(mk("x"), .9)],
          ("qB", "dense"): [(mk("y"), .8)],
          ("qB", "bm25"): [(mk("y"), .7)]}
    def chunk_ids(rows):
        return [c["chunk_id"] for c in rows]
    out = T._retrieve(Stub(mp), ["qA", "qB"], True)
    assert chunk_ids(out) == ["y", "x"], out

    # Duplicate IDs inside one query/route are counted once.
    mp = {("q", "dense"): [(mk("x"), .9), (mk("x"), .8), (mk("z"), .7)],
          ("q", "bm25"): [(mk("x"), .6)]}
    out = T._retrieve(Stub(mp), ["q"], True)
    assert chunk_ids(out)[:2] == ["x", "z"], out

    # Boosted variant also uses RRF across repeated mentions.
    mp = {("q", "dense"): [(mk("x"), .9), (mk("y"), .8), (mk("z"), .7)],
          ("q", "bm25"): [(mk("y"), .6)]}
    out = T._retrieve_boosted(Stub(mp), ["q"], True)
    assert chunk_ids(out)[:2] == ["y", "x"], out
    print("retrieve_rrf probe: OK")
