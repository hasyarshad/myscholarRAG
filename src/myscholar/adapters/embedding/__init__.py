"""Embedding backends.

Two adapters, one on GPU for ingestion and one on CPU for serving, and they must
agree exactly — the index is built by the first and searched by the second.

The query instruction lives here rather than in either of them because it is the
one thing they must not decide separately. It was defined in the torch adapter
alone, with a comment about keeping the asymmetry inside one class; when the
ONNX adapter arrived it simply did not have it, so every production query was
embedded differently from the passages it was searching. Nothing failed. Cosine
parity between the two models was a perfect 1.00000, because the models *were*
identical — the disagreement was in what each one was asked to embed. Retrieval
just quietly lost 0.07 MRR, which looked exactly like quantisation damage and
was very nearly attributed to it.
"""

#: BGE models were trained with an instruction prefix on queries but not on
#: passages. Omitting it costs a few points of retrieval accuracy; applying it
#: to passages as well is worse.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
