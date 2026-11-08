"""Embedding backends: torch on a local GPU, and ONNX Runtime on CPU in production.

An index built by one is searched by the other, so the query instruction is defined
once here for both.
"""

#: BGE models were trained with an instruction prefix on queries but not on
#: passages. Omitting it from queries cost 0.07 MRR on this corpus, with no error;
#: applying it to passages as well is worse.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
