"""HTTP interface.

Streams progress rather than tokens, because verification happens after
generation and unverified text must never reach a visitor.
"""
