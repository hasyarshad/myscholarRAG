"""Language model backends.

Both adapters call HTTP APIs directly with httpx instead of vendor SDKs, which keeps
the production image small.
"""
