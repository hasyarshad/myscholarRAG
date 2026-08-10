"""Language model backends.

Both adapters speak HTTP directly rather than through vendor SDKs. That keeps
the production image small, avoids a dependency whose only job is to build a
JSON body, and makes the two implementations easy to compare side by side.
"""
