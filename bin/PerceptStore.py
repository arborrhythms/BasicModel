"""Backwards-compat re-export shim.

The canonical class is :class:`RadixLayer` in :mod:`bin.Layers`. This
module preserves the old ``PerceptStore`` name (and the auxiliary
``RadixTrie`` / ``BytesFallbackEncoder`` exports) so that existing
callers and tests that still write ``from PerceptStore import
PerceptStore`` keep working unchanged.

New code should import :class:`RadixLayer` directly from
:mod:`bin.Layers`.
"""

from Layers import RadixLayer as PerceptStore
from Layers import RadixLayer, RadixTrie, BytesFallbackEncoder, _RadixNode

__all__ = [
    "PerceptStore",
    "RadixLayer",
    "RadixTrie",
    "BytesFallbackEncoder",
    "_RadixNode",
]
