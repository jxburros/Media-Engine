"""Bounded, load-scoped text-encoding reuse. No prompt or tensor is persisted."""
import hashlib
import json
from collections import OrderedDict


class TextCache:
    def __init__(self, max_bytes=32 * 1024 * 1024, max_entries=64):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def get(self, key, compute):
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).digest()
        found = self.entries.pop(digest, None)
        if found is not None:
            self.entries[digest] = found
            return self._clone(found[0])
        result = compute()
        size = self._size(result)
        if size <= self.max_bytes:
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                _, (_, old_size) = self.entries.popitem(last=False)
                self.bytes -= old_size
            self.entries[digest] = (self._clone(result), size)
            self.bytes += size
        return result

    @classmethod
    def _size(cls, item):
        if item is None:
            return 0
        if isinstance(item, (list, tuple)):
            return sum(cls._size(x) for x in item)
        return item.numel() * item.element_size()

    @classmethod
    def _clone(cls, item):
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            return type(item)(cls._clone(x) for x in item)
        return item.detach().clone()
