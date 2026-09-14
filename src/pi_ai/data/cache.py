class TensorCache:
    """Opt-in in-memory cache; disabled by default to protect Pi RAM."""
    def __init__(self) -> None:
        self._items = {}

    def get(self, key):
        return self._items.get(key)

    def put(self, key, value) -> None:
        self._items[key] = value
