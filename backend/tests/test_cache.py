from app.cache import FallbackJsonCache, InMemoryJsonCache


class FailingJsonCache:
    def get_json(self, key: str):
        return None

    def set_json(self, key: str, value: dict, *, ttl_seconds: int) -> None:
        return None


def test_fallback_cache_uses_memory_when_primary_is_unavailable() -> None:
    cache = FallbackJsonCache(FailingJsonCache(), InMemoryJsonCache())

    cache.set_json("key", {"answer": "cached"}, ttl_seconds=60)

    assert cache.get_json("key") == {"answer": "cached"}
