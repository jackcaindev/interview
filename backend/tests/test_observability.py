from app.observability import langsmith_enabled


class FakeSettings:
    langsmith_project = "test-project"

    def __init__(self, tracing: str, api_key: str) -> None:
        self.langsmith_tracing = tracing
        self.langsmith_api_key = api_key


def test_langsmith_requires_tracing_flag_and_api_key() -> None:
    assert not langsmith_enabled(FakeSettings("false", "lsv2_key"))
    assert not langsmith_enabled(FakeSettings("true", ""))
    assert langsmith_enabled(FakeSettings("true", "lsv2_key"))
