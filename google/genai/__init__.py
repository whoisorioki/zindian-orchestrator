class Client:
    """Stub Client for google.genai used in CI testing."""
    def __init__(self, api_key=None, http_options=None):
        self.api_key = api_key
        self.http_options = http_options
        self.models = Models()

class Models:
    """Stub Models for google.genai used in CI testing."""
    def generate_content(self, model, contents, config=None):
        class DummyResponse:
            def __init__(self):
                self.text = "[]"
                self.parsed = []
        return DummyResponse()
