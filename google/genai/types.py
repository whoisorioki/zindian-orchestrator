class HttpOptions:
    """Stub HttpOptions for google.genai used in CI testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GenerateContentConfig:
    """Stub GenerateContentConfig for google.genai used in CI testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
