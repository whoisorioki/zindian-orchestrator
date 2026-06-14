class MockConnection:
    def __init__(self, *args, **kwargs):
        pass
    def execute(self, sql, params=None):
        return self
    def fetchone(self):
        return None
    def fetchall(self):
        return []
    def commit(self):
        pass
    @property
    def description(self):
        return []

def connect(path=None):
    return MockConnection()
