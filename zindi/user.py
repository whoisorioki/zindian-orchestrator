class Zindian:
    """Lightweight stub of the Zindi client used for CI testing.
    Provides only the attributes and methods accessed by the repository.
    """

    def __init__(self, username: str | None = None, fixed_password: str | None = None):
        self.username = username
        self.password = fixed_password
        self._Zindian__auth_data = {"auth_token": "dummy-token"}
        self._Zindian__headers = {}
        # The real client tracks remaining submissions; we provide a dummy value.
        self.remaining_subimissions = 10  # note: the typo matches usage in codebase
        self.my_rank = 1

    def select_a_challenge(self, fixed_index: int):
        """Select a competition by index – no-op in stub."""
        print(f"[zindi stub] select_a_challenge called with index {fixed_index}")

    def submit(self, filepaths: list, comments: list):
        """Pretend to submit files – returns a dummy dict."""
        print(f"[zindi stub] submit called with {filepaths}, comments={comments}")
        return {"status": "submitted"}

    def leaderboard(self, to_print: bool = False):
        """Return a dummy leaderboard; optionally print it."""
        if to_print:
            print("[zindi stub] leaderboard: Rank 1 – Dummy User")
        return [{"rank": 1, "user": "Dummy User"}]

    def download_dataset(self, destination: str, make_destination: bool = True):
        """Pretend to download a dataset – returns an empty list of files."""
        print(f"[zindi stub] download_dataset to {destination}, make_destination={make_destination}")
        return []
