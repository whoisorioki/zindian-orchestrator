from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import validate_challenge_config
from .paths import resolve_competition_paths


class ConfigNotPopulated(RuntimeError):
    """Raised when a required config field is null."""

    pass


@dataclass
class ChallengeConfig:
    """Reader for challenge_config.json with null guards and field validation."""

    path: Path
    _data: Dict[str, Any]

    # Fields that MUST NOT be null
    REQUIRED_FIELDS = frozenset(
        [
            "metric",
            "metric_direction",
            "use_probabilities",
            "automl_permitted",
            "data_modality",
        ]
    )

    @classmethod
    def load(cls, path: str | None = None) -> "ChallengeConfig":
        """Load and validate challenge_config.json."""
        if path is None:
            path_obj = resolve_competition_paths().config_path
        else:
            path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"challenge_config.json not found at {path_obj}")

        data = json.loads(path_obj.read_text(encoding="utf-8"))

        # Validate schema
        validate_challenge_config(data)

        # Check for null values in required fields
        for field in cls.REQUIRED_FIELDS:
            if data.get(field) is None:
                raise ConfigNotPopulated(
                    f"Required field '{field}' is null in challenge_config.json. "
                    f"Skill 02 (intake) must populate this from Zindi API."
                )

        return cls(path=path_obj, _data=data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._data.get(key, default)

    def get_required(self, key: str) -> Any:
        """Get config value, raise ConfigNotPopulated if null."""
        value = self._data.get(key)
        if value is None:
            raise ConfigNotPopulated(
                f"Required field '{key}' is null in challenge_config.json"
            )
        return value

    def __getitem__(self, key: str) -> Any:
        """Dict-like access."""
        return self._data[key]

    def __repr__(self) -> str:
        return f"ChallengeConfig(competition={self.get('slug')}, metric={self.get('metric')})"

    # Convenience properties for common fields
    @property
    def metric(self) -> str:
        return self.get_required("metric")

    @property
    def metric_direction(self) -> str:
        return self.get_required("metric_direction")

    @property
    def use_probabilities(self) -> bool:
        return self.get_required("use_probabilities")

    @property
    def automl_permitted(self) -> bool:
        return self.get_required("automl_permitted")

    @property
    def data_modality(self) -> str:
        return self.get_required("data_modality")

    @property
    def domain(self) -> Optional[str]:
        """Domain of competition (e.g., 'solar', 'finance', 'generic')."""
        return self.get("domain")

    @property
    def allowed_external_data(self) -> bool:
        return self.get("allowed_external_data", False)

    @property
    def daily_limit(self) -> Optional[int]:
        return self.get("daily_limit")

    @property
    def slug(self) -> str:
        return self.get("slug", "unknown")


def get_seed(default: int = 42) -> int:
    """Convenience helper to read the reproducibility seed from the active challenge config.

    Returns the integer seed found at `challenge_config.json` -> `reproducibility.seed`,
    or `default` if absent.
    """
    try:
        cfg = ChallengeConfig.load()
        return int(cfg.get("reproducibility", {}).get("seed", default))
    except Exception:
        return default
