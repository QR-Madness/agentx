"""
Name pools — the deck behind the profile editor's random-name picker.

Two pools: RANDOM (the curated constant below — human names with the roster's
nature/bird undertone plus everyday warmth) and PREFERRED (user-starred names,
persisted as a cheap JSON array). Deals are without replacement and always
exclude names already worn by an existing profile, so the deck never offers a
duplicate; the exclusion happens at deal time, which means a starred name that
later gets used stays in PREFERRED but sits out until it frees up again.

Persistence mirrors the ProfileManager convention (``data/name_pools.json``;
JSON instead of YAML because the store is two flat string arrays), including
the module singleton.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 40

# The curated random pool. Flavor is deliberate: the user's roster runs on
# human names with a nature/bird undertone (Hazel, Ash, Reed, Wren, Lark...) —
# the deck deals more of that family plus everyday-warm classics. In-use names
# are filtered at deal time, so current roster names may appear here.
RANDOM_POOL: tuple[str, ...] = (
    # Nature & bird undertone
    "Alder", "Ash", "Aspen", "Birch", "Briar", "Brooke", "Bryn", "Cedar",
    "Clay", "Cliff", "Colt", "Coral", "Daisy", "Dale", "Dawn", "Dell",
    "Fern", "Finch", "Flint", "Gale", "Glen", "Hazel", "Heath", "Heather",
    "Holly", "Iris", "Ivy", "Jay", "June", "Juniper", "Lark", "Laurel",
    "Linden", "Misty", "Moss", "Oakley", "Olive", "Opal", "Pearl", "Petra",
    "Poppy", "Rain", "Raven", "Reed", "Ridge", "River", "Robin", "Rosa",
    "Rowan", "Sage", "Skye", "Sorrel", "Summer", "Sunny", "Teal", "Vale",
    "Wade", "Willow", "Winter", "Wren",
    # Everyday warmth
    "Ada", "Alma", "Amos", "Archie", "Bea", "Bess", "Cal", "Cass",
    "Celia", "Clara", "Cleo", "Cora", "Deb", "Dot", "Edie", "Eli",
    "Ella", "Elsie", "Etta", "Felix", "Flo", "Gemma", "Gil", "Goldie",
    "Greta", "Gus", "Hank", "Hattie", "Hugh", "Ida", "Ike", "Jo",
    "Josie", "Jude", "Lena", "Leo", "Lila", "Lou", "Lucy", "Mabel",
    "Mae", "Maeve", "Marty", "Mavis", "Max", "Mel", "Milo", "Minnie",
    "Nell", "Nina", "Nora", "Ollie", "Otis", "Otto", "Polly", "Ray",
    "Rex", "Rita", "Rosie", "Ruby", "Rufus", "Ruth", "Sadie", "Sal",
    "Scout", "Sid", "Stella", "Ted", "Tess", "Theo", "Tilly", "Toby",
    "Vera", "Vince", "Viv", "Walt", "Wes", "Winnie", "Zeke",
)

# First-load seed for PREFERRED: the unused half of the naming family the
# roster was built from. Deal-time exclusion keeps any that get used out of
# the deck, so the seed list can stay static.
PREFERRED_SEEDS: tuple[str, ...] = (
    "Robin", "Skye", "Finch", "June", "Rowan", "Sage", "Gale",
    "Flint", "Clay", "Ridge", "Moss", "Jay", "Bryn",
)


def _clean(name: str) -> str:
    return " ".join(name.split())


class NamePools:
    """Load/deal/star names; persists the starred pool to *path*."""

    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path(__file__).parent.parent.parent.parent / "data" / "name_pools.json"
        self.path = path
        self._data = self._load()

    def _load(self) -> dict[str, list[str]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {"preferred": list(PREFERRED_SEEDS), "custom_random": []}
            self._save(data)
            return data
        except (ValueError, OSError) as e:
            logger.warning(f"name_pools: unreadable {self.path} ({e}); using seeds in memory")
            return {"preferred": list(PREFERRED_SEEDS), "custom_random": []}
        return {
            "preferred": [_clean(n) for n in raw.get("preferred", []) if isinstance(n, str) and n.strip()],
            "custom_random": [_clean(n) for n in raw.get("custom_random", []) if isinstance(n, str) and n.strip()],
        }

    def _save(self, data: dict[str, list[str]] | None = None) -> None:
        payload = data if data is not None else self._data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @property
    def preferred(self) -> list[str]:
        return list(self._data["preferred"])

    def sample(self, pool: str = "random", count: int = 10, exclude: Iterable[str] = ()) -> list[str]:
        """Deal *count* names (clamped 1–20) from *pool*, never a name in
        *exclude* (case-insensitive). Short pools return everything, shuffled."""
        count = max(1, min(20, count))
        excluded = {_clean(n).casefold() for n in exclude if n and n.strip()}
        source: list[str] = (
            self._data["preferred"]
            if pool == "preferred"
            else [*RANDOM_POOL, *self._data["custom_random"]]
        )
        seen: set[str] = set()
        candidates: list[str] = []
        for name in source:
            key = name.casefold()
            if key in excluded or key in seen:
                continue
            seen.add(key)
            candidates.append(name)
        if count >= len(candidates):
            dealt = list(candidates)
            random.shuffle(dealt)
            return dealt
        return random.sample(candidates, count)

    def add_preferred(self, name: str) -> list[str]:
        name = _clean(name)
        if not name:
            raise ValueError("Name is empty.")
        if len(name) > MAX_NAME_LENGTH:
            raise ValueError(f"Name is longer than {MAX_NAME_LENGTH} characters.")
        if name.casefold() not in {n.casefold() for n in self._data["preferred"]}:
            self._data["preferred"].append(name)
            self._save()
        return self.preferred

    def remove_preferred(self, name: str) -> list[str]:
        key = _clean(name).casefold()
        kept = [n for n in self._data["preferred"] if n.casefold() != key]
        if len(kept) != len(self._data["preferred"]):
            self._data["preferred"] = kept
            self._save()
        return self.preferred


_instance: NamePools | None = None


def get_name_pools() -> NamePools:
    global _instance
    if _instance is None:
        _instance = NamePools()
    return _instance
