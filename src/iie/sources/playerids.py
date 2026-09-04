"""Resolving Sleeper player IDs to nflverse ``gsis_id``.

The two corpora do not share a key, and the obvious bridge does not work.
Measured against the live Sleeper player set, on active fantasy-relevant players:

* Sleeper's own ``gsis_id`` field is populated on only ~20%.
* Sleeper has ``sportradar_id`` on ~99%, but nflverse's players table does not
  carry that column, so it cannot be joined on.
* DynastyProcess publishes a crosswalk holding ``sleeper_id`` and ``gsis_id`` in
  one row, which lifts coverage to ~92%.

The residual matters more than its size suggests. Almost all of it is players
with zero or one years of experience -- which is precisely the waiver-wire
population this engine exists to price. An unmatched rookie is not a rounding
error here; it is the breakout the engine was supposed to catch. So there is a
third tier: a normalized name-and-position match, applied only when it is
unambiguous, because a wrong join silently attributes one player's production to
another and that is worse than a null.

Every resolution records which tier produced it. ``coverage()`` reports the split
so a thin match rate shows up as a number rather than as quietly bad features.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

DYNASTYPROCESS_IDS_URL = "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv"

# The crosswalk is updated as rookies are added; a day old is plenty fresh.
ID_MAP_TTL = 24 * 3600

# Generational suffixes are written inconsistently across sources and never
# disambiguate anyone in practice, so they are stripped before matching.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_NON_ALPHA = re.compile(r"[^a-z ]+")


def normalize_name(name: str | None) -> str | None:
    """Fold a display name to a comparison key: lowercase, no punctuation, no suffix.

    ``"Marvin Harrison Jr."`` and ``"marvin harrison"`` must land on the same key,
    since Sleeper and nflverse disagree about suffixes constantly.
    """
    if not name:
        return None
    cleaned = _NON_ALPHA.sub("", str(name).lower().replace(".", " ").replace("-", " "))
    parts = [p for p in cleaned.split() if p and p not in _SUFFIXES]
    return "".join(parts) or None


def _clean_sleeper_id(value: Any) -> str | None:
    """The crosswalk is CSV, so a numeric sleeper_id arrives as ``"6918.0"``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text.removesuffix(".0")


def fetch_id_map(cache_dir: Path | str, *, session: requests.Session | None = None) -> pd.DataFrame:
    """Download (or reuse) the DynastyProcess crosswalk."""
    target = Path(cache_dir) / "db_playerids.csv"
    fresh = target.exists() and time.time() - target.stat().st_mtime < ID_MAP_TTL

    if not fresh:
        target.parent.mkdir(parents=True, exist_ok=True)
        resp = (session or requests).get(DYNASTYPROCESS_IDS_URL, timeout=60)
        resp.raise_for_status()
        tmp = target.with_suffix(".csv.tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        log.info("fetched player id crosswalk (%.1f MB)", target.stat().st_size / 1e6)

    return pd.read_csv(target, low_memory=False)


def build_bridge(client, id_cache_dir: Path | str, nflverse_cache) -> PlayerIdBridge:
    """Assemble the bridge from its three sources.

    All three are cached, so this is cheap to call repeatedly: Sleeper's player
    list for a day, the crosswalk for a day, and the nflverse players table until
    its TTL expires. Wiring them by hand is how the pipeline ended up depending on
    a file someone had fetched manually.
    """
    return PlayerIdBridge(
        client.get_players(),
        id_map=fetch_id_map(id_cache_dir, session=client.cache.session),
        nflverse_players=nflverse_cache.load("players"),
    )


@dataclass(frozen=True)
class Resolution:
    """A resolved player, tagged with the tier that resolved it."""

    gsis_id: str
    method: str  # "sleeper" | "crosswalk" | "name"


class PlayerIdBridge:
    """Maps Sleeper player IDs to nflverse ``gsis_id``.

    Build once per run and reuse; construction is pure dictionary work over
    already-fetched frames, but the frames themselves are large.
    """

    def __init__(
        self,
        sleeper_players: dict[str, dict[str, Any]],
        id_map: pd.DataFrame | None = None,
        nflverse_players: pd.DataFrame | None = None,
    ) -> None:
        self._sleeper = sleeper_players
        self._crosswalk = self._build_crosswalk(id_map)
        self._by_name = self._build_name_index(nflverse_players)
        self._resolved: dict[str, Resolution] = {}
        self._unresolved: set[str] = set()

    @staticmethod
    def _build_crosswalk(id_map: pd.DataFrame | None) -> dict[str, str]:
        if id_map is None or "sleeper_id" not in id_map or "gsis_id" not in id_map:
            return {}
        out: dict[str, str] = {}
        for sleeper_id, gsis_id in zip(id_map["sleeper_id"], id_map["gsis_id"], strict=False):
            key = _clean_sleeper_id(sleeper_id)
            if key and isinstance(gsis_id, str) and gsis_id.strip():
                out[key] = gsis_id.strip()
        return out

    @staticmethod
    def _build_name_index(nflverse_players: pd.DataFrame | None) -> dict[tuple[str, str], str]:
        """Index ``(normalized_name, position) -> gsis_id``, dropping collisions.

        A name shared by two players at the same position cannot be resolved this
        way, so it is removed entirely rather than resolved to whichever row came
        first. Silence is the correct output for an ambiguous match.
        """
        if nflverse_players is None or nflverse_players.empty:
            return {}

        buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
        for _, row in nflverse_players.iterrows():
            gsis_id = row.get("gsis_id")
            position = row.get("position")
            key_name = normalize_name(row.get("display_name"))
            if not (
                isinstance(gsis_id, str) and gsis_id and key_name and isinstance(position, str)
            ):
                continue
            buckets[(key_name, position)].add(gsis_id)

        collisions = sum(1 for ids in buckets.values() if len(ids) > 1)
        if collisions:
            log.info("name index: dropped %d ambiguous name/position keys", collisions)
        return {key: next(iter(ids)) for key, ids in buckets.items() if len(ids) == 1}

    def is_defense(self, sleeper_id: str) -> bool:
        """Whether a Sleeper id names a team defense rather than a person.

        Sleeper uses the team abbreviation itself as the id for a defense ("LAC",
        "GB"), so these have no counterpart in any player table and never resolve.
        Distinguishing them from genuine resolution failures is what keeps a whole
        unpriceable position from being mistaken for a broken bridge.
        """
        player = self._sleeper.get(str(sleeper_id)) or {}
        if player.get("position") == "DEF":
            return True
        return "DEF" in (player.get("fantasy_positions") or [])

    def resolve(self, sleeper_id: str) -> Resolution | None:
        """Resolve one Sleeper player ID, trying each tier in confidence order."""
        sleeper_id = str(sleeper_id)
        if sleeper_id in self._resolved:
            return self._resolved[sleeper_id]
        if sleeper_id in self._unresolved:
            return None

        player = self._sleeper.get(sleeper_id) or {}

        direct = player.get("gsis_id")
        if isinstance(direct, str) and direct.strip():
            return self._remember(sleeper_id, direct.strip(), "sleeper")

        mapped = self._crosswalk.get(sleeper_id)
        if mapped:
            return self._remember(sleeper_id, mapped, "crosswalk")

        key = (normalize_name(player.get("full_name")), player.get("position"))
        if key[0] and key[1]:
            by_name = self._by_name.get(key)
            if by_name:
                return self._remember(sleeper_id, by_name, "name")

        self._unresolved.add(sleeper_id)
        return None

    def _remember(self, sleeper_id: str, gsis_id: str, method: str) -> Resolution:
        resolution = Resolution(gsis_id, method)
        self._resolved[sleeper_id] = resolution
        return resolution

    def resolve_many(self, sleeper_ids: Iterable[str]) -> dict[str, str]:
        """Resolve a batch, omitting the ones that do not resolve."""
        out = {}
        for sleeper_id in sleeper_ids:
            hit = self.resolve(sleeper_id)
            if hit:
                out[str(sleeper_id)] = hit.gsis_id
        return out

    def coverage(self, sleeper_ids: Iterable[str] | None = None) -> dict[str, int]:
        """Resolution counts by tier, so a degraded match rate is visible.

        Pass the IDs you actually care about -- coverage over every player Sleeper
        has ever listed is a flattering and useless number, because it is
        dominated by retired players nobody can claim.
        """
        if sleeper_ids is not None:
            for sleeper_id in sleeper_ids:
                self.resolve(sleeper_id)

        counts = {"sleeper": 0, "crosswalk": 0, "name": 0}
        for resolution in self._resolved.values():
            counts[resolution.method] += 1
        counts["resolved"] = len(self._resolved)
        counts["unresolved"] = len(self._unresolved)
        return counts
