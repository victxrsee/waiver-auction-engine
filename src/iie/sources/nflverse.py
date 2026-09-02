"""nflverse ingestion: the opportunity side of the value model.

The Sleeper corpus tells us what leagues *paid* for a player. It cannot tell us
what the player was actually worth, because Sleeper exposes no snap counts, no
target share, and no depth charts. That half comes from nflverse, which
publishes season-partitioned Parquet as GitHub release assets.

Fetching is deliberately not routed through ``iie.sources.cache``. That cache is
built for many small JSON responses keyed by URL hash; these are a handful of
multi-megabyte binary files with meaningful names, and a crawl wants them on disk
under those names so they can be inspected and hand-deleted. The shared
discipline that does carry over is TTL semantics: a completed season is immutable
and cached forever, while the in-progress season is refetched on a timer because
nflverse republishes it as games are played.

Season coverage differs per dataset and is declared below, so asking for a season
that was never published fails immediately with a clear message rather than as a
404 halfway through a crawl.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

# nflverse republishes the live season as games finish. Six hours is well inside
# their update cadence and keeps a same-day re-crawl off the network.
CURRENT_SEASON_TTL = 6 * 3600


@dataclass(frozen=True)
class Dataset:
    """One nflverse release. ``stem`` + season names the asset within ``tag``."""

    tag: str
    stem: str
    first_season: int | None  # None => not season-partitioned

    @property
    def seasonal(self) -> bool:
        return self.first_season is not None

    def asset(self, season: int | None) -> str:
        return f"{self.stem}_{season}.parquet" if self.seasonal else f"{self.stem}.parquet"

    def url(self, season: int | None) -> str:
        return f"{RELEASE_URL}/{self.tag}/{self.asset(season)}"


# First seasons are the real published floors, checked against the release
# listings rather than assumed. snap_counts starting at 2012 is the binding
# constraint on any model that uses opportunity share.
DATASETS: dict[str, Dataset] = {
    "snap_counts": Dataset("snap_counts", "snap_counts", 2012),
    "injuries": Dataset("injuries", "injuries", 2009),
    "depth_charts": Dataset("depth_charts", "depth_charts", 2001),
    "player_week": Dataset("stats_player", "stats_player_week", 1999),
    "roster_weekly": Dataset("weekly_rosters", "roster_weekly", 2002),
    "players": Dataset("players", "players", None),
    # Not season-partitioned: one file covers every season.
    "schedules": Dataset("schedules", "games", None),
}


def current_season(today: dt.date | None = None) -> int:
    """NFL season label for a date.

    A season is named for the calendar year it kicks off in, so January and
    February still belong to the previous season. March is the league-year
    boundary and the point at which nflverse starts publishing the new season.
    """
    # UTC rather than local: the boundary is a whole month wide, so tz slop is
    # irrelevant, and a fixed clock keeps the label reproducible across machines.
    today = today or dt.datetime.now(dt.UTC).date()
    return today.year if today.month >= 3 else today.year - 1


class NflverseCache:
    """Downloads and caches nflverse release assets, returning DataFrames."""

    def __init__(
        self,
        root: Path | str,
        *,
        session: requests.Session | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    def _resolve(self, name: str, season: int | None) -> tuple[Dataset, int | None]:
        try:
            dataset = DATASETS[name]
        except KeyError:
            raise KeyError(
                f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}"
            ) from None

        if not dataset.seasonal:
            return dataset, None
        if season is None:
            raise ValueError(f"dataset {name!r} is season-partitioned; pass a season")
        if season < dataset.first_season:
            raise ValueError(
                f"nflverse publishes {name!r} from {dataset.first_season}; asked for {season}"
            )
        return dataset, season

    def path(self, name: str, season: int | None = None) -> Path:
        dataset, season = self._resolve(name, season)
        return self.root / dataset.tag / dataset.asset(season)

    def fetch(self, name: str, season: int | None = None) -> Path:
        """Return a local path to the asset, downloading only if stale or absent."""
        dataset, season = self._resolve(name, season)
        target = self.root / dataset.tag / dataset.asset(season)

        if target.exists() and not self._is_stale(target, season):
            return target

        url = dataset.url(season)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        backoff = 2.0

        for attempt in range(self.max_retries):
            try:
                with self.session.get(url, timeout=self.timeout, stream=True) as resp:
                    if resp.status_code == 404:
                        raise FileNotFoundError(f"nflverse has no asset at {url}")
                    resp.raise_for_status()
                    with tmp.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            except requests.RequestException as exc:
                log.warning("nflverse fetch failed (%s), retry %d: %s", exc, attempt + 1, url)
                tmp.unlink(missing_ok=True)
                time.sleep(backoff)
                backoff *= 2
                continue
            # Renamed only after the body is fully written, so an interrupted
            # download can never be mistaken for a complete cached asset.
            tmp.replace(target)
            log.info("fetched %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
            return target

        raise RuntimeError(f"giving up after {self.max_retries} attempts: {url}")

    def _is_stale(self, target: Path, season: int | None) -> bool:
        """Completed seasons never go stale; the live season expires on a timer."""
        if season is None or season >= current_season():
            return time.time() - target.stat().st_mtime > CURRENT_SEASON_TTL
        return False

    def load(
        self, name: str, season: int | None = None, columns: list[str] | None = None
    ) -> pd.DataFrame:
        return pd.read_parquet(self.fetch(name, season), columns=columns)

    def load_seasons(
        self, name: str, seasons: list[int], columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Concatenate several seasons, skipping any nflverse has not published.

        A partial result beats an exception here: the live season routinely lacks
        assets that every prior season has, and a crawl should still run.
        """
        frames = []
        for season in seasons:
            try:
                frames.append(self.load(name, season, columns=columns))
            except (FileNotFoundError, ValueError) as exc:
                log.warning("skipping %s %s: %s", name, season, exc)
        if not frames:
            return pd.DataFrame(columns=columns or [])
        return pd.concat(frames, ignore_index=True)
