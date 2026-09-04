"""Tests for the command line entry point.

Mostly about wiring and failure messages: the commands themselves are thin, but a
mis-wired subcommand or a silent failure on a missing input is exactly what makes
a pipeline unreproducible.
"""

from __future__ import annotations

import pytest

from iie.cli import _seasons, build_parser, main


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024", ("2024",)),
        ("2024,2025", ("2024", "2025")),
        ("2024, 2025 ", ("2024", "2025")),
        ("", ()),
    ],
)
def test_season_lists_parse(text, expected):
    assert _seasons(text) == expected


def test_every_stage_is_reachable():
    """Crawl, build and evaluate are separable so a model change never re-crawls."""
    parser = build_parser()
    for command in ("crawl", "claims", "evaluate", "status"):
        args = parser.parse_args([command])
        assert args.command == command
        assert callable(args.func)


def test_crawl_defaults_are_conservative():
    args = build_parser().parse_args(["crawl"])
    assert args.target == 500
    assert args.rate == 12.0
    assert args.seasons == "2024,2025"


def test_crawl_can_resume_without_a_seed():
    """An existing frontier holds queued leagues; re-seeding is not required."""
    assert build_parser().parse_args(["crawl"]).seed is None


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_evaluate_explains_a_missing_claim_table(tmp_path, caplog):
    code = main(["--data", str(tmp_path), "evaluate"])
    assert code == 1
    assert "run `iie claims` first" in caplog.text


def test_status_works_on_an_empty_data_root(tmp_path, capsys):
    """Status must describe an empty project rather than crashing on it."""
    assert main(["--data", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "frontier: not created" in out
    assert "not built" in out


def test_status_reports_a_populated_corpus(tmp_path, capsys):
    from iie.crawl.store import CorpusStore

    with CorpusStore(tmp_path / "corpus") as store:
        store.add_league(
            {
                "league_id": "L1",
                "season": "2025",
                "sport": "nfl",
                "status": "complete",
                "total_rosters": 12,
                "draft_id": "D1",
                "roster_positions": ["WR"],
                "scoring_settings": {},
                "settings": {},
            }
        )

    assert main(["--data", str(tmp_path), "status"]) == 0
    assert "leagues                       1 rows" in capsys.readouterr().out
