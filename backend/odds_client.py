import requests
from datetime import datetime, timezone
from typing import Any

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# Sportsbooks to display (h2h = moneyline)
BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]


class OddsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict[str, Any] = {}

    def get_raw(self) -> dict:
        url = f"{ODDS_API_BASE}/sports/{SPORT}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        self._cache = {"data": resp.json(), "fetched_at": datetime.now(timezone.utc).isoformat()}
        return self._cache

    def get_todays_mlb_games(self) -> list[dict]:
        raw = self.get_raw()
        events = raw.get("data", raw) if isinstance(raw, dict) and "data" in raw else raw
        if isinstance(events, dict):
            events = events.get("data", [])

        today = datetime.now(timezone.utc).date()
        games = []

        for event in events:
            game_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            if game_time.date() != today:
                continue

            odds_by_book = _parse_bookmaker_odds(event.get("bookmakers", []))
            games.append({
                "id": event["id"],
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event["commence_time"],
                "odds": odds_by_book,
                "best_odds": _best_odds(odds_by_book),
            })

        return games


def _parse_bookmaker_odds(bookmakers: list) -> dict:
    """Return {bookmaker_title: {home: int, away: int}} for h2h markets."""
    result = {}
    for bm in bookmakers:
        key = bm["key"]
        if key not in BOOKMAKERS:
            continue
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in market["outcomes"]}
            result[bm["title"]] = prices
    return result


def _best_odds(odds_by_book: dict) -> dict:
    """Find the best (highest) moneyline for each team across all books."""
    best: dict[str, int] = {}
    for book_odds in odds_by_book.values():
        for team, price in book_odds.items():
            if team not in best or price > best[team]:
                best[team] = price
    return best
