"""Diagnostic probe for Understat response status, season, and anti-bot pages."""

import argparse
import asyncio

import aiohttp

from understat_client import UNDERSTAT_HEADERS


async def probe(team: str, season: str) -> None:
    url = f"https://understat.com/getTeamData/{team}/{season}"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(headers=UNDERSTAT_HEADERS, timeout=timeout) as session:
        async with session.get(url, allow_redirects=False) as response:
            body = await response.text(errors="replace")
            print(f"url={url}")
            print(f"status={response.status}")
            print(f"content_type={response.headers.get('Content-Type', '')}")
            print(f"location={response.headers.get('Location', '')}")
            print(f"body={body[:300]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Understat getTeamData endpoint")
    parser.add_argument("team", nargs="?", default="Liverpool")
    parser.add_argument("season", nargs="?", default="2026")
    args = parser.parse_args()
    asyncio.run(probe(args.team, args.season))
