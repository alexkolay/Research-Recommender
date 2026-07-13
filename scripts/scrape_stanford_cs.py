"""
Scrapes the Stanford Computer Science faculty directory and individual
faculty profile pages, producing a structured JSON file of professors and
their research interests for use by the recommender agent.

Source: https://www.cs.stanford.edu/people/faculty (paginated, Drupal
"Load More" style: ?page=0, ?page=1, ... until a page returns no cards)
"""

import json
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.cs.stanford.edu"
LISTING_URL = f"{BASE_URL}/people/faculty"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}
REQUEST_DELAY_SECONDS = 0.5

OUTPUT_PATH = "data/stanford_cs_faculty.json"


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    faculty = []
    for card in soup.find_all("article", class_="su-card--minimal"):
        name_tag = card.select_one(".su-person-short-title, h3 a") or card.find("a", href=True)
        link_tag = card.find("h3")
        a = link_tag.find("a", href=True) if link_tag else None
        if not a:
            continue
        short_title_tag = card.find("div", class_="su-person-short-title")
        faculty.append(
            {
                "name": a.get_text(strip=True),
                "profile_url": BASE_URL + a["href"],
                "short_title": short_title_tag.get_text(strip=True) if short_title_tag else None,
            }
        )
    return faculty


def fetch_all_faculty_stubs() -> list[dict]:
    all_faculty = []
    page = 0
    while True:
        url = f"{LISTING_URL}?page={page}"
        print(f"Fetching listing page {page}: {url}", file=sys.stderr)
        soup = fetch(url)
        page_faculty = parse_listing_page(soup)
        if not page_faculty:
            break
        all_faculty.extend(page_faculty)
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return all_faculty


def parse_profile(soup: BeautifulSoup) -> dict:
    data = {"title": None, "bio": None, "email": None}

    full_title = soup.find("div", class_="su-person-full-title")
    if full_title:
        data["title"] = full_title.get_text(strip=True)

    body = soup.find("div", class_="su-person-body")
    if body:
        data["bio"] = body.get_text(" ", strip=True)

    email_link = soup.find("a", href=lambda h: h and h.startswith("mailto:"))
    if email_link:
        data["email"] = email_link["href"].replace("mailto:", "").strip()

    return data


def main():
    stubs = fetch_all_faculty_stubs()
    print(f"Found {len(stubs)} faculty entries", file=sys.stderr)

    faculty = []
    for i, stub in enumerate(stubs, start=1):
        print(f"[{i}/{len(stubs)}] {stub['name']}", file=sys.stderr)
        person = dict(stub)
        try:
            profile_soup = fetch(stub["profile_url"])
            person.update(parse_profile(profile_soup))
        except requests.RequestException as e:
            print(f"  WARNING: failed to fetch profile ({e})", file=sys.stderr)
        faculty.append(person)
        time.sleep(REQUEST_DELAY_SECONDS)

    result = {
        "school": "Stanford University",
        "department": "Computer Science",
        "source_url": LISTING_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
