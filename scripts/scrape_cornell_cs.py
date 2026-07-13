"""
Scrapes the Cornell Bowers CIS people directory, filtered to Computer
Science departmental faculty, producing a structured JSON file of
professors and their research interests for use by the recommender agent.

Source: https://www.cs.cornell.edu/directory?department=15&roles[147]=147
  (department=15 -> Computer Science, roles[147]=147 -> "Faculty (Department)")

Each directory card already contains name, title, email, research areas,
bio, location, office, and a personal homepage link -- no secondary
per-profile fetch is needed.
"""

import json
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.cs.cornell.edu"
LISTING_URL = f"{BASE_URL}/directory?department=15&roles%5B147%5D=147"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}
REQUEST_DELAY_SECONDS = 0.5

OUTPUT_PATH = "data/cornell_cs_faculty.json"


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_card(card) -> dict | None:
    name_div = card.find("div", class_="name")
    if not name_div:
        return None
    # Most faculty have a linked name (-> /people/<slug>); visiting
    # scientists/scholars without a full profile page just have a <span>.
    a = name_div.find("a", href=True)
    name = (a or name_div).get_text(strip=True)
    profile_url = (BASE_URL + a["href"] if a["href"].startswith("/") else a["href"]) if a else None

    title_div = card.find("div", class_="position-titles")
    title = title_div.get_text(" ", strip=True) if title_div else None

    email_tag = card.find("div", class_="email")
    email = None
    if email_tag:
        link = email_tag.find("a", href=lambda h: h and h.startswith("mailto:"))
        if link:
            email = link["href"].replace("mailto:", "").strip()

    research_areas = []
    for subgroup in card.find_all("div", class_="card-content-group-subgroup"):
        label = subgroup.find("div", class_="label")
        if label and label.get_text(strip=True) == "Research Areas":
            content = subgroup.find("div", class_="item-content")
            if content:
                research_areas = [
                    r.strip() for r in content.get_text(strip=True).split(";") if r.strip()
                ]

    bio_div = card.find("div", class_="is-bio")
    bio = bio_div.get_text(" ", strip=True) if bio_div else None

    location = None
    office = None
    for item in card.find_all("div", class_="item item--detail hidden"):
        label = item.find("div", class_="label")
        content = item.find("div", class_="item-content")
        if not label or not content:
            continue
        label_text = label.get_text(strip=True)
        if label_text == "Location":
            location = content.get_text(strip=True)
        elif label_text == "Office":
            office = content.get_text(strip=True)

    website = None
    website_div = card.find("div", class_="website")
    if website_div:
        link = website_div.find("a", href=True)
        if link:
            website = link["href"]

    return {
        "name": name,
        "profile_url": profile_url,
        "title": title,
        "email": email,
        "research_areas": research_areas,
        "bio": bio,
        "location": location,
        "office": office,
        "website": website,
    }


def fetch_all_faculty() -> list[dict]:
    faculty = []
    page = 0
    while True:
        url = f"{LISTING_URL}&page={page}"
        print(f"Fetching directory page {page}: {url}", file=sys.stderr)
        soup = fetch(url)
        cards = soup.find_all("div", class_="directory-card")
        if not cards:
            break
        for card in cards:
            entry = parse_card(card)
            if entry:
                faculty.append(entry)
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return faculty


def main():
    faculty = fetch_all_faculty()
    print(f"Found {len(faculty)} faculty entries", file=sys.stderr)

    result = {
        "school": "Cornell University",
        "department": "Computer Science",
        "source_url": LISTING_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
