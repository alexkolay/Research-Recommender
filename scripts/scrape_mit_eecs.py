"""
Scrapes the MIT EECS faculty directory (filtered to the "CS" role facet),
producing a structured JSON file of professors and their research interests
for use by the recommender agent.

Source: https://www.eecs.mit.edu/people/?fwp_role=faculty-cs

Unlike Purdue/Stanford, MIT's listing page itself contains everything
needed (name, title, contact info, and research-area tags) -- there is no
free-text bio on individual profile pages, so no secondary fetch is needed.
"""

import json
import sys

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.eecs.mit.edu"
LISTING_URL = f"{BASE_URL}/people/?fwp_role=faculty-cs"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}

OUTPUT_PATH = "data/mit_eecs_faculty.json"


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_entry(card) -> dict:
    name_tag = card.find("h5")
    a = name_tag.find("a", href=True) if name_tag else None
    if not a:
        return None

    title_tag = card.find("p")
    title = title_tag.get_text(strip=True) if title_tag else None

    email = None
    office = None
    phone = None
    contact_list = card.find("ul")
    if contact_list:
        for li in contact_list.find_all("li"):
            email_link = li.find("a", href=lambda h: h and h.startswith("mailto:"))
            if email_link:
                email = email_link["href"].replace("mailto:", "").strip()
                continue
            text = li.get_text(strip=True)
            if text.startswith("Office:"):
                office = text.replace("Office:", "").strip()
            elif not email and any(c.isdigit() for c in text):
                phone = text

    research_areas = []
    research_div = card.find("div", class_="people-research")
    if research_div:
        research_areas = [a_tag.get_text(strip=True) for a_tag in research_div.find_all("a")]

    return {
        "name": a.get_text(strip=True),
        "profile_url": a["href"],
        "title": title,
        "email": email,
        "phone": phone,
        "office": office,
        "research_areas": research_areas,
    }


def main():
    print(f"Fetching MIT EECS CS-role faculty listing: {LISTING_URL}", file=sys.stderr)
    soup = fetch(LISTING_URL)

    faculty = []
    for card in soup.find_all("div", class_="people-entry"):
        entry = parse_entry(card)
        if entry:
            faculty.append(entry)

    print(f"Found {len(faculty)} faculty entries", file=sys.stderr)

    result = {
        "school": "Massachusetts Institute of Technology",
        "department": "Electrical Engineering and Computer Science (CS-role faculty)",
        "source_url": LISTING_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
