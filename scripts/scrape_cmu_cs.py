"""
Scrapes the Carnegie Mellon Computer Science Department faculty directory
and individual profile pages, producing a structured JSON file of
professors and their research interests for use by the recommender agent.

Source: https://www.csd.cs.cmu.edu/people/faculty (paginated Drupal table,
?page=0..5)
"""

import json
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.csd.cs.cmu.edu"
LISTING_URL = f"{BASE_URL}/people/faculty"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}
REQUEST_DELAY_SECONDS = 0.5

OUTPUT_PATH = "data/cmu_cs_faculty.json"


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_row(tr) -> dict | None:
    name_cell = tr.find("td", class_="views-field-field-first-name")
    a = name_cell.find("a", href=True) if name_cell else None
    if not a:
        return None

    for br in name_cell.find_all("br"):
        br.replace_with("\n")
    lines = [line.strip() for line in name_cell.get_text().split("\n") if line.strip()]
    # lines[0] is "Last, First" (the name); remaining lines are title/office/phone/email
    title = lines[1] if len(lines) > 1 else None

    email = None
    email_link = name_cell.find("a", href=lambda h: h and h.startswith("mailto:"))
    if email_link:
        email = email_link["href"].replace("mailto:", "").strip()

    name_parts = lines[0].split(",", 1)
    name = f"{name_parts[1].strip()} {name_parts[0].strip()}" if len(name_parts) == 2 else lines[0]

    return {
        "name": name,
        "profile_url": BASE_URL + a["href"],
        "title": title,
        "email": email,
    }


def fetch_all_listing_rows() -> list[dict]:
    rows = []
    page = 0
    while True:
        url = f"{LISTING_URL}?page={page}"
        print(f"Fetching listing page {page}: {url}", file=sys.stderr)
        soup = fetch(url)
        page_rows = [
            parse_listing_row(tr) for tr in soup.select("table.cols-2 tbody tr")
        ]
        page_rows = [r for r in page_rows if r]
        if not page_rows:
            break
        rows.extend(page_rows)
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return rows


def _section_tags(soup: BeautifulSoup, heading_text: str) -> list[str]:
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True) == heading_text:
            container = h.find_next_sibling()
            if container:
                return [p.get_text(strip=True) for p in container.find_all("p") if p.get_text(strip=True)]
    return []


def parse_profile(soup: BeautifulSoup) -> dict:
    data = {"research_areas": [], "bio": ""}

    areas = _section_tags(soup, "Research Areas")
    interests = _section_tags(soup, "Research Interests")
    # Dedupe while preserving order
    combined = list(dict.fromkeys(areas + interests))
    data["research_areas"] = combined

    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True) == "Research Statement":
            paragraphs = []
            for sib in h.find_next_siblings():
                if sib.name in ("h2", "h3"):
                    break
                if sib.name == "p":
                    paragraphs.append(sib.get_text(" ", strip=True))
            data["bio"] = "\n\n".join(paragraphs)
            break

    return data


def main():
    stubs = fetch_all_listing_rows()
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
        "school": "Carnegie Mellon University",
        "department": "Computer Science",
        "source_url": LISTING_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
