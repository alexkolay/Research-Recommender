"""
Scrapes the Purdue Computer Science faculty directory and individual
faculty profile pages, producing a structured JSON file of professors
and their research interests for use by the recommender agent.

Source: https://www.cs.purdue.edu/people/faculty/index.html
"""

import json
import time
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.cs.purdue.edu"
INDEX_URL = f"{BASE_URL}/people/faculty/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}
REQUEST_DELAY_SECONDS = 0.5

OUTPUT_PATH = "data/purdue_cs_faculty.json"


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_faculty_index(soup: BeautifulSoup) -> list[dict]:
    """Parse the directory index page into a list of faculty stubs."""
    faculty = []
    for item in soup.find_all("div", class_="people-item"):
        name_tag = item.select_one("p.people-name a")
        if not name_tag:
            continue
        title_tag = item.select_one("p.people-title")
        campus_tag = item.select_one("p.people-campus")
        area_tags = item.select("p.areas a")

        faculty.append(
            {
                "name": name_tag.get_text(strip=True),
                "profile_url": urljoin(BASE_URL, name_tag["href"]),
                "position": item.get("data-position", "").strip(),
                "title": title_tag.get_text(strip=True) if title_tag else None,
                "campus": campus_tag.get_text(strip=True) if campus_tag else None,
                "research_areas_index": [a.get_text(strip=True) for a in area_tags],
            }
        )
    return faculty


def parse_bio_paragraphs(info_div: BeautifulSoup) -> str:
    """Bio text sits between the <br id="fullbio"> marker and the next
    <h3> heading (Selected Publications), as free-floating <p> tags."""
    marker = info_div.find("br", id="fullbio")
    if not marker:
        return ""
    paragraphs = []
    for sibling in marker.find_next_siblings():
        if sibling.name == "h3":
            break
        if sibling.name == "p":
            paragraphs.append(sibling.get_text(" ", strip=True))
    return "\n\n".join(paragraphs)


def parse_profile(soup: BeautifulSoup) -> dict:
    data = {
        "bio": "",
        "research_areas": [],
        "education": [],
        "awards": [],
        "selected_publications": [],
        "email": None,
        "office": None,
        "websites": [],
    }

    info = soup.find("div", class_="info")
    if info is None:
        return data

    # Research areas (h2 -> ul > li > a)
    h2 = info.find("h2", string=lambda s: s and "Research Areas" in s)
    if h2:
        ul = h2.find_next_sibling("ul")
        if ul:
            data["research_areas"] = [a.get_text(strip=True) for a in ul.find_all("a")]

    # Education (h3 -> series of <p>)
    for h3 in info.find_all("h3"):
        label = h3.get_text(strip=True)
        if label == "Education":
            for sib in h3.find_next_siblings():
                if sib.name != "p":
                    break
                data["education"].append(sib.get_text(strip=True))
        elif label == "Awards":
            ul = h3.find_next_sibling("ul")
            if ul:
                for li in ul.find_all("li"):
                    text = " ".join(li.get_text(" ", strip=True).split())
                    data["awards"].append(text)
        elif label == "Selected Publications":
            # Each publication is a sibling <div> (no <ul>) containing a <p> citation.
            for sib in h3.find_next_siblings():
                if sib.name == "h3":
                    break
                if sib.name == "div":
                    text = " ".join(sib.get_text(" ", strip=True).split())
                    if text:
                        data["selected_publications"].append(text)

    data["bio"] = parse_bio_paragraphs(info)

    contact = soup.find("div", class_="contact")
    if contact:
        email_tag = contact.select_one("p.bio-email a")
        if email_tag:
            data["email"] = email_tag.get_text(strip=True)
        office_tag = contact.select_one("p.bio-office")
        if office_tag:
            data["office"] = office_tag.get_text(strip=True)
        for p in contact.find_all("p"):
            a = p.find("a")
            if a and p != email_tag and "mailto:" not in a.get("href", ""):
                data["websites"].append(
                    {"label": a.get_text(strip=True), "url": a["href"]}
                )

    return data


def main():
    print(f"Fetching faculty index: {INDEX_URL}", file=sys.stderr)
    index_soup = fetch(INDEX_URL)
    faculty = parse_faculty_index(index_soup)
    print(f"Found {len(faculty)} faculty entries", file=sys.stderr)

    for i, person in enumerate(faculty, start=1):
        print(f"[{i}/{len(faculty)}] {person['name']}", file=sys.stderr)
        try:
            profile_soup = fetch(person["profile_url"])
            person.update(parse_profile(profile_soup))
        except requests.RequestException as e:
            print(f"  WARNING: failed to fetch profile ({e})", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)

    result = {
        "school": "Purdue University",
        "department": "Computer Science",
        "source_url": INDEX_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
