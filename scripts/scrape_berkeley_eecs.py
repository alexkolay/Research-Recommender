"""
Scrapes the UC Berkeley EECS faculty list, producing a structured JSON file
of professors and their research interests for use by the recommender
agent.

Source: https://www2.eecs.berkeley.edu/Faculty/Lists/faculty.html

Note: eecs.berkeley.edu (the current site) is behind Cloudflare bot
protection and returns 403 to plain HTTP clients. The legacy www2
subdomain serves the same faculty list without that protection.

Berkeley's EECS department doesn't offer a clean CS-only sub-filter (unlike
MIT's explicit CS/EE role facet), so this scrapes the full EECS faculty
list -- consistent with treating EECS as Berkeley's CS-equivalent
department, the same way Purdue's full CS "Faculty & Lecturers" page was
scraped in its entirety.

Each entry is a single free-text <p> with <br>-separated lines and
<strong> labels (e.g. "Research Interests:", "Education:") rather than
clean structured fields -- parsed by splitting on <br> and matching label
prefixes.
"""

import json
import re
import sys

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www2.eecs.berkeley.edu"
LISTING_URL = f"{BASE_URL}/Faculty/Lists/faculty.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-recommender-bot/1.0)"}

OUTPUT_PATH = "data/berkeley_eecs_faculty.json"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


_LABELS = ("Research Interests:", "Education:", "Office Hours:")


def _label_content(lines: list[str], i: int, label: str) -> tuple[str, int]:
    """Returns (content, next_index). The label and its content are usually
    on the same line, but the source HTML sometimes inserts whitespace that
    splits them onto separate lines -- if content is empty, peek ahead."""
    content = lines[i][len(label):].strip()
    if not content and i + 1 < len(lines) and lines[i + 1] not in _LABELS:
        i += 1
        content = lines[i]
    return content, i


def parse_detail_paragraph(p) -> dict:
    for br in p.find_all("br"):
        br.replace_with("\n")
    lines = [line.strip() for line in p.get_text().split("\n") if line.strip()]

    result = {
        "title": None,
        "contact_line": None,
        "research_areas": [],
        "education": None,
        "office_hours": None,
    }
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Research Interests:"):
            content, i = _label_content(lines, i, "Research Interests:")
            result["research_areas"] = [a.strip() for a in content.split(";") if a.strip()]
        elif line.startswith("Education:"):
            content, i = _label_content(lines, i, "Education:")
            result["education"] = content
        elif line.startswith("Office Hours:"):
            content, i = _label_content(lines, i, "Office Hours:")
            result["office_hours"] = content
        elif "@" in line:
            result["contact_line"] = line
        elif result["title"] is None:
            result["title"] = line
        i += 1
    return result


def parse_item(item) -> dict | None:
    name_tag = item.find("h3")
    a = name_tag.find("a", href=True) if name_tag else None
    if not a:
        return None

    p = item.find("p")
    details = parse_detail_paragraph(p) if p else {}

    email = None
    office = None
    if details.get("contact_line"):
        m = EMAIL_RE.search(details["contact_line"])
        if m:
            email = m.group(0)
        office = details["contact_line"].split(",")[0].strip() if "," in details["contact_line"] else None

    return {
        "name": a.get_text(strip=True),
        "profile_url": BASE_URL + a["href"] if a["href"].startswith("/") else a["href"],
        "title": details.get("title"),
        "email": email,
        "office": office,
        "research_areas": details.get("research_areas") or [],
        "education": details.get("education"),
    }


def main():
    print(f"Fetching Berkeley EECS faculty list: {LISTING_URL}", file=sys.stderr)
    soup = fetch(LISTING_URL)

    faculty = []
    for item in soup.find_all("div", class_="cc-image-list__item"):
        entry = parse_item(item)
        if entry:
            faculty.append(entry)

    print(f"Found {len(faculty)} faculty entries", file=sys.stderr)

    result = {
        "school": "University of California, Berkeley",
        "department": "Electrical Engineering and Computer Sciences",
        "source_url": LISTING_URL,
        "faculty": faculty,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(faculty)} faculty records to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
