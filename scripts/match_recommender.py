"""
Matches a structured resume (produced by resume_extract.py) against scraped
faculty research data (one or more schools) and program info (e.g. Purdue's
SURF), and produces a ranked list of professor recommendations with grounded
rationale.

By default, faculty and program files are auto-discovered from the data/
directory (data/*_faculty.json and data/*_program.json), so adding a new
school's scraped data automatically brings it into future matching runs.

Output includes both a cumulative top-N ranking across all schools and a
top-K ranking within each individual school. research_areas on every
recommendation is filled in from the scraped faculty data itself (not left
to the model to copy), so it's always accurate even when the model's prose
rationale doesn't repeat it verbatim.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python3 scripts/match_recommender.py data/alex_kolay_resume.extracted.json \
        [--faculty data/purdue_cs_faculty.json data/stanford_cs_faculty.json ...] \
        [--programs data/purdue_surf_program.json ...] \
        [--top-n 10] \
        [--per-school-top-n 5] \
        [-o output.json]
"""

import argparse
import difflib
import json
import re
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5"
DATA_DIR = Path("data")

MATCH_INSTRUCTIONS = """
You are a research-advisor matching assistant for undergraduate students
considering research opportunities at several universities. You will be
given:

1. A student's structured resume data (education, research/work experience,
   projects, skills, and a synthesized research_interest_summary + keywords).
2. A list of faculty across one or more universities, each with their school,
   research areas, and a bio describing their research.
3. Known existing relationships: cases where the resume's stated
   advisor/mentor was automatically matched (by name) to someone in the
   faculty list below, prior to your analysis.
4. General information about relevant research programs (e.g. Purdue's SURF).

Your job has two parts:
1. "cumulative_recommendations": the top {top_n} faculty members overall this
   student should consider reaching out to or applying to work with, ranked
   best-fit first, across ALL the universities represented in the faculty
   list (not just one).
2. "by_school_recommendations": for EACH school represented in the faculty
   list, the top {per_school_top_n} faculty members at that school, ranked
   best-fit first within that school. Include every school that appears in
   the faculty list, even if none of its faculty made the cumulative list.

The same grounding and ranking standards apply to both parts -- a professor
can appear in both the cumulative list and their school's list.

Grounding requirements (do not skip these):
- Every rationale must cite SPECIFIC evidence from the student's resume
  (a named project, course, research experience, or skill) AND specific
  language from the professor's own bio/research_areas -- not generic
  statements like "shares an interest in AI."
- Only recommend faculty who actually appear in the provided faculty list.
  Copy their "profile_url" and "school" exactly as given -- never invent or
  guess a URL or school name.
- Prefer faculty whose research substantively overlaps with the student's
  demonstrated experience over faculty who only share a broad, popular
  keyword (e.g. "machine learning") with no deeper connection.
- If a faculty member appears in the "KNOWN EXISTING RELATIONSHIPS" list
  below, and you choose to recommend them, you MUST explicitly say in the
  rationale that the student already has a research relationship with this
  person (don't frame it as a cold intro), and tailor suggested_next_steps
  accordingly (e.g. "follow up on your prior work with them" rather than
  "reach out to introduce yourself").
- In "suggested_next_steps", be concrete: mention whether applying via a
  program like SURF makes sense for this pairing (only for schools that
  have such a program in the provided info), whether emailing the professor
  directly based on a specific shared interest makes sense, etc. Don't be
  generic ("reach out to discuss opportunities").

Return ONLY a single JSON object matching the required schema. Do not invent
information not present in the provided data.

STUDENT RESUME DATA:
---
{resume_json}
---

KNOWN EXISTING RELATIONSHIPS (detected by name-matching the resume's stated
advisors/mentors against the faculty list; empty array means none detected):
---
{existing_relationships_json}
---

FACULTY LIST (JSON array, each with name/school/title/campus/research_areas/bio/profile_url):
---
{faculty_json}
---

PROGRAM INFO (e.g. SURF -- may not apply to every school in the faculty list above):
---
{programs_json}
---
"""

RECOMMENDATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "rank": {"type": "integer"},
        "professor_name": {"type": "string"},
        "school": {"type": "string"},
        "profile_url": {"type": "string"},
        "research_areas": {"type": "array", "items": {"type": "string"}},
        "is_existing_relationship": {"type": "boolean"},
        "rationale": {
            "type": "string",
            "description": "Grounded explanation citing specific resume evidence and specific professor bio/research language.",
        },
        "suggested_next_steps": {
            "type": "string",
            "description": "Concrete action: program application, direct email with specific talking point, etc.",
        },
    },
    "required": [
        "rank",
        "professor_name",
        "school",
        "profile_url",
        "research_areas",
        "is_existing_relationship",
        "rationale",
        "suggested_next_steps",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_summary": {
            "type": "string",
            "description": "2-3 sentence synthesis of the student's research profile, for context.",
        },
        "cumulative_recommendations": {
            "type": "array",
            "description": "Top-N faculty overall, ranked across all schools.",
            "items": RECOMMENDATION_ITEM_SCHEMA,
        },
        "by_school_recommendations": {
            "type": "array",
            "description": "One entry per school, each with its own top-K ranked faculty.",
            "items": {
                "type": "object",
                "properties": {
                    "school": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": RECOMMENDATION_ITEM_SCHEMA,
                    },
                },
                "required": ["school", "recommendations"],
                "additionalProperties": False,
            },
        },
        "program_applicability_note": {
            "type": "string",
            "description": "General note on whether/how known programs (e.g. SURF) fit this student's situation.",
        },
    },
    "required": [
        "candidate_summary",
        "cumulative_recommendations",
        "by_school_recommendations",
        "program_applicability_note",
    ],
    "additionalProperties": False,
}

_TITLE_PREFIX_RE = re.compile(r"^(professor|prof\.?|dr\.?)\s+", re.IGNORECASE)
_NAME_SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(name: str) -> str:
    return _TITLE_PREFIX_RE.sub("", name).strip().lower()


def _last_name_token(name: str) -> str:
    tokens = [t for t in name.replace(",", " ").split() if t]
    while tokens and tokens[-1].strip(".").lower() in _NAME_SUFFIXES:
        tokens.pop()
    if not tokens:
        return ""
    return tokens[-1].strip(".,").lower()


def find_existing_relationships(
    resume_data: dict, faculty_context: list[dict], cutoff: float = 0.75
) -> list[dict]:
    """Fuzzy-match advisor/mentor names mentioned in the resume against the
    faculty list by last name. This exists because upstream resume
    extraction can introduce small misspellings (e.g. "Aref" -> "Areff"),
    which would otherwise cause the model to treat an existing research
    relationship as a fresh recommendation instead of flagging it
    explicitly."""
    faculty_by_last_name: dict[str, list[dict]] = {}
    for f in faculty_context:
        last = _last_name_token(f["name"])
        if last:
            faculty_by_last_name.setdefault(last, []).append(f)

    matches = []
    for exp in resume_data.get("research_experience", []):
        mentor = exp.get("advisor_or_mentor")
        if not mentor:
            continue
        mentor_last = _last_name_token(_normalize_name(mentor))
        if not mentor_last:
            continue
        close = difflib.get_close_matches(
            mentor_last, faculty_by_last_name.keys(), n=1, cutoff=cutoff
        )
        if not close:
            continue
        for faculty in faculty_by_last_name[close[0]]:
            matches.append(
                {
                    "resume_text": mentor,
                    "matched_faculty_name": faculty["name"],
                    "matched_faculty_school": faculty.get("school"),
                    "matched_faculty_profile_url": faculty["profile_url"],
                    "experience_title": exp.get("title"),
                }
            )
    return matches


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def discover_files(pattern: str) -> list[Path]:
    return sorted(DATA_DIR.glob(pattern))


def build_faculty_context(faculty_data: dict) -> list[dict]:
    """Trim one school's faculty records to matching-relevant fields
    (tagging each with its school), and drop faculty with no research
    signal at all (nothing to match on)."""
    school = faculty_data.get("school")
    trimmed = []
    for f in faculty_data["faculty"]:
        has_signal = bool(f.get("bio")) or bool(f.get("research_areas"))
        if not has_signal or not f.get("profile_url"):
            continue
        trimmed.append(
            {
                "name": f["name"],
                "school": school,
                "title": f.get("title"),
                "campus": f.get("campus"),
                "research_areas": f.get("research_areas") or f.get("research_areas_index"),
                "bio": f.get("bio"),
                "profile_url": f["profile_url"],
            }
        )
    return trimmed


def load_faculty_context(faculty_paths: list[Path]) -> list[dict]:
    context = []
    for path in faculty_paths:
        context.extend(build_faculty_context(load_json(path)))
    return context


def build_prompt(
    resume_data: dict,
    faculty_context: list[dict],
    existing_relationships: list[dict],
    programs: list[dict],
    top_n: int,
    per_school_top_n: int,
) -> str:
    return MATCH_INSTRUCTIONS.format(
        top_n=top_n,
        per_school_top_n=per_school_top_n,
        resume_json=json.dumps(resume_data, indent=2, ensure_ascii=False),
        existing_relationships_json=json.dumps(existing_relationships, indent=2, ensure_ascii=False),
        faculty_json=json.dumps(faculty_context, indent=2, ensure_ascii=False),
        programs_json=json.dumps(programs, indent=2, ensure_ascii=False),
    )


def call_claude_for_matches(prompt: str) -> dict:
    client = anthropic.Anthropic()
    # Streamed: at max_tokens=32000 the request may run long enough that the
    # SDK refuses a non-streaming call outright (10-minute timeout floor).
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Response was truncated at max_tokens before completing -- "
            "increase max_tokens in call_claude_for_matches()."
        )
    text_block = next(b for b in response.content if b.type == "text")
    return json.loads(text_block.text)


def backfill_research_areas(result: dict, faculty_context: list[dict]) -> None:
    """Overwrite each recommendation's research_areas with the ground-truth
    value from the scraped faculty data (keyed by profile_url), so listed
    areas are always accurate even if the model's copy of them was
    incomplete or omitted."""
    areas_by_url = {f["profile_url"]: f.get("research_areas") or [] for f in faculty_context}

    def fix(rec: dict) -> None:
        if rec["profile_url"] in areas_by_url:
            rec["research_areas"] = areas_by_url[rec["profile_url"]]

    for rec in result.get("cumulative_recommendations", []):
        fix(rec)
    for group in result.get("by_school_recommendations", []):
        for rec in group.get("recommendations", []):
            fix(rec)


def match_resume(
    resume_path: Path,
    faculty_paths: list[Path],
    program_paths: list[Path],
    top_n: int,
    per_school_top_n: int,
) -> dict:
    resume_data = load_json(resume_path)
    faculty_context = load_faculty_context(faculty_paths)
    programs = [load_json(p) for p in program_paths]

    existing_relationships = find_existing_relationships(resume_data, faculty_context)
    prompt = build_prompt(
        resume_data, faculty_context, existing_relationships, programs, top_n, per_school_top_n
    )
    result = call_claude_for_matches(prompt)
    backfill_research_areas(result, faculty_context)
    result["_source_resume"] = str(resume_path)
    result["_faculty_sources"] = [str(p) for p in faculty_paths]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume_path", type=Path, help="Path to a resume_extract.py output JSON file")
    parser.add_argument(
        "--faculty", type=Path, nargs="+", default=None,
        help="Path(s) to faculty research-interest JSON files (default: auto-discover data/*_faculty.json)",
    )
    parser.add_argument(
        "--programs", type=Path, nargs="+", default=None,
        help="Path(s) to program info JSON files (default: auto-discover data/*_program.json)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of cumulative (cross-school) recommendations to return")
    parser.add_argument("--per-school-top-n", type=int, default=5, help="Number of recommendations to return per individual school")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: <resume_name>.matches.json next to input)",
    )
    args = parser.parse_args()

    faculty_paths = args.faculty or discover_files("*_faculty.json")
    program_paths = args.programs or discover_files("*_program.json")

    if not faculty_paths:
        raise SystemExit("No faculty JSON files found or specified (looked for data/*_faculty.json)")

    output_path = args.output or args.resume_path.with_suffix(".matches.json")

    result = match_resume(
        args.resume_path, faculty_paths, program_paths, args.top_n, args.per_school_top_n
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    schools = sorted({g["school"] for g in result["by_school_recommendations"]})
    print(
        f"Wrote {len(result['cumulative_recommendations'])} cumulative + "
        f"{sum(len(g['recommendations']) for g in result['by_school_recommendations'])} per-school "
        f"recommendations to {output_path} (schools: {', '.join(schools)})"
    )


if __name__ == "__main__":
    main()
