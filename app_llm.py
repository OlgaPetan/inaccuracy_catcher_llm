import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
import os

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets.")
    st.stop()

OPENAI_MODEL = "gpt-5.2"


# =========================
# AMENITIES (PRIORITY LISTS)
# =========================

HIGH_INTENT_AMENITIES = {
    "Comfort & Wellness": [
        "Hot tub", "Sauna", "Gym", "Exercise equipment",
        "Fireplace", "Indoor fireplace", "Fire pit",
    ],
    "Convenience & Functionality": [
        "Dedicated workspace", "EV charger",
        "Free parking on premises", "Free street parking",
        "Paid parking on premises", "Paid parking off premises",
    ],
    "Leisure & Outdoor": [
        "Pool", "Beach access", "Lake access", "Waterfront",
        "Patio or balcony", "BBQ grill", "Outdoor kitchen",
        "Outdoor dining area", "Resort access", "Ski-in/Ski-out",
    ],
    "Entertainment & Experience": [
        "Pool table", "Ping pong table", "Arcade games", "Game console",
        "Board games", "Movie theater", "Theme room", "Mini golf",
        "Climbing wall", "Bowling alley", "Laser tag", "Hockey rink",
        "Boat slip", "Skate ramp", "Life size games", "Bikes", "Kayak",
    ],
    "Scenic Views": [
        "Ocean view", "Sea view", "Beach view", "Lake view", "River view",
        "Bay view", "Mountain view", "Valley view", "City skyline view",
        "City view", "Garden view", "Pool view", "Park view", "Courtyard view",
        "Resort view", "Vineyard view", "Desert view", "Water view",
    ],
}

LOW_PRIORITY_AMENITIES = {
    "Secondary/basic": [
        "Room-darkening shades", "Body soap", "Shampoo", "Conditioner", "Shower gel",
        "Dishes and silverware", "Baking sheet", "Barbecue utensils", "Blender",
        "Bread maker", "Coffee", "Cooking basics", "Dining table", "Wine glasses",
        "Trash compactor", "Ethernet connection", "Pocket wifi", "Outlet covers",
        "Table corner guards", "Window guards", "Bidet", "Bathtub",
        "Single level home", "Cleaning available during stay", "Kitchenette",
        "Laundromat nearby", "Carbon monoxide alarm", "Smoke alarm",
        "Fire extinguisher", "First aid kit", "Ceiling fan", "Portable fans",
        "Extra pillows and blankets", "Hangers", "Mosquito net", "Bed linens",
        "Drying rack for clothing", "Clothing storage", "Cleaning products",
        "Air conditioning", "Dryer", "Essentials", "Heating", "Hot water",
        "Kitchen", "TV", "Washer", "Wifi", "Oven", "Microwave", "Stove",
        "Refrigerator", "Freezer", "Mini fridge", "Rice maker", "Toaster",
        "Dishwasher", "Coffee maker", "Private entrance", "Luggage dropoff allowed",
        "Long term stays allowed", "Hair dryer", "Iron", "Safe", "Crib",
        "High chair", "Children’s books and toys", "Baby bath", "Baby monitor",
        "Baby safety gates", "Changing table", "Pack ’n play / Travel crib",
        "Babysitter recommendations", "Children’s dinnerware",
    ]
}


# =========================
# SMALL UTILITIES
# =========================

def safe_df(df: pd.DataFrame):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        try:
            st.dataframe(df)
        except Exception:
            st.dataframe(df.astype(str))
    except Exception:
        st.dataframe(df.astype(str))

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_key(k: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (k or "").lower()).strip("_")

def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )

def highlight_html(text: str, needle: str, max_chars: int = 380) -> str:
    t = text or ""
    n = (needle or "").strip()
    if not t:
        return "<em>(empty)</em>"
    if not n:
        return escape_html(t[:max_chars])

    tl = t.lower()
    nl = n.lower()
    idx = tl.find(nl)
    if idx < 0 and len(nl) >= 25:
        idx = tl.find(nl[:25])
    if idx < 0:
        return escape_html(t[:max_chars])

    start = max(0, idx - max_chars // 3)
    end = min(len(t), idx + len(n) + (max_chars // 3) * 2)
    snippet = t[start:end]

    local_idx = idx - start
    local_end = min(len(snippet), local_idx + len(n))

    pre = escape_html(snippet[:local_idx])
    mid = escape_html(snippet[local_idx:local_end])
    post = escape_html(snippet[local_end:])
    return f"{pre}<mark>{mid}</mark>{post}"

def flatten_json(data: Any, parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    def rec(obj: Any, path: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                nk = f"{path}{sep}{k}" if path else str(k)
                rec(v, nk)
        elif isinstance(obj, list):
            out[path] = obj
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    rec(v, f"{path}{sep}{i}")
        else:
            out[path] = obj
    rec(data, parent)
    return out

def _collect_strings(obj: Any) -> List[str]:
    out: List[str] = []
    def rec(x):
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, list):
            for i in x:
                rec(i)
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
    rec(obj)
    seen = set()
    res = []
    for s in out:
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res

def detect_title_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if isinstance(v, str):
            kn = normalize_key(k)
            if kn in {"title", "listing_title", "name"} or kn.endswith("_title"):
                return k
    cands = []
    for k, v in flat.items():
        if isinstance(v, str):
            s = v.strip()
            if 10 <= len(s) <= 160 and "\n" not in s:
                cands.append((len(s), k))
    return sorted(cands)[0][1] if cands else None

def is_review_key(k: str) -> bool:
    kn = normalize_key(k)
    return ("review" in kn) or ("reviews" in kn)

def is_indexed_path(k: str) -> bool:
    return bool(re.search(r"(^|\.)\d+(\.|$)", k))

def detect_text_keys(flat: Dict[str, Any], min_len: int = 120) -> List[str]:
    keys = []
    for k, v in flat.items():
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= min_len or ("\n" in s) or (len(re.split(r"(?<=[.!?])\s+", s)) >= 2 and len(s) >= 80):
                keys.append(k)
    return sorted(keys)

PREFERRED_TEXT_NORMAL_KEYS = {"summary", "the_space", "guest_access", "other_things_to_note", "description"}

def choose_editable_text_keys(flat: Dict[str, Any], title_key: Optional[str]) -> List[str]:
    preferred = []
    for k, v in flat.items():
        if isinstance(v, str) and normalize_key(k) in PREFERRED_TEXT_NORMAL_KEYS:
            preferred.append(k)

    detected = detect_text_keys(flat)
    out = []
    for k in preferred + detected:
        if k == title_key:
            continue
        if is_review_key(k):
            continue
        if k not in out:
            out.append(k)
    return out

def build_corpus(title: str, texts: Dict[str, str]) -> Dict[str, str]:
    corpus = {"title": title or ""}
    for k, v in (texts or {}).items():
        corpus[str(k)] = "" if v is None else str(v)
    return corpus


# =========================
# HOUSE RULES (EDITABLE TEXT)
# =========================

def house_rules_to_text(hr: Any) -> str:
    """Stable, readable text for house_rules dict/list/string."""
    if hr is None:
        return ""
    if isinstance(hr, str):
        return hr.strip()

    if isinstance(hr, list):
        parts = []
        for item in hr:
            if isinstance(item, str) and item.strip():
                parts.append(f"- {item.strip()}")
            else:
                nested = house_rules_to_text(item)
                if nested:
                    parts.append(nested)
        return "\n".join(parts).strip()

    if isinstance(hr, dict):
        lines = []
        for section, items in hr.items():
            sec = str(section).strip() if section is not None else ""
            if sec:
                lines.append(f"{sec}:")
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, str) and it.strip():
                        lines.append(f"- {it.strip()}")
                    else:
                        nested = house_rules_to_text(it)
                        if nested:
                            lines.append(f"- {nested}")
            else:
                nested = house_rules_to_text(items)
                if nested:
                    lines.append(f"- {nested}")
            lines.append("")
        return "\n".join(lines).strip()

    return str(hr).strip()


# =========================
# AMENITIES: included / not_included
# =========================

def extract_amenities_included_not_included(data: Any) -> Tuple[List[str], List[str]]:
    included: List[str] = []
    not_included: List[str] = []
    found_any = False

    def walk(obj: Any):
        nonlocal found_any
        if isinstance(obj, dict):
            for k, v in obj.items():
                kn = normalize_key(str(k))
                if kn == "amenities_included" and isinstance(v, (dict, list)):
                    found_any = True
                    included.extend(_collect_strings(v))
                elif kn == "amenities_not_included" and isinstance(v, (dict, list)):
                    found_any = True
                    not_included.extend(_collect_strings(v))
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        res = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                res.append(s)
        return res

    included = dedupe(included)
    not_included = dedupe(not_included)

    if not found_any:
        # fallback: treat any "amenit*" field as included
        all_found: List[str] = []

        def walk_old(obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "amenit" in normalize_key(str(k)) and isinstance(v, (dict, list)):
                        all_found.extend(_collect_strings(v))
                    walk_old(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk_old(it)

        walk_old(data)
        return dedupe(all_found), []

    return included, not_included


# =========================
# REVIEWS (READ-ONLY)
# =========================

def build_readonly_reviews(flat: Dict[str, Any]) -> Dict[str, str]:
    readonly: Dict[str, str] = {}

    review_container_keys = []
    for k, v in flat.items():
        if not is_review_key(k):
            continue
        if is_indexed_path(k):
            continue
        if isinstance(v, (str, list)):
            review_container_keys.append(k)

    for k in sorted(set(review_container_keys)):
        v = flat.get(k)
        if isinstance(v, str):
            txt = v.strip()
            if txt:
                readonly[k] = txt[:12000]
        elif isinstance(v, list):
            parts: List[str] = []
            for item in v[:200]:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for _, vv in item.items():
                        if isinstance(vv, str) and vv.strip():
                            parts.append(vv.strip())
            if parts:
                readonly[k] = "\n\n".join(parts)[:12000]

    return readonly


# =========================
# NUMBER PARSING
# =========================

_NUM_WORDS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20
}

def word_to_int(w: str) -> Optional[int]:
    return _NUM_WORDS.get(re.sub(r"[^a-z]", "", (w or "").lower()))

def parse_int_maybe(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        try:
            return int(x)
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"\d{1,4}", s):
            return int(s)
        m = re.search(r"\b(\d{1,4})\b", s)  # "5 bedrooms" -> 5
        if m:
            return int(m.group(1))
        if re.fullmatch(r"[A-Za-z]+", s):
            return word_to_int(s)
        m2 = re.search(r"\b([A-Za-z]+)\b", s)
        if m2:
            return word_to_int(m2.group(1))
    return None


# =========================
# AMENITY MATCHING
# =========================

AMENITY_SYNONYMS = {
    "hot tub": ["hot tub", "hot tubs", "jacuzzi", "spa tub", "whirlpool", "spa"],
    "pool": ["pool", "pools", "swimming pool", "swimming pools"],
    "bbq grill": ["bbq", "barbecue", "bbq grill", "grill", "grills"],
    "dedicated workspace": ["dedicated workspace", "workspace", "work desk", "desk"],
    "ev charger": ["ev charger", "electric vehicle charger", "tesla charger"],
    "fire pit": ["fire pit", "fire-pit", "fire pits"],
    "gym": ["gym", "gyms", "fitness center", "fitness centre"],
    "sauna": ["sauna", "saunas"],
    "lake view": ["lake view", "lake views"],
}

_AMENITY_MODIFIER_TOKENS = {
    "shared", "private", "in", "the", "a", "an", "and",
    "building", "premises", "upon", "request", "available",
    "paid", "free", "street", "on", "off",
}

def normalize_amenity_phrase(s: str) -> str:
    t = (s or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[–—]", " ", t)
    t = re.sub(r"[^a-z0-9\s/]+", " ", t)
    return normalize_ws(t)

def amenity_tokens(s: str) -> List[str]:
    t = normalize_amenity_phrase(s)
    toks = [x for x in re.split(r"[\s/]+", t) if x]
    toks = [x for x in toks if x not in _AMENITY_MODIFIER_TOKENS]
    return toks

def amenity_soft_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_norm = normalize_amenity_phrase(a)
    b_norm = normalize_amenity_phrase(b)
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    at = set(amenity_tokens(a_norm))
    bt = set(amenity_tokens(b_norm))
    if not at or not bt:
        return False
    return at.issubset(bt) or bt.issubset(at)

def canon_amenity(a: str) -> str:
    a0 = normalize_amenity_phrase(a)
    for canon, syns in AMENITY_SYNONYMS.items():
        for s in syns:
            s0 = normalize_amenity_phrase(s)
            if re.search(rf"(?i)\b{re.escape(s0)}\b", a0):
                return canon
    return a0

def all_known_amenities() -> Tuple[Dict[str, str], set, set]:
    display = {}
    high = set()
    low = set()
    for _, items in HIGH_INTENT_AMENITIES.items():
        for it in items:
            c = canon_amenity(it)
            high.add(c)
            display.setdefault(c, it)
    for _, items in LOW_PRIORITY_AMENITIES.items():
        for it in items:
            c = canon_amenity(it)
            low.add(c)
            display.setdefault(c, it)
    return display, high, low

def _regex_for_phrase(phrase: str) -> re.Pattern:
    s0 = normalize_amenity_phrase(phrase)
    escaped = re.escape(s0)
    escaped = escaped.replace("\\ ", r"[\s\-]+")
    escaped = escaped.replace("\\/", r"[\s]*\/[\s]*")
    return re.compile(r"(?i)\b" + escaped + r"\b")

def find_amenity_hits(text: str, canon: str) -> List[Tuple[int, int, str]]:
    t = text or ""
    syns = AMENITY_SYNONYMS.get(canon, [canon])
    hits = []
    for s in syns:
        pat = _regex_for_phrase(s)
        for m in pat.finditer(t):
            hits.append((m.start(), m.end(), t[m.start():m.end()]))
    hits.sort(key=lambda x: x[0])
    return hits

def claim_present_in_included(claim: str, amenities_included: List[str]) -> bool:
    if not claim:
        return False
    for a in amenities_included or []:
        if amenity_soft_match(claim, a):
            return True
    return False


# =========================
# INCLUDED vs NOT INCLUDED CONTRADICTIONS
# =========================

def detect_included_not_included_conflicts(inc: List[str], out: List[str]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for a_in in inc or []:
        for a_out in out or []:
            if amenity_soft_match(a_in, a_out) or amenity_soft_match(canon_amenity(a_in), canon_amenity(a_out)):
                issues.append({
                    "issue_type": "Amenity list contradiction (included vs not included)",
                    "severity": "high",
                    "field": "amenities",
                    "claim": a_in,
                    "ground_truth": a_out,
                    "evidence": f"included: {a_in} | not_included: {a_out}",
                    "reason": f"Amenity appears both included and not included (variation match): '{a_in}' vs '{a_out}'."
                })
    return issues


# =========================
# TITLE RULE
# =========================

def is_amenity_issue(issue: Dict[str, Any]) -> bool:
    t = (issue.get("issue_type") or "").lower()
    return ("amenity" in t) or ("amenities" in t) or ("shared vs private" in t) or ("title amenity" in t)

def _count_high_intent_in_title(title: str) -> int:
    _display, high_set, _low_set = all_known_amenities()
    ct = 0
    for canon in high_set:
        if find_amenity_hits(title or "", canon):
            ct += 1
    return ct

def _suppress_title_amenity_issues_if_title_full(issues: List[Dict[str, Any]], title: str) -> List[Dict[str, Any]]:
    if _count_high_intent_in_title(title) < 3:
        return issues
    out: List[Dict[str, Any]] = []
    for it in issues:
        itype = (it.get("issue_type") or "").lower()
        field = normalize_key(str(it.get("field", "") or ""))
        if "title amenity stuffing" in itype:
            out.append(it)
            continue
        if field == "title" and is_amenity_issue(it):
            continue
        out.append(it)
    return out


# =========================
# DISPLAY POLICY (QA)
# =========================

IMPORTANT_AMENITY_CATEGORIES = {
    "Comfort & Wellness",
    "Convenience & Functionality",
    "Leisure & Outdoor",
    "Scenic Views",
}

def amenity_category_index() -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for cat, items in HIGH_INTENT_AMENITIES.items():
        for it in items:
            idx[canon_amenity(it)] = cat
    for cat, items in LOW_PRIORITY_AMENITIES.items():
        for it in items:
            idx[canon_amenity(it)] = cat
    return idx

def important_amenity_set() -> set:
    idx = amenity_category_index()
    return {a for a, cat in idx.items() if cat in IMPORTANT_AMENITY_CATEGORIES}

def _clean_amenity_hint(s: str) -> str:
    x = normalize_ws(s).lower()
    x = re.sub(r"^\s*\d+\s*[x×]\s*", "", x)
    x = re.sub(r"^\s*\d+\s+", "", x)
    x = re.sub(r"\b(shared|private|in\s+building|upon\s+request|available\s+upon\s+request)\b", "", x)
    return normalize_ws(x).strip()

def guess_issue_amenity_canon(issue: Dict[str, Any]) -> Optional[str]:
    for k in ("claim", "evidence", "reason"):
        raw = issue.get(k) or ""
        if not isinstance(raw, str):
            continue
        raw = raw.strip()
        if not raw:
            continue
        cleaned = _clean_amenity_hint(raw)
        cleaned = cleaned.replace("shared pool", "pool").replace("private pool", "pool")
        cleaned = cleaned.replace("shared hot tub", "hot tub").replace("private hot tub", "hot tub")
        if cleaned:
            return canon_amenity(cleaned)
    return None

def filter_issues_for_qa(issues: List[Dict[str, Any]], show_all: bool = False) -> List[Dict[str, Any]]:
    if show_all:
        return issues
    important = important_amenity_set()
    out: List[Dict[str, Any]] = []
    for it in issues:
        itype = (it.get("issue_type") or "").lower()
        if itype.startswith("amenity list contradiction"):
            out.append(it)
            continue
        if "amenity selected but not mentioned" in itype:
            continue
        if not is_amenity_issue(it):
            out.append(it)
            continue
        if "title amenity stuffing" in itype:
            out.append(it)
            continue
        canon = guess_issue_amenity_canon(it)
        if canon and canon in important:
            out.append(it)
    return out


# =========================
# COUNT RULE ENGINE
# =========================

def detect_structured_count(flat: Dict[str, Any], key_hints: List[str], key_exclude: List[str]) -> Tuple[Optional[int], Optional[str]]:
    best_key = None
    best_val = None
    best_score = -1
    for k, v in flat.items():
        kn = normalize_key(k)
        if any(h in kn for h in key_hints) and not any(ex in kn for ex in key_exclude):
            n = parse_int_maybe(v)
            if n is None:
                continue
            score = 0
            if "total" in kn:
                score += 3
            if "num" in kn or "count" in kn or "number" in kn:
                score += 2
            if kn.endswith("_id"):
                score -= 3
            if score > best_score:
                best_score = score
                best_key = k
                best_val = n
    return best_val, best_key

def extract_explicit_count_claims(text: str, explicit_patterns: List[re.Pattern], exclude_patterns: Optional[List[re.Pattern]] = None) -> List[Tuple[int, str]]:
    t = text or ""
    claims: List[Tuple[int, str]] = []
    exclude_patterns = exclude_patterns or []
    for pat in explicit_patterns:
        for m in pat.finditer(t):
            ev = (m.group(0) or "").strip()
            if not ev:
                continue
            if any(ex.search(ev) for ex in exclude_patterns):
                continue
            raw = None
            if m.lastindex and m.lastindex >= 1:
                raw = m.group(1)
            if raw is None:
                mm = re.search(r"(\d{1,4}|[A-Za-z]+)", ev)
                raw = mm.group(1) if mm else None
            n = parse_int_maybe(raw)
            if n is not None:
                claims.append((n, ev))
    return claims

def run_count_rules(flat: Dict[str, Any], corpus: Dict[str, str], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for rule in rules:
        structured_value, structured_key = detect_structured_count(
            flat=flat,
            key_hints=rule["structured_key_hints"],
            key_exclude=rule.get("structured_key_exclude", []),
        )
        if structured_value is None:
            continue
        for field, text in corpus.items():
            claims = extract_explicit_count_claims(
                text=text,
                explicit_patterns=rule["text_explicit_patterns"],
                exclude_patterns=rule.get("text_exclude_patterns", []),
            )
            for n, ev in claims:
                if n != structured_value:
                    issues.append({
                        "issue_type": rule["issue_type"],
                        "severity": rule.get("severity", "high"),
                        "field": field,
                        "claim": str(n),
                        "ground_truth": f"{structured_value} (from {structured_key})" if structured_key else str(structured_value),
                        "evidence": ev,
                        "reason": f"{field} suggests {rule['label']} is {n}, but the structured value is {structured_value}."
                    })
    return issues

COUNT_RULES = [
    {
        "label": "bedrooms",
        "issue_type": "Room count mismatch",
        "severity": "high",
        "structured_key_hints": ["bedrooms", "bedroom", "num_bedrooms", "rooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*bedrooms?\b"),
            re.compile(r"(?i)\b(\d{1,3})\s*br\b"),
        ],
        "text_exclude_patterns": [],
    },
    {
        "label": "beds",
        "issue_type": "Bed count mismatch",
        "severity": "high",
        "structured_key_hints": ["total_beds", "beds_count", "bed_count", "num_beds", "number_of_beds", "beds"],
        "structured_key_exclude": ["bedroom", "bedrooms"],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*beds?\b"),
        ],
        "text_exclude_patterns": [re.compile(r"(?i)\bbedrooms?\b")],
    },
    {
        "label": "bathrooms",
        "issue_type": "Bathroom count mismatch",
        "severity": "high",
        "structured_key_hints": ["bathrooms", "bathroom", "baths", "num_bathrooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(?:bathrooms?|baths?)\b"),
            re.compile(r"(?i)\b(\d{1,3})\s*ba\b"),
        ],
        "text_exclude_patterns": [],
    },
    {
        "label": "max guests",
        "issue_type": "Max capacity mismatch",
        "severity": "high",
        "structured_key_hints": ["max_guests", "guests", "accommodates", "capacity"],
        "structured_key_exclude": ["min", "minimum"],
        "text_explicit_patterns": [
            re.compile(r"(?i)\bsleeps?\s+(\d{1,3}|[A-Za-z]+)\b"),
            re.compile(r"(?i)\baccommodates?\s+(\d{1,3}|[A-Za-z]+)\b"),
            re.compile(r"(?i)\bup to\s+(\d{1,3}|[A-Za-z]+)\s+guests?\b"),
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+guests?\s+maximum\b"),
        ],
        "text_exclude_patterns": [],
    },
]


# =========================
# LLM HELPERS
# =========================

def openai_client(api_key: str):
    try:
        from openai import OpenAI  # type: ignore
        return ("new", OpenAI(api_key=api_key))
    except Exception:
        import openai  # type: ignore
        openai.api_key = api_key
        return ("old", openai)

def llm_json(mode_client, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    mode, client = mode_client
    if mode == "new":
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
    else:
        resp = client.ChatCompletion.create(model=model, messages=messages, temperature=0.0)
        txt = resp["choices"][0]["message"]["content"] or "{}"

    try:
        return json.loads(txt)
    except Exception:
        s = txt.find("{")
        e = txt.rfind("}")
        if s >= 0 and e > s:
            return json.loads(txt[s:e+1])
        return {"issues": [], "field_mapping": {}}

def _find_best_field_for_evidence(evidence: str, corpus: Dict[str, str]) -> Optional[str]:
    ev = (evidence or "").strip()
    if not ev:
        return None
    ev_l = ev.lower()
    best_field = None
    best_idx = None
    for field, text in corpus.items():
        t = text or ""
        idx = t.lower().find(ev_l)
        if idx >= 0:
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_field = field
    return best_field

def ground_and_locate_llm_issues(
    llm_issues: List[Dict[str, Any]],
    corpus: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    listed_canons = {canon_amenity(a) for a in (amenities_included or []) + (amenities_not_included or [])}

    for it in llm_issues or []:
        issue_type = str(it.get("issue_type", "") or "")
        claim = str(it.get("claim", "") or "")
        evidence = str(it.get("evidence", "") or "").strip()

        if "amenit" in issue_type.lower() or "amenities" in issue_type.lower():
            canon_guess = guess_issue_amenity_canon(it)
            if canon_guess and canon_guess not in listed_canons:
                continue
            if claim and claim_present_in_included(claim, amenities_included):
                continue

        needle = evidence if evidence else claim
        if not needle.strip():
            out.append(it)
            continue

        located_field = _find_best_field_for_evidence(needle, corpus)
        if located_field is None:
            continue

        it["field"] = located_field
        out.append(it)

    return out


# =========================
# DETERMINISTIC AMENITY CHECKS
# =========================

def deterministic_required_amenity_checks(
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    display, high_set, _low_set = all_known_amenities()

    included_canons = {canon_amenity(a) for a in (amenities_included or [])}
    not_included_canons = {canon_amenity(a) for a in (amenities_not_included or [])}
    listed_canons = included_canons | not_included_canons

    corpus = {"title": title, **texts}
    title_is_full = _count_high_intent_in_title(title) >= 3

    for canon in sorted(high_set):
        best_mention = None
        for field, text in corpus.items():
            hits = find_amenity_hits(text, canon)
            if hits:
                best_mention = (field, hits[0][2])
                break

        if best_mention and canon not in included_canons:
            if canon not in listed_canons:
                continue
            f, ev = best_mention
            if normalize_key(f) == "title" and title_is_full:
                continue
            gt_label = "Listed as NOT included" if canon in not_included_canons else "Not in amenities_included"
            issues.append({
                "issue_type": "Amenity mentioned but not selected",
                "severity": "high",
                "field": f,
                "claim": display.get(canon, canon),
                "ground_truth": gt_label,
                "evidence": ev,
                "reason": f"{f} mentions '{display.get(canon, canon)}' but it is not in amenities_included."
            })

    return issues


# =========================
# CORE CHECKER
# =========================

def build_ground_truth_pairs(flat: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = []
    for k, v in flat.items():
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, str) and len(v) > 350:
            continue
        pairs.append({"key": k, "value": v})
    return pairs[:450]

def run_llm_checker(
    flat: Dict[str, Any],
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode_client = openai_client(api_key)
    gt_pairs = build_ground_truth_pairs(flat)

    payload = {
        "ground_truth_pairs": gt_pairs,
        "amenities_included": amenities_included,
        "amenities_not_included": amenities_not_included,
        "text_fields": {"title": title, **texts},
        "policy_notes": [
            "Only flag amenity inconsistencies if the amenity is listed in amenities_included or amenities_not_included.",
            "If an amenity is not listed in either list, do NOT flag it.",
            "TITLE RULE: Do not create title-level amenity inaccuracies if title already has 3+ high-intent amenities; only use 'Title amenity stuffing'.",
        ],
        "output_schema": {"field_mapping": "best-effort", "issues": "list"},
    }

    system = (
        "You are a discrepancy checker. Return ONLY valid JSON with keys: field_mapping, issues.\n"
        "Be conservative. AMENITIES RULE: Only flag amenity inconsistencies for amenities present in amenities_included or amenities_not_included.\n"
        "TITLE RULE: Do not add amenity inaccuracies for title if title already contains 3+ high-intent amenities; only use 'Title amenity stuffing'.\n"
        "Each issue must include: issue_type, severity, field, reason, evidence, claim(optional), ground_truth(optional)."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    out = llm_json(mode_client, model, messages)
    llm_issues_raw = out.get("issues") or []
    mapping = out.get("field_mapping") or {}

    corpus = build_corpus(title, texts)
    llm_issues = ground_and_locate_llm_issues(llm_issues_raw, corpus, amenities_included, amenities_not_included)

    det_issues = deterministic_required_amenity_checks(title, texts, amenities_included, amenities_not_included)
    count_issues = run_count_rules(flat=flat, corpus=corpus, rules=COUNT_RULES)
    conflict_issues = detect_included_not_included_conflicts(amenities_included, amenities_not_included)

    merged = []
    seen = set()
    for it in (llm_issues + det_issues + conflict_issues + count_issues):
        key = (
            str(it.get("issue_type", "")),
            str(it.get("field", "")),
            (str(it.get("evidence", ""))[:80]),
            (str(it.get("claim", ""))[:80]),
            (str(it.get("ground_truth", ""))[:80]),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)

    merged = _suppress_title_amenity_issues_if_title_full(merged, title)
    return merged, mapping


# =========================
# STREAMLIT DISPLAY HELPERS
# =========================

def load_json_upload(upload) -> Any:
    raw = upload.read()
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))

def issues_to_df(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["severity", "issue_type", "field", "reason", "claim", "ground_truth", "evidence"]
    if not issues:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(issues)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def sev_rank(s: str) -> int:
    s = (s or "").lower()
    return {"high": 0, "medium": 1, "low": 2}.get(s, 3)

def render_issue_cards(issues: List[Dict[str, Any]], title: str, texts: Dict[str, str]):
    if not issues:
        st.success("No issues found (with current settings).")
        return

    combined_all = title + "\n\n" + "\n\n".join([v for v in texts.values() if v])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for it in issues:
        grouped.setdefault(it.get("field", "text") or "text", []).append(it)

    for field, items in grouped.items():
        st.subheader(f"Field: {field} ({len(items)})")

        if normalize_key(field) == "title":
            base = title
        elif normalize_key(field) == "amenities":
            base = combined_all
        else:
            base = texts.get(field, "")
            if not base:
                base = combined_all

        for it in sorted(items, key=lambda x: (sev_rank(x.get("severity")), x.get("issue_type", ""))):
            sev = (it.get("severity") or "low").lower()
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            with st.expander(f"{icon} {it.get('issue_type','Issue')} — {str(it.get('reason',''))[:90]}"):
                st.markdown(f"**Reason:** {it.get('reason','')}")
                if it.get("ground_truth"):
                    st.markdown(f"**Ground truth:** `{it.get('ground_truth')}`")
                if it.get("claim"):
                    st.markdown(f"**Claim:** `{it.get('claim')}`")
                ev = it.get("evidence") or ""
                st.markdown("**Evidence (highlighted):**")
                st.markdown(
                    f"<div style='padding:10px;border:1px solid #ddd;border-radius:10px'>{highlight_html(base, ev)}</div>",
                    unsafe_allow_html=True,
                )

def dedupe_issues_for_display(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def norm(s: Any) -> str:
        return normalize_ws("" if s is None else str(s)).lower()

    def field_priority(f: str) -> int:
        fn = normalize_key(f or "")
        if fn == "title":
            return 0
        if fn == "summary":
            return 1
        if fn == "house_rules":
            return 2
        if fn == "amenities":
            return 9
        return 5

    buckets: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for it in issues:
        key = (
            norm(it.get("issue_type")),
            norm(it.get("claim")),
            norm(it.get("ground_truth")),
            norm(it.get("evidence"))[:120],
            norm(it.get("reason"))[:120],
        )
        buckets.setdefault(key, []).append(it)

    out: List[Dict[str, Any]] = []
    for _, items in buckets.items():
        best = sorted(
            items,
            key=lambda x: (
                sev_rank(x.get("severity")),
                field_priority(str(x.get("field", ""))),
                -len(norm(x.get("evidence"))),
            ),
        )[0]
        out.append(best)

    out.sort(key=lambda x: (sev_rank(x.get("severity")), str(x.get("issue_type", "")), str(x.get("field", ""))))
    return out


# =========================
# STREAMLIT APP
# =========================

def main():
    st.set_page_config(page_title="JSON Discrepancy Checker — LLM", layout="wide")
    st.title("JSON Discrepancy Checker — LLM semantic")

    st.sidebar.header("Upload JSON")
    upload = st.sidebar.file_uploader("JSON file", type=["json"])

    st.sidebar.header("LLM settings (hard-coded)")
    st.sidebar.write(f"Model: `{OPENAI_MODEL}`")

    run_live = st.sidebar.checkbox("Re-check automatically while editing", value=True)
    show_all_issues = st.sidebar.checkbox("Show all issues (no filtering)", value=False)

    if not upload:
        st.info("Upload a JSON file.")
        return

    try:
        data = load_json_upload(upload)
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        return

    # Your JSON can be a list of listings
    if isinstance(data, list):
        if not data:
            st.error("JSON list is empty.")
            return

        options = []
        for i, obj in enumerate(data):
            if isinstance(obj, dict):
                t = obj.get("listing_title") or obj.get("title") or obj.get("name") or f"item_{i}"
                lid = obj.get("target_listing_id") or ""
                options.append((i, f"[{i}] {t} {f'(id={lid})' if lid else ''}".strip()))
            else:
                options.append((i, f"[{i}] (non-dict item)"))

        sel = st.sidebar.selectbox("Which listing in the JSON list?", options=options, format_func=lambda x: x[1])
        data = data[sel[0]]

    if not isinstance(data, dict):
        st.error("Selected JSON item must be an object (dict).")
        return

    flat = flatten_json(data)

    amenities_included, amenities_not_included = extract_amenities_included_not_included(data)

    title_key = detect_title_key(flat)
    title_val = str(flat.get(title_key, "")) if title_key else ""

    # Remove description from UI
    text_keys = choose_editable_text_keys(flat, title_key=title_key)
    text_keys = [k for k in text_keys if normalize_key(k) != "description"]

    readonly_reviews = build_readonly_reviews(flat)

    # ✅ House rules editable: dict/list/string supported
    house_rules_obj = data.get("house_rules")
    house_rules_initial = house_rules_to_text(house_rules_obj)

    st.header("Editable listing text")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Title")
        title_edit = st.text_input("Title", value=title_val, key="__title")

        st.subheader("Amenities included (ground truth)")
        st.caption(f"Detected {len(amenities_included)}")
        safe_df(pd.DataFrame({"amenity": amenities_included})) if amenities_included else st.write("—")

        st.subheader("Amenities not included (reference)")
        st.caption(f"Detected {len(amenities_not_included)}")
        safe_df(pd.DataFrame({"amenity": amenities_not_included})) if amenities_not_included else st.write("—")

    with c2:
        edited_texts: Dict[str, str] = {}

        st.subheader("Editable text fields")
        if not text_keys:
            st.info("No editable text fields detected in this JSON.")
        for k in text_keys:
            edited_texts[k] = st.text_area(k, value=str(flat.get(k, "")), height=160, key=f"__txt_{k}")

        st.subheader("Read-only review fields (not editable)")
        if readonly_reviews:
            for k, txt in readonly_reviews.items():
                try:
                    st.text_area(k, value=txt, height=160, key=f"__ro_{k}", disabled=True)
                except TypeError:
                    st.markdown(f"**{k} (read-only)**")
                    st.code(txt[:4000])
        else:
            st.write("—")

        # ✅ House rules editable form (what you asked for)
        st.subheader("House rules (editable)")
        house_rules_edit = st.text_area(
            "house_rules",
            value=house_rules_initial,
            height=220,
            key="__txt_house_rules",
            help="This is editable so you can test the checker by changing house rules."
        )
        edited_texts["house_rules"] = house_rules_edit

    # Include reviews in checking too (unchanged behavior)
    all_texts_for_checking = {**edited_texts, **readonly_reviews}

    def run_once():
        issues, mapping = run_llm_checker(
            flat=flat,
            title=title_edit,
            texts=all_texts_for_checking,
            amenities_included=amenities_included,
            amenities_not_included=amenities_not_included,
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
        )
        st.session_state.setdefault("runs", [])
        st.session_state["runs"].append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "issues": issues,
            "mapping": mapping,
            "title": title_edit,
            "texts": all_texts_for_checking,
        })
        st.session_state["current"] = st.session_state["runs"][-1]

    if run_live:
        run_once()
    else:
        if st.button("Run checker"):
            run_once()

    current = st.session_state.get("current")
    if not current:
        return

    issues = current["issues"]

    # UI filtering/dedupe
    issues_display = issues if show_all_issues else filter_issues_for_qa(issues, show_all=False)
    issues_display = dedupe_issues_for_display(issues_display)

    st.header("Results")

    counts = {"high": 0, "medium": 0, "low": 0}
    for it in issues_display:
        counts[(it.get("severity") or "low").lower()] = counts.get((it.get("severity") or "low").lower(), 0) + 1
    m1, m2, m3 = st.columns(3)
    m1.metric("High", counts.get("high", 0))
    m2.metric("Medium", counts.get("medium", 0))
    m3.metric("Low", counts.get("low", 0))

    df = issues_to_df(issues_display).copy()
    df = df.sort_values(by=["severity", "issue_type"], key=lambda s: s.map(sev_rank), ascending=True)
    st.subheader("Issues table")
    safe_df(df)

    st.download_button("Download issues (CSV)", df.to_csv(index=False).encode("utf-8"), "issues.csv", "text/csv")
    st.download_button("Download issues (JSON)", json.dumps(issues_display, ensure_ascii=False, indent=2).encode("utf-8"), "issues.json", "application/json")

    st.subheader("Issue details (highlights + reasons)")
    render_issue_cards(issues_display, title_edit, all_texts_for_checking)

    st.sidebar.header("Runs")
    st.sidebar.write(f"{len(st.session_state.get('runs', []))} run(s) this session")
    if st.sidebar.button("Clear runs"):
        st.session_state["runs"] = []
        st.session_state["current"] = None
        st.rerun()


if __name__ == "__main__":
    main()
