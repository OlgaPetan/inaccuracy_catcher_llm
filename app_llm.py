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
# AMENITIES
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
    """Robust Streamlit display (handles Arrow conversion failures)."""
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


def _collect_strings(obj):
    """Recursively collect non-empty strings from dict/list structures."""
    out = []

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
            if 10 <= len(s) <= 140 and "\n" not in s:
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


def choose_editable_text_keys(flat: Dict[str, Any], title_key: Optional[str], house_rules_key: Optional[str]) -> List[str]:
    preferred = []
    for k, v in flat.items():
        if isinstance(v, str) and normalize_key(k) in PREFERRED_TEXT_NORMAL_KEYS:
            preferred.append(k)

    detected = detect_text_keys(flat)
    out = []
    for k in preferred + detected:
        if k == title_key:
            continue
        if k == house_rules_key:
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
# AMENITIES EXTRACTION (NEW API KEYS)
# =========================

def extract_all_amenities_fallback(data: Any) -> List[str]:
    """Fallback: old behavior - collect from any key containing 'amenit'."""
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kn = normalize_key(str(k))
                if "amenit" in kn and isinstance(v, (dict, list)):
                    found.extend(_collect_strings(v))
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    seen = set()
    res = []
    for s in found:
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res


def extract_amenities_included_not_included(data: Any) -> Tuple[List[str], List[str]]:
    """
    New API:
      - amenities_included
      - amenities_not_included

    If neither exists, fall back to old extraction and return (all, []).
    """
    included: List[str] = []
    not_included: List[str] = []
    found_any_new = False

    def walk(obj):
        nonlocal found_any_new, included, not_included
        if isinstance(obj, dict):
            for k, v in obj.items():
                kn = normalize_key(str(k))
                if kn == "amenities_included" and isinstance(v, (dict, list)):
                    found_any_new = True
                    included.extend(_collect_strings(v))
                elif kn == "amenities_not_included" and isinstance(v, (dict, list)):
                    found_any_new = True
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

    if not found_any_new:
        return extract_all_amenities_fallback(data), []

    return included, not_included


# =========================
# HOUSE RULES (ONLY TEXT BOX SUPPORT)
# =========================

def find_node_by_key(data: Any, target_key: str) -> Optional[Any]:
    """Find first occurrence of key anywhere in nested JSON. Returns value under that key."""
    target = normalize_key(target_key)

    def rec(x: Any) -> Optional[Any]:
        if isinstance(x, dict):
            for k, v in x.items():
                if normalize_key(str(k)) == target:
                    return v
                hit = rec(v)
                if hit is not None:
                    return hit
        elif isinstance(x, list):
            for item in x:
                hit = rec(item)
                if hit is not None:
                    return hit
        return None

    return rec(data)


def house_rules_to_text(hr: Any) -> str:
    """Stable plain-text representation for dict/list/string house rules."""
    if hr is None:
        return ""
    if isinstance(hr, str):
        return normalize_ws(hr)

    if isinstance(hr, list):
        parts = []
        for item in hr:
            if isinstance(item, str) and item.strip():
                parts.append(f"- {item.strip()}")
            elif isinstance(item, (dict, list)):
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
                    elif isinstance(it, (dict, list)):
                        nested = house_rules_to_text(it)
                        if nested:
                            lines.append(f"- {nested}")
            else:
                nested = house_rules_to_text(items)
                if nested:
                    lines.append(f"- {nested}")
            lines.append("")
        return "\n".join(lines).strip()

    return normalize_ws(str(hr))


# =========================
# REVIEWS (checked for inconsistencies too)
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
        if re.fullmatch(r"[A-Za-z]+", s):
            return word_to_int(s)
    return None


# =========================
# AMENITY MATCHING (DETERMINISTIC + SOFT)
# =========================

def extract_number_near(text: str, idx: int, window: int = 40) -> Optional[int]:
    if not text:
        return None
    start = max(0, idx - window)
    snippet = text[start:idx]
    m = re.search(r"(\d{1,3})\s*$", snippet)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    toks = re.findall(r"[A-Za-z']+", snippet.lower())
    if toks:
        return word_to_int(toks[-1])
    return None


AMENITY_SYNONYMS = {
    "hot tub": ["hot tub", "jacuzzi", "spa tub", "whirlpool", "spa"],
    "pool": ["pool", "swimming pool"],
    "bbq grill": ["bbq", "barbecue", "bbq grill", "grill"],
    "dedicated workspace": ["dedicated workspace", "workspace", "work desk", "desk"],
    "ev charger": ["ev charger", "electric vehicle charger", "tesla charger"],
    "game console": ["game console", "ps5", "playstation", "xbox", "nintendo switch"],
    "movie theater": ["movie theater", "home theater", "cinema room"],
    "fire pit": ["fire pit", "fire-pit"],
    "indoor fireplace": ["indoor fireplace", "fireplace"],
    "gym": ["gym", "fitness center", "fitness centre"],
    "sauna": ["sauna"],
    "tv": ["tv", "television"],
    "air conditioning": ["air conditioning", "a/c", "ac"],
    "dryer": ["dryer", "tumble dryer"],
}

_AMENITY_MODIFIER_TOKENS = {
    "shared", "private", "in", "the", "a", "an", "and",
    "building", "premises", "upon", "request", "available",
    "paid", "free", "street", "on", "off", "fully", "not",
    "fully_fenced", "fenced", "not_fully_fenced",
}

def normalize_amenity_phrase(s: str) -> str:
    t = (s or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[–—]", " ", t)
    t = re.sub(r"[^a-z0-9\s/]+", " ", t)
    t = normalize_ws(t)
    return t

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
    for canon in AMENITY_SYNONYMS.keys():
        if amenity_soft_match(a0, canon):
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

def find_amenity_hits(text: str, canon: str) -> List[Tuple[int, int, str]]:
    t = text or ""
    syns = AMENITY_SYNONYMS.get(canon, [canon])
    hits = []
    for s in syns:
        s0 = normalize_amenity_phrase(s)
        escaped = re.escape(s0).replace("\\ ", r"[\s\-]+")
        pat = re.compile(r"(?i)\b" + escaped + r"\b")
        for m in pat.finditer(t):
            hits.append((m.start(), m.end(), t[m.start():m.end()]))
    hits.sort(key=lambda x: x[0])
    return hits


# =========================
# DISPLAY policy: reduce noise for QA
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

def is_amenity_issue(issue: Dict[str, Any]) -> bool:
    t = (issue.get("issue_type") or "").lower()
    return ("amenity" in t) or ("amenities" in t) or ("shared vs private" in t) or ("title amenity" in t)

def _clean_amenity_hint(s: str) -> str:
    x = normalize_ws(s).lower()
    x = re.sub(r"^\s*\d+\s*[x×]\s*", "", x)
    x = re.sub(r"^\s*\d+\s+", "", x)
    x = re.sub(r"\b(shared|private|in\s+building|upon\s+request|available\s+upon\s+request)\b", "", x)
    x = normalize_ws(x)
    return x.strip()

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
        if not is_amenity_issue(it):
            out.append(it)
            continue
        itype = (it.get("issue_type") or "").lower()
        if "title amenity stuffing" in itype:
            out.append(it)
            continue
        canon = guess_issue_amenity_canon(it)
        if canon and canon in important:
            out.append(it)
    return out

def claim_present_in_selected_amenities(claim: str, amenities_selected: List[str]) -> bool:
    if not claim:
        return False
    for a in amenities_selected or []:
        if amenity_soft_match(claim, a):
            return True
    return False


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

def detect_structured_sum_from_dict(
    flat: Dict[str, Any],
    dict_key_hint_terms: List[str],
    value_key_exclude: List[str],
) -> Tuple[Optional[int], Optional[str]]:
    hint_terms = set([normalize_key(x) for x in dict_key_hint_terms])

    for k, v in flat.items():
        if not isinstance(v, dict):
            continue
        keys_norm = [normalize_key(str(x)) for x in v.keys()]
        keys_join = " ".join(keys_norm)

        if not any(t in keys_join for t in hint_terms):
            continue

        total = 0
        found_any = False
        for kk, vv in v.items():
            kn = normalize_key(str(kk))
            if any(ex in kn for ex in value_key_exclude):
                continue
            n = parse_int_maybe(vv)
            if n is not None:
                total += n
                found_any = True

        if found_any and total > 0:
            return total, k

    return None, None

def extract_explicit_count_claims(
    text: str,
    explicit_patterns: List[re.Pattern],
    exclude_patterns: Optional[List[re.Pattern]] = None,
) -> List[Tuple[int, str]]:
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

def extract_summed_component_claim(text: str, component_pattern: re.Pattern, min_components: int = 2) -> Optional[Tuple[int, str]]:
    t = text or ""
    total = 0
    anchor = None
    components = 0

    for m in component_pattern.finditer(t):
        raw = m.group(1)
        n = parse_int_maybe(raw)
        if n is None:
            continue
        total += n
        components += 1
        if anchor is None:
            anchor = (m.group(0) or "").strip()

    if total > 0 and components >= min_components and anchor:
        return total, anchor
    return None

def run_count_rules(flat: Dict[str, Any], corpus: Dict[str, str], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for rule in rules:
        structured_value, structured_key = detect_structured_count(
            flat=flat,
            key_hints=rule["structured_key_hints"],
            key_exclude=rule.get("structured_key_exclude", []),
        )

        if structured_value is None and rule.get("dict_component_terms"):
            structured_value, structured_key = detect_structured_sum_from_dict(
                flat=flat,
                dict_key_hint_terms=rule["dict_component_terms"],
                value_key_exclude=rule.get("dict_value_exclude", []),
            )

        if structured_value is None:
            continue

        for field, text in corpus.items():
            claims = extract_explicit_count_claims(
                text=text,
                explicit_patterns=rule["text_explicit_patterns"],
                exclude_patterns=rule.get("text_exclude_patterns", []),
            )

            if rule.get("text_component_pattern"):
                summed = extract_summed_component_claim(text, rule["text_component_pattern"])
                if summed:
                    claims.append(summed)

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
        "label": "total beds",
        "issue_type": "Bed count mismatch",
        "severity": "high",
        "structured_key_hints": ["total_beds", "beds_count", "bed_count", "num_beds", "number_of_beds"],
        "structured_key_exclude": ["bedroom", "bedrooms"],
        "dict_component_terms": ["king", "queen", "double", "full", "twin", "single", "sofa", "bunk", "murphy", "crib"],
        "dict_value_exclude": ["room", "bedroom"],
        "text_explicit_patterns": [re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*beds?\b")],
        "text_exclude_patterns": [re.compile(r"(?i)\bbedrooms?\b")],
        "text_component_pattern": re.compile(
            r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(king|queen|double|full|twin|single|sofa bed|sofabed|bunk|murphy|crib)\b"
        ),
    },
    {
        "label": "bedrooms",
        "issue_type": "Room count mismatch",
        "severity": "high",
        "structured_key_hints": ["bedrooms", "bedroom", "num_bedrooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+bedrooms?\b")],
        "text_exclude_patterns": [],
    },
    {
        "label": "bathrooms",
        "issue_type": "Bathroom count mismatch",
        "severity": "high",
        "structured_key_hints": ["bathrooms", "bathroom", "baths", "num_bathrooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(?:bathrooms?|baths?)\b")],
        "text_exclude_patterns": [],
    },
    {
        "label": "parking spaces",
        "issue_type": "Parking capacity mismatch",
        "severity": "medium",
        "structured_key_hints": ["parking_spaces", "parking_spots", "parking", "spaces", "spots"],
        "structured_key_exclude": ["policy", "rules", "allowed", "fee", "fees"],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(?:parking\s+spaces|parking\s+spots)\b"),
            re.compile(r"(?i)\bpark(?:ing)?\s+for\s+(\d{1,3}|[A-Za-z]+)\s+cars?\b"),
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
    """
    Post-process LLM issues:
      - drop ungrounded
      - locate correct field
      - amenity rule: do NOT keep amenity issues if the amenity is NOT listed in included or not_included
    """
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
            if claim and claim_present_in_selected_amenities(claim, amenities_included):
                continue

        needle = evidence if evidence else claim
        if not needle.strip():
            out.append(it)
            continue

        located_field = _find_best_field_for_evidence(needle, corpus)
        if located_field is None:
            if ("amenit" in issue_type.lower() or "amenities" in issue_type.lower()):
                canon_guess = guess_issue_amenity_canon(it)
                if canon_guess and canon_guess in listed_canons:
                    it["field"] = "amenities"
                    out.append(it)
            continue

        it["field"] = located_field

        if evidence and claim:
            if _find_best_field_for_evidence(evidence, corpus) is None and _find_best_field_for_evidence(claim, corpus) is not None:
                it["evidence"] = claim

        out.append(it)

    return out


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

def deterministic_required_amenity_checks(
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
) -> List[Dict[str, Any]]:
    """
    KEY RULE (your request):
      - Do NOT flag amenity inaccuracies if the amenity is not listed in either list.
      - Only enforce amenity consistency for amenities present in amenities_included or amenities_not_included.
    """
    issues: List[Dict[str, Any]] = []
    display, high_set, _low_set = all_known_amenities()

    included_canons = {canon_amenity(a) for a in (amenities_included or [])}
    not_included_canons = {canon_amenity(a) for a in (amenities_not_included or [])}
    listed_canons = included_canons | not_included_canons

    corpus = {"title": title, **texts}
    surfaced = set(high_set)

    for canon in sorted(surfaced):
        best_mention = None  # (field, evidence_substring)
        max_count = None
        max_field = None
        max_evidence = None

        for field, text in corpus.items():
            hits = find_amenity_hits(text, canon)
            if hits and best_mention is None:
                best_mention = (field, hits[0][2])

            if hits:
                for s, e, _ in hits:
                    n = extract_number_near(text, s)
                    if n is not None and n >= 2:
                        if max_count is None or n > max_count:
                            max_count = n
                            max_field = field
                            max_evidence = (text[max(0, s-35):min(len(text), e+35)]).strip()

        # Mentioned in text but NOT included -> only flag if amenity is LISTED somewhere
        if best_mention is not None and canon not in included_canons:
            if canon not in listed_canons:
                continue  # <-- your rule
            f, ev = best_mention
            gt_label = "Listed as NOT included" if canon in not_included_canons else "Not in included list"
            issues.append({
                "issue_type": "Amenity mentioned but not selected",
                "severity": "high",
                "field": f,
                "claim": display.get(canon, canon),
                "ground_truth": gt_label,
                "evidence": ev,
                "reason": f"{f} mentions '{display.get(canon, canon)}' but it is not in amenities_included."
            })

        # Count mismatch only if included (selected)
        if max_count is not None and canon in included_canons:
            issues.append({
                "issue_type": "Amenity count mismatch",
                "severity": "medium",
                "field": max_field or "text",
                "claim": f"{max_count}× {display.get(canon, canon)}",
                "ground_truth": f"Included amenities has '{display.get(canon, canon)}' (single selection)",
                "evidence": max_evidence or display.get(canon, canon),
                "reason": f"{max_field or 'Text'} suggests {max_count} {display.get(canon, canon)}(s), but amenities_included only indicates it is selected once."
            })

    # Selected-but-not-mentioned remains valid because it IS listed (included)
    combined_text = " ".join([title] + list(texts.values())).lower()
    for canon in sorted(included_canons):
        if canon not in surfaced:
            continue
        if not find_amenity_hits(combined_text, canon):
            issues.append({
                "issue_type": "Amenity selected but not mentioned",
                "severity": "medium",
                "field": "amenities",
                "claim": display.get(canon, canon),
                "ground_truth": "Selected in amenities_included",
                "evidence": display.get(canon, canon),
                "reason": f"amenities_included contains '{display.get(canon, canon)}' but it is not mentioned in the title/text fields."
            })

    # Title stuffing unchanged
    high_in_title = []
    for canon in sorted(high_set):
        if find_amenity_hits(title, canon):
            high_in_title.append(display.get(canon, canon))
    if len(high_in_title) > 3:
        issues.append({
            "issue_type": "Title amenity stuffing",
            "severity": "medium",
            "field": "title",
            "claim": f"{len(high_in_title)} high-priority amenities",
            "ground_truth": "<= 3 high-priority amenities in title",
            "evidence": ", ".join(high_in_title[:6]),
            "reason": f"Title includes {len(high_in_title)} high-priority amenities. Flag titles that contain more than 3."
        })

    return issues

def run_llm_checker(
    flat: Dict[str, Any],
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
    exclusive_keys: List[str],
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode_client = openai_client(api_key)

    gt_pairs = build_ground_truth_pairs(flat)
    exclusive_payload = [{"key": k, "value": str(flat.get(k, ""))} for k in exclusive_keys]

    payload = {
        "ground_truth_pairs": gt_pairs,
        "amenities_included": amenities_included,
        "amenities_not_included": amenities_not_included,
        "text_fields": {"title": title, **texts},
        "high_intent_amenities": HIGH_INTENT_AMENITIES,
        "low_priority_amenities": LOW_PRIORITY_AMENITIES,
        "exclusive_fields": exclusive_payload,
        "policy_notes": [
            "Only flag amenity inconsistencies if the amenity is listed in amenities_included or amenities_not_included.",
            "If an amenity is not listed in either list, do NOT flag it as missing/incorrect.",
        ],
        "pitfalls": [
            "Sometimes numbers refer to parking capacity or hot tub capacity; do NOT treat those as max guest capacity.",
        ],
        "required_issue_types": [
            "Max capacity mismatch",
            "Room count mismatch",
            "Bathroom count mismatch",
            "Sleeping arrangement mismatch",
            "Shared vs private amenity mismatch",
            "Amenities mentioned in text but not selected (or vice-versa)",
            "Pet friendliness conflict",
            "Property type inconsistency",
            "Extra guest fee inconsistency",
            "Exclusive text leakage",
        ],
        "output_schema": {
            "field_mapping": "best-effort mapping of canonical fields to JSON keys/values",
            "issues": "list of issues with dynamic reasons",
        }
    }

    system = (
        "You are a discrepancy checker. Return ONLY valid JSON with keys: field_mapping, issues.\n"
        "Be conservative: only flag when evidence clearly supports the claim.\n"
        "AMENITIES RULE: Only flag amenity inconsistencies for amenities present in amenities_included or amenities_not_included.\n"
        "If an amenity is not listed in either list, do NOT flag it.\n"
        "Ignore numbers that refer to parking capacity or hot tub capacity when assessing max guests.\n"
        "Exclusive rule: house_rules content should only exist in house_rules and not in other text fields.\n"
        "Each issue must include: issue_type, severity(high/medium/low), field, reason, evidence, claim(optional), ground_truth(optional)."
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

    merged = []
    seen = set()
    for it in (llm_issues + det_issues + count_issues):
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

        if field == "title":
            base = title
        elif field == "amenities":
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
    use_sample = st.sidebar.checkbox("Use sample JSON", value=False)

    st.sidebar.header("LLM settings (hard-coded)")
    st.sidebar.write(f"Model: `{OPENAI_MODEL}`")

    run_live = st.sidebar.checkbox("Re-check automatically while editing", value=True)

    if use_sample and not upload:
        data = {
            "title": "Cozy villa with hot tub — sleeps 4",
            "max_guests": 6,
            "bedrooms": 2,
            "bathrooms": 1,
            "property_type": "apartment",
            "amenities_included": ["Hot tub", "Wifi", "BBQ grill"],
            "amenities_not_included": ["Pool", "Sauna"],
            "house_rules": {"During your stay": ["No parties", "Quiet hours after 10pm"]},
            "summary": "A stylish apartment. Extra guest fee applies.",
            "the_space": "Sleeps 4. Hot tub. BBQ available.",
            "reviews": [{"text": "Great place!"}, {"text": "Loved it."}]
        }
    elif upload:
        try:
            data = load_json_upload(upload)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            return
    else:
        st.info("Upload a JSON file (or enable sample JSON).")
        return

    if isinstance(data, list):
        st.warning("JSON root is a list. Using first item.")
        data = data[0] if data else {}

    if not isinstance(data, dict):
        st.error("JSON root must be an object (dict).")
        return

    flat = flatten_json(data)

    # Amenities (new keys)
    amenities_included, amenities_not_included = extract_amenities_included_not_included(data)

    # House rules text (for exclusive field + QA text box)
    house_rules_obj = find_node_by_key(data, "house_rules")
    house_rules_text_default = house_rules_to_text(house_rules_obj)
    flat["house_rules"] = house_rules_text_default
    house_rules_key = "house_rules"
    exclusive_keys = [house_rules_key]

    title_key = detect_title_key(flat)

    # Remove description from UI
    text_keys = choose_editable_text_keys(flat, title_key=title_key, house_rules_key=house_rules_key)
    text_keys = [k for k in text_keys if normalize_key(k) != "description"]

    readonly_reviews = build_readonly_reviews(flat)
    title_val = str(flat.get(title_key, "")) if title_key else ""

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

        if readonly_reviews:
            st.subheader("Read-only review fields (not editable)")
            for k, txt in readonly_reviews.items():
                try:
                    st.text_area(k, value=txt, height=160, key=f"__ro_{k}", disabled=True)
                except TypeError:
                    st.markdown(f"**{k} (read-only)**")
                    st.code(txt[:4000])

        # ✅ House rules textbox ONLY (below reviews) — per your request
        st.subheader("House rules (text box for QA testing)")
        house_rules_text_edit = st.text_area(
            "House rules",
            value=house_rules_text_default,
            height=220,
            key="__house_rules_edit",
            help="Edit here to test the checker. Used in the LLM payload as the exclusive house_rules field.",
        )

    all_texts_for_checking = {**edited_texts, **readonly_reviews}

    def run_once():
        flat["house_rules"] = str(house_rules_text_edit or "")

        issues, mapping = run_llm_checker(
            flat=flat,
            title=title_edit,
            texts=all_texts_for_checking,
            amenities_included=amenities_included,
            amenities_not_included=amenities_not_included,
            exclusive_keys=exclusive_keys,
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

    show_all_issues = st.sidebar.checkbox(
        "Show all issues (including low-importance amenities)",
        value=False
    )

    issues_display = filter_issues_for_qa(issues, show_all=show_all_issues)
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
