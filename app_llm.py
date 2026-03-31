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
        "High chair", "Children's books and toys", "Baby bath", "Baby monitor",
        "Baby safety gates", "Changing table", "Pack 'n play / Travel crib",
        "Babysitter recommendations", "Children's dinnerware",
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
    """
    Case-insensitive highlight.
    If exact needle isn't found, highlight the longest meaningful token found in text.
    """
    t = text or ""
    n = (needle or "").strip()
    if not t:
        return "<em>(empty)</em>"
    if not n:
        return escape_html(t[:max_chars])

    tl = t.lower()
    nl = n.lower()

    idx = tl.find(nl)
    if idx < 0:
        # fallback: try longest token
        toks = [x for x in re.findall(r"[A-Za-z0-9\.\/\-]{3,}", n) if len(x) >= 3]
        toks.sort(key=len, reverse=True)
        for tok in toks[:6]:
            tidx = tl.find(tok.lower())
            if tidx >= 0:
                n = tok
                nl = tok.lower()
                idx = tidx
                break

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
# NUMBER PARSING (INT + FLOAT)
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
        # keep int for integer rules
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return None
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"\d{1,4}", s):
            return int(s)
        m = re.search(r"\b(\d{1,4})\b", s)
        if m:
            return int(m.group(1))
        if re.fullmatch(r"[A-Za-z]+", s):
            return word_to_int(s)
        m2 = re.search(r"\b([A-Za-z]+)\b", s)
        if m2:
            return word_to_int(m2.group(1))
    return None

def parse_number_maybe(x: Any) -> Optional[float]:
    """Parses decimals like 6.5 from both structured and text."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        m = re.search(r"\b(\d{1,4}(?:\.\d{1,2})?)\b", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        # words only -> int
        if re.fullmatch(r"[A-Za-z]+", s):
            v = word_to_int(s)
            return float(v) if v is not None else None
    return None


# =========================
# FUZZY TEXT MATCHING (NEW)
# =========================

def normalize_text_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, remove punctuation, normalize whitespace"""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', (text or '').lower())).strip()

def stem_word(word: str) -> str:
    """Basic stemming: remove trailing 's' for plural handling"""
    return word.rstrip('s') if len(word) > 2 else word

def fuzzy_text_contains(needle: str, haystack: str) -> bool:
    """
    Check if needle appears in haystack with fuzzy matching.
    Handles: punctuation variations, plurals, case differences.
    """
    if not needle or not haystack:
        return False
    
    norm_needle = normalize_text_for_matching(needle)
    norm_haystack = normalize_text_for_matching(haystack)
    
    # Direct substring match after normalization
    if norm_needle in norm_haystack:
        return True
    
    # Word-by-word matching with stemming
    needle_words = [stem_word(w) for w in norm_needle.split()]
    haystack_words = [stem_word(w) for w in norm_haystack.split()]
    
    # All needle words (stemmed) must appear in haystack
    return all(
        any(needle_word == haystack_word for haystack_word in haystack_words)
        for needle_word in needle_words
    )


# =========================
# AMENITY MATCHING (IMPROVED)
# =========================

# Expanded synonyms to handle access-related amenities and parking variations
AMENITY_SYNONYMS = {
    "hot tub": ["hot tub", "hot tubs", "jacuzzi", "spa tub", "whirlpool", "spa"],
    "pool": ["pool", "pools", "swimming pool", "swimming pools", "pool access"],
    "bbq grill": ["bbq", "barbecue", "bbq grill", "grill", "grills"],
    "dedicated workspace": ["dedicated workspace", "workspace", "work desk", "desk"],
    "ev charger": ["ev charger", "electric vehicle charger", "tesla charger"],
    "fire pit": ["fire pit", "fire-pit", "fire pits"],
    "gym": ["gym", "gyms", "fitness center", "fitness centre"],
    "sauna": ["sauna", "saunas"],
    "pool table": ["pool table", "billiards", "billiard table"],
    "ping pong table": ["ping pong table", "table tennis"],
    # Access-related amenities
    "lake access": ["lake access", "access to lake", "access to the lake", "lakefront", "lake front"],
    "beach access": ["beach access", "access to beach", "access to the beach", "beachfront", "beach front"],
    "resort access": ["resort access", "access to resort", "access to the resort", "resort amenities", "resort facilities", "resort"],
    "waterfront": ["waterfront", "water front"],
    # Parking variations
    "free parking on premises": ["free parking on premises", "free parking", "parking on site", "onsite parking", "on-site parking", "complimentary parking", "parking"],
    "free street parking": ["free street parking", "street parking"],
    "paid parking on premises": ["paid parking on premises", "paid parking", "parking"],
}

_AMENITY_MODIFIER_TOKENS = {
    "shared", "private", "in", "the", "a", "an", "and",
    "building", "premises", "upon", "request", "available",
    "paid", "free", "street", "on", "off",
}

def normalize_amenity_phrase(s: str) -> str:
    t = (s or "").lower()
    t = t.replace("'", "'")
    t = re.sub(r"[–—]", " ", t)
    t = re.sub(r"[^a-z0-9\s/]+", " ", t)
    return normalize_ws(t)

def amenity_tokens(s: str) -> List[str]:
    t = normalize_amenity_phrase(s)
    toks = [x for x in re.split(r"[\s/]+", t) if x]
    toks = [x for x in toks if x not in _AMENITY_MODIFIER_TOKENS]
    return toks

def _whole_word_contains(hay: str, needle: str) -> bool:
    """Word-boundary contains to prevent kitchen matching kitchenette."""
    if not hay or not needle:
        return False
    pat = re.compile(r"(?i)\b" + re.escape(needle) + r"\b")
    return bool(pat.search(hay))

def amenity_soft_match(a: str, b: str) -> bool:
    """
    Improved variation tolerant matching using fuzzy logic.
    Now handles: "free parking" vs "parking", "lake access" vs "access", "Lake view" vs "Lake Views!"
    """
    if not a or not b:
        return False
    
    # Use fuzzy matching first
    if fuzzy_text_contains(a, b) or fuzzy_text_contains(b, a):
        return True
    
    a_norm = normalize_amenity_phrase(a)
    b_norm = normalize_amenity_phrase(b)

    if a_norm == b_norm:
        return True

    # Whole-word contain (not inside another word)
    if _whole_word_contains(a_norm, b_norm) or _whole_word_contains(b_norm, a_norm):
        return True

    # Token-based matching with improved logic
    at = set(amenity_tokens(a_norm))
    bt = set(amenity_tokens(b_norm))
    if not at or not bt:
        return False
    
    # If all core tokens of one are in the other, it's a match
    if at.issubset(bt) or bt.issubset(at):
        return True
    
    # For access-related amenities, check if the core amenity word is present
    access_variants = ["access", "front"]
    for variant in access_variants:
        if variant in at or variant in bt:
            # Remove access/front tokens and check if remaining tokens match
            at_core = at - {variant, "to", "the"}
            bt_core = bt - {variant, "to", "the"}
            if at_core and bt_core and (at_core.issubset(bt_core) or bt_core.issubset(at_core)):
                return True
    
    return False

def canon_amenity(a: str) -> str:
    """Convert amenity to canonical form, handling variations better"""
    a0 = normalize_amenity_phrase(a)
    
    # First try exact synonym matches
    for canon, syns in AMENITY_SYNONYMS.items():
        for s in syns:
            s0 = normalize_amenity_phrase(s)
            if a0 == s0:
                return canon
    
    # Then try whole-word contains for more flexible matching
    for canon, syns in AMENITY_SYNONYMS.items():
        for s in syns:
            s0 = normalize_amenity_phrase(s)
            if _whole_word_contains(a0, s0):
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

def selectable_amenity_set() -> set:
    """Your "selectable Airbnb amenities" proxy: union of HIGH + LOW dictionaries."""
    _display, high, low = all_known_amenities()
    return set(high) | set(low)

def _regex_for_phrase(phrase: str) -> re.Pattern:
    """
    Improved regex generation that's less strict about exact spacing.
    """
    s0 = normalize_amenity_phrase(phrase)
    escaped = re.escape(s0)
    # Allow flexible spacing
    escaped = escaped.replace("\\ ", r"\s+")
    escaped = escaped.replace("\\/", r"[\s]*\/[\s]*")
    return re.compile(r"(?i)\b" + escaped + r"\b")

def find_amenity_hits(text: str, canon: str) -> List[Tuple[int, int, str]]:
    """
    Find amenity mentions in text with improved fuzzy matching.
    Now correctly handles "Lake view" vs "Lake Views!" and prevents false positives.
    """
    t = text or ""
    syns = AMENITY_SYNONYMS.get(canon, [canon])
    hits = []
    
    for s in syns:
        # Multi-word amenities: match full phrase with fuzzy logic
        if " " in s:
            # Try exact pattern first
            pat = _regex_for_phrase(s)
            for m in pat.finditer(t):
                hits.append((m.start(), m.end(), t[m.start():m.end()]))
            
            # If no exact match, try fuzzy matching
            if not hits:
                # Normalize both the synonym and text
                norm_syn = normalize_text_for_matching(s)
                norm_text = normalize_text_for_matching(t)
                syn_words = [stem_word(w) for w in norm_syn.split()]
                
                # Search for the phrase in the normalized text
                if norm_syn in norm_text:
                    # Find the position in the original text
                    idx = norm_text.find(norm_syn)
                    # Map back to original text position (approximate)
                    char_count = 0
                    for i, char in enumerate(t):
                        if char.isalnum() or char.isspace():
                            if char_count == idx:
                                start = i
                                break
                            if normalize_text_for_matching(char):
                                char_count += 1
                    else:
                        start = 0
                    
                    # Find end position
                    end = start + len(s)
                    # Adjust to find actual end in original text
                    while end < len(t) and not t[end-1].isalnum():
                        end -= 1
                    
                    if start < end:
                        hits.append((start, end, t[start:end]))
        else:
            # Single-word amenities: strict word boundaries with plural support
            pat = re.compile(r"(?i)\b" + re.escape(s) + r"s?\b")
            for m in pat.finditer(t):
                # Ensure not part of compound word
                match_start = m.start()
                match_end = m.end()
                
                if match_start > 0 and t[match_start - 1].isalnum():
                    continue
                if match_end < len(t) and t[match_end].isalnum():
                    continue
                    
                hits.append((match_start, match_end, t[match_start:match_end]))
    
    hits.sort(key=lambda x: x[0])
    return hits


# =========================
# QUALIFIER: PRIVATE vs SHARED (pool/hot tub)
# =========================

QUALIFIED_CANONS = {"pool", "hot tub"}

def qualifier_from_structured_amenities(amenities_included: List[str], canon: str) -> Optional[str]:
    """Return 'shared'/'private' if amenity string contains it."""
    for a in amenities_included or []:
        if amenity_soft_match(a, canon) or canon_amenity(a) == canon:
            a0 = normalize_amenity_phrase(a)
            if _whole_word_contains(a0, "shared"):
                return "shared"
            if _whole_word_contains(a0, "private"):
                return "private"
    return None

def qualifier_from_text(text: str, canon: str) -> Optional[str]:
    """
    Look for 'private/shared' near the amenity mention.
    """
    t = text or ""
    hits = find_amenity_hits(t, canon)
    if not hits:
        return None

    # take earliest hit and inspect window around it
    s, e, _ = hits[0]
    w_start = max(0, s - 120)
    w_end = min(len(t), e + 60)
    window = t[w_start:w_end].lower()

    # distance heuristic: choose closest qualifier token to the mention
    def nearest(q: str) -> Optional[int]:
        best = None
        for m in re.finditer(rf"\b{q}\b", window):
            dist = abs((w_start + m.start()) - s)
            if best is None or dist < best:
                best = dist
        return best

    d_priv = nearest("private")
    d_shared = nearest("shared")

    if d_priv is None and d_shared is None:
        return None
    if d_priv is None:
        return "shared"
    if d_shared is None:
        return "private"
    return "private" if d_priv <= d_shared else "shared"


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
    "Entertainment & Experience",  # include pool table etc.
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

        # always keep contradiction + qualifier mismatches
        if itype.startswith("amenity list contradiction") or "shared vs private" in itype:
            out.append(it)
            continue

        if not is_amenity_issue(it):
            out.append(it)
            continue

        # keep title stuffing
        if "title amenity stuffing" in itype:
            out.append(it)
            continue

        canon = guess_issue_amenity_canon(it)
        if canon and canon in important:
            out.append(it)

    return out


# =========================
# COUNT RULE ENGINE (with "total vs subset" scoring + float support)
# =========================

_SUBSET_CONTEXT = re.compile(
    r"(?i)\b(first floor|second floor|third floor|upstairs|downstairs|main level|ground floor|on the .* floor|including|located|conveniently)\b"
)
_TOTAL_CONTEXT = re.compile(
    r"(?i)\b(\d+\s*[- ]?\s*bedroom(?:s)?\s*(home|house|villa|property|apartment)?)\b|\b(\d+)\s*br\b"
)

def detect_structured_count(flat: Dict[str, Any], key_hints: List[str], key_exclude: List[str], value_type: str) -> Tuple[Optional[float], Optional[str]]:
    best_key = None
    best_val: Optional[float] = None
    best_score = -1

    for k, v in flat.items():
        kn = normalize_key(k)
        if any(h in kn for h in key_hints) and not any(ex in kn for ex in key_exclude):
            n = parse_number_maybe(v) if value_type == "float" else (float(parse_int_maybe(v)) if parse_int_maybe(v) is not None else None)
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

def extract_count_claims_with_pos(text: str, patterns: List[re.Pattern], value_type: str) -> List[Tuple[float, str, int, int]]:
    t = text or ""
    out: List[Tuple[float, str, int, int]] = []
    for pat in patterns:
        for m in pat.finditer(t):
            ev = (m.group(0) or "").strip()
            if not ev:
                continue
            raw = None
            if m.lastindex and m.lastindex >= 1:
                raw = m.group(1)
            if raw is None:
                mm = re.search(r"(\d{1,4}(?:\.\d{1,2})?|[A-Za-z]+)", ev)
                raw = mm.group(1) if mm else None

            n = parse_number_maybe(raw) if value_type == "float" else (float(parse_int_maybe(raw)) if parse_int_maybe(raw) is not None else None)
            if n is None:
                continue
            out.append((n, ev, m.start(), m.end()))
    return out

def choose_best_claim(claims: List[Tuple[float, str, int, int]], structured_value: float, value_type: str, tol: float) -> Optional[Tuple[float, str]]:
    if not claims:
        return None

    scored = []
    for n, ev, s, e in claims:
        score = 0

        # total-context signals
        if _TOTAL_CONTEXT.search(ev):
            score += 5
        if "-" in ev and "bedroom" in ev.lower():
            score += 3
        if re.search(r"(?i)\bhome\b|\bhouse\b|\bvilla\b|\bproperty\b|\bapartment\b", ev):
            score += 2

        # subset-context penalty
        if _SUBSET_CONTEXT.search(ev):
            score -= 6

        # matching structured gets a bump
        if abs(n - structured_value) <= tol:
            score += 4

        # earlier is slightly preferred (tie-breaker)
        score -= (s / 10000.0)

        scored.append((score, n, ev))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    return (best[1], best[2])

def run_count_rules(flat: Dict[str, Any], corpus: Dict[str, str], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for rule in rules:
        value_type = rule.get("value_type", "int")
        tol = float(rule.get("tolerance", 0.0))

        structured_value, structured_key = detect_structured_count(
            flat=flat,
            key_hints=rule["structured_key_hints"],
            key_exclude=rule.get("structured_key_exclude", []),
            value_type=value_type,
        )
        if structured_value is None:
            continue

        for field, text in corpus.items():
            claims = extract_count_claims_with_pos(text, rule["text_explicit_patterns"], value_type=value_type)
            best = choose_best_claim(claims, structured_value, value_type=value_type, tol=tol)
            if not best:
                continue

            n, ev = best
            if abs(n - structured_value) > tol:
                gt_str = f"{structured_value:g} (from {structured_key})" if structured_key else f"{structured_value:g}"
                issues.append({
                    "issue_type": rule["issue_type"],
                    "severity": rule.get("severity", "high"),
                    "field": field,
                    "claim": f"{n:g}",
                    "ground_truth": gt_str,
                    "evidence": ev,
                    "reason": f"{field} suggests {rule['label']} is {n:g}, but the structured value is {structured_value:g}."
                })

    return issues

COUNT_RULES = [
    {
        "label": "bedrooms",
        "issue_type": "Room count mismatch",
        "severity": "high",
        "value_type": "int",
        "tolerance": 0.0,
        "structured_key_hints": ["bedrooms", "bedroom", "num_bedrooms", "rooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*bedrooms?\b"),
            re.compile(r"(?i)\b(\d{1,3})\s*br\b"),
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*bedroom\s+home\b"),
        ],
    },
    {
        "label": "beds",
        "issue_type": "Bed count mismatch",
        "severity": "high",
        "value_type": "int",
        "tolerance": 0.0,
        "structured_key_hints": ["total_beds", "beds_count", "bed_count", "num_beds", "number_of_beds", "beds"],
        "structured_key_exclude": ["bedroom", "bedrooms"],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s*[- ]?\s*beds?\b"),
        ],
    },
    {
        "label": "bathrooms",
        "issue_type": "Bathroom count mismatch",
        "severity": "high",
        "value_type": "float",
        "tolerance": 0.01,
        "structured_key_hints": ["bathrooms", "bathroom", "baths", "num_bathrooms"],
        "structured_key_exclude": [],
        "text_explicit_patterns": [
            re.compile(r"(?i)\b(\d{1,3}(?:\.\d{1,2})?)\s+(?:bathrooms?|baths?)\b"),
            re.compile(r"(?i)\b(\d{1,3}(?:\.\d{1,2})?)\s*ba\b"),
        ],
    },
    {
        "label": "max guests",
        "issue_type": "Max capacity mismatch",
        "severity": "high",
        "value_type": "int",
        "tolerance": 0.0,
        "structured_key_hints": ["max_guests", "guests", "accommodates", "capacity"],
        "structured_key_exclude": ["min", "minimum"],
        "text_explicit_patterns": [
            re.compile(r"(?i)\bsleeps?\s+(\d{1,3}|[A-Za-z]+)\b"),
            re.compile(r"(?i)\baccommodates?\s+(\d{1,3}|[A-Za-z]+)\b"),
            re.compile(r"(?i)\bup to\s+(\d{1,3}|[A-Za-z]+)\s+guests?\b"),
            re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+guests?\s+maximum\b"),
        ],
    },
]


# =========================
# DETERMINISTIC AMENITY CHECKS (selectable-set scoped)
# =========================

def deterministic_amenity_checks(
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    display, high_set, low_set = all_known_amenities()

    selectable = selectable_amenity_set()
    included_canons = {canon_amenity(a) for a in (amenities_included or [])}
    not_included_canons = {canon_amenity(a) for a in (amenities_not_included or [])}

    corpus = {"title": title, **texts}
    title_is_full = _count_high_intent_in_title(title) >= 3

    # 3) Mentioned but not selected (only for selectable amenities; includes gym)
    # We only iterate selectable amenities we *care* about surfacing (high intent),
    # but this still correctly captures "gym missing" and avoids "tennis court".
    for canon in sorted(high_set):
        best_mention = None
        for field, text in corpus.items():
            hits = find_amenity_hits(text, canon)
            if hits:
                best_mention = (field, hits[0][2])
                break

        if best_mention and canon not in included_canons:
            f, ev = best_mention
            if normalize_key(f) == "title" and title_is_full:
                # title amenity inaccuracies suppressed when title already has >=3 high-intent
                continue
            gt_label = "Listed as NOT included" if canon in not_included_canons else "Not in amenities_included"
            issues.append({
                "issue_type": "Amenity mentioned but not selected",
                "severity": "high",
                "field": f,
                "claim": display.get(canon, canon),
                "ground_truth": gt_label,
                "evidence": ev,
                "reason": f"'{display.get(canon, canon)}' is mentioned but not selected. Found in: {f}."
            })

    # 6) shared vs private mismatch for pool/hot tub
    for canon in sorted(QUALIFIED_CANONS):
        struct_q = qualifier_from_structured_amenities(amenities_included, canon)
        if not struct_q:
            continue

        for field, text in corpus.items():
            txt_q = qualifier_from_text(text, canon)
            if txt_q and txt_q != struct_q:
                issues.append({
                    "issue_type": "Shared vs private amenity mismatch",
                    "severity": "high",
                    "field": field,
                    "claim": f"{txt_q} {canon}",
                    "ground_truth": f"{struct_q} {canon} (from amenities_included)",
                    "evidence": f"{txt_q}",
                    "reason": f"Text implies '{txt_q} {canon}', but structured amenities indicate '{struct_q} {canon}'. Found in: {field}."
                })
                break

    # 7/8) selected (high-intent) but not mentioned anywhere
    combined = " ".join([title] + [v for v in texts.values() if v]).lower()
    for canon in sorted(included_canons):
        if canon not in high_set:
            continue
        if not find_amenity_hits(combined, canon):
            issues.append({
                "issue_type": "Amenity selected but not mentioned",
                "severity": "medium",
                "field": "amenities_included",
                "claim": display.get(canon, canon),
                "ground_truth": "Selected in amenities_included",
                "evidence": display.get(canon, canon),
                "reason": f"'{display.get(canon, canon)}' is selected but never mentioned in title/text."
            })

    # title stuffing
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
            "reason": f"Title contains {len(high_in_title)} high-priority amenities."
        })

    return issues


# =========================
# LLM HELPERS + POST-PROCESSING (field grounding + evidence anchoring)
# =========================

def openai_client(api_key: str):
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=api_key)

def llm_json(client, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content or "{}"
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

def _anchor_evidence(issue: Dict[str, Any], corpus: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (field, evidence) where evidence is guaranteed to exist in that field if possible.
    """
    claim = str(issue.get("claim", "") or "").strip()
    evidence = str(issue.get("evidence", "") or "").strip()
    itype = str(issue.get("issue_type", "") or "")

    # 1) try evidence
    if evidence:
        f = _find_best_field_for_evidence(evidence, corpus)
        if f:
            return f, evidence

    # 2) try claim
    if claim:
        f = _find_best_field_for_evidence(claim, corpus)
        if f:
            return f, claim

    # 3) amenity fallback: find canonical hit in corpus
    if "amenit" in itype.lower():
        canon = guess_issue_amenity_canon(issue)
        if canon:
            for field, text in corpus.items():
                hits = find_amenity_hits(text, canon)
                if hits:
                    return field, hits[0][2]

    return None, None

def ground_and_locate_llm_issues(
    llm_issues: List[Dict[str, Any]],
    corpus: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in llm_issues or []:
        f, ev = _anchor_evidence(it, corpus)
        if not f or not ev:
            # drop ungrounded "phantom" issues
            continue
        it["field"] = f
        it["evidence"] = ev

        # fix generic reason
        r = str(it.get("reason", "") or "")
        if r and "listing text" in r.lower() and "found in" not in r.lower():
            it["reason"] = r + f" (Found in: {f})"
        elif r and "found in" not in r.lower():
            it["reason"] = r + f" (Found in: {f})"

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

def run_llm_checker(
    flat: Dict[str, Any],
    title: str,
    texts: Dict[str, str],
    amenities_included: List[str],
    amenities_not_included: List[str],
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    client = openai_client(api_key)

    payload = {
        "ground_truth_pairs": build_ground_truth_pairs(flat),
        "amenities_included": amenities_included,
        "amenities_not_included": amenities_not_included,
        "text_fields": {"title": title, **texts},
        "policy_notes": [
            "Only treat amenities as cross-checkable if they are in the selectable Airbnb amenity set (provided via categories lists).",
            "Do not flag non-selectable items (e.g., tennis courts) as amenity mismatches.",
            "TITLE RULE: Do not create title-level amenity inaccuracies if title already has 3+ high-intent amenities; only use 'Title amenity stuffing'.",
        ],
        "output_schema": {"field_mapping": "best-effort", "issues": "list"},
    }

    system = (
        "You are a discrepancy checker. Return ONLY valid JSON with keys: field_mapping, issues.\n"
        "Be conservative.\n"
        "For each issue, include: issue_type, severity, field, reason, evidence, claim(optional), ground_truth(optional).\n"
        "Do not hallucinate claims; evidence must be a substring of a provided field.\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    out = llm_json(client, model, messages)
    llm_issues_raw = out.get("issues") or []
    mapping = out.get("field_mapping") or {}

    corpus = build_corpus(title, texts)
    llm_issues = ground_and_locate_llm_issues(llm_issues_raw, corpus)

    det_issues = deterministic_amenity_checks(title, texts, amenities_included, amenities_not_included)
    count_issues = run_count_rules(flat=flat, corpus=corpus, rules=COUNT_RULES)
    contradiction_issues = detect_included_not_included_conflicts(amenities_included, amenities_not_included)

    merged = []
    seen = set()
    for it in (llm_issues + det_issues + contradiction_issues + count_issues):
        key = (
            str(it.get("issue_type", "")),
            str(it.get("field", "")),
            (str(it.get("evidence", ""))[:120]),
            (str(it.get("claim", ""))[:120]),
            (str(it.get("ground_truth", ""))[:120]),
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
        elif normalize_key(field) in {"amenities", "amenities_included", "amenities_not_included"}:
            base = combined_all
        else:
            base = texts.get(field, "")
            if not base:
                base = combined_all

        for it in sorted(items, key=lambda x: (sev_rank(x.get("severity")), x.get("issue_type", ""))):
            sev = (it.get("severity") or "low").lower()
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            with st.expander(f"{icon} {it.get('issue_type','Issue')} — {str(it.get('reason',''))[:110]}"):
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
        if fn == "the_space":
            return 1
        if fn == "summary":
            return 2
        if fn == "house_rules":
            return 3
        if fn.startswith("review"):
            return 6
        if fn.startswith("amenities"):
            return 9
        return 5

    buckets: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for it in issues:
        key = (
            norm(it.get("issue_type")),
            norm(it.get("claim")),
            norm(it.get("ground_truth")),
            norm(it.get("evidence"))[:120],
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
    st.title("JSON Discrepancy Checker — LLM semantic (FIXED)")

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

    selected_idx = None
    selected_id = ""

    # JSON can be list of listings
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
        selected_idx = sel[0]
        obj = data[selected_idx]
        if isinstance(obj, dict):
            selected_id = str(obj.get("target_listing_id") or "")
        data = obj

    if not isinstance(data, dict):
        st.error("Selected JSON item must be an object (dict).")
        return

    # Scope keys to prevent Streamlit state mixing across listings
    scope = f"{selected_idx}_{selected_id}".strip("_")

    flat = flatten_json(data)

    amenities_included, amenities_not_included = extract_amenities_included_not_included(data)

    title_key = detect_title_key(flat)
    title_val = str(flat.get(title_key, "")) if title_key else ""

    # Remove description from UI (per your earlier requirement)
    text_keys = choose_editable_text_keys(flat, title_key=title_key)
    text_keys = [k for k in text_keys if normalize_key(k) != "description"]

    readonly_reviews = build_readonly_reviews(flat)

    # House rules editable
    house_rules_obj = data.get("house_rules")
    house_rules_initial = house_rules_to_text(house_rules_obj)

    st.header("Editable listing text")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Title")
        title_edit = st.text_input("Title", value=title_val, key=f"__title_{scope}")

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
            edited_texts[k] = st.text_area(
                k,
                value=str(flat.get(k, "")),
                height=160,
                key=f"__txt_{scope}_{k}",
            )

        st.subheader("Read-only review fields (not editable)")
        if readonly_reviews:
            for k, txt in readonly_reviews.items():
                st.text_area(
                    k,
                    value=txt,
                    height=160,
                    key=f"__ro_{scope}_{k}",
                    disabled=True,
                )
        else:
            st.write("—")

        st.subheader("House rules (editable)")
        house_rules_edit = st.text_area(
            "house_rules",
            value=house_rules_initial,
            height=220,
            key=f"__txt_{scope}_house_rules",
            help="Editable so QA can test rule-related inaccuracies.",
        )
        edited_texts["house_rules"] = house_rules_edit

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
            "scope": scope,
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
