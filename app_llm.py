import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass  # Not in Streamlit

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Set it in Streamlit secrets or as an environment variable.")
    
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
        # Arrow conversion edge cases (mixed types)
        st.dataframe(df.astype(str))

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_key(k: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (k or "").lower()).strip("_")

def is_review_key(k: str) -> bool:
    kn = normalize_key(k)
    return ("review" in kn) or ("reviews" in kn)

def is_indexed_path(k: str) -> bool:
    return bool(re.search(r"(^|\.)\d+(\.|$)", k))

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

def detect_amenities_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if normalize_key(k) == "amenities" and isinstance(v, list) and all(isinstance(x, str) for x in v):
            return k
    for k, v in flat.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v) and "amenit" in normalize_key(k):
            return k
    return None

def detect_house_rules_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if isinstance(v, str) and "house_rules" in normalize_key(k) and len(v.strip()) >= 30:
            return k
    return None

def detect_text_keys(flat: Dict[str, Any], min_len: int = 120) -> List[str]:
    keys = []
    for k, v in flat.items():
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= min_len or ("\n" in s) or (len(re.split(r"(?<=[.!?])\s+", s)) >= 2 and len(s) >= 80):
                keys.append(k)
    return sorted(keys)

PREFERRED_TEXT_NORMAL_KEYS = {
    "summary", "the_space", "guest_access", "other_things_to_note", "description"
}

def choose_editable_text_keys(flat: Dict[str, Any], title_key: Optional[str], house_rules_key: Optional[str]) -> List[str]:
    # include preferred textual keys if present (even if short)
    preferred = []
    for k, v in flat.items():
        if isinstance(v, str) and normalize_key(k) in PREFERRED_TEXT_NORMAL_KEYS:
            preferred.append(k)

    # fallback/delta: add other long text keys
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


# =========================
# REVIEWS (these are also checked for inconsistencies)
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
# AMENITY (DETERMINISTIC)
# =========================

_NUM_WORDS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20
}
def word_to_int(w: str) -> Optional[int]:
    return _NUM_WORDS.get(re.sub(r"[^a-z]", "", (w or "").lower()))

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
    "hot tub": ["hot tub", "jacuzzi", "spa tub", "whirlpool"],
    "pool": ["pool", "swimming pool"],
    "bbq grill": ["bbq", "barbecue", "bbq grill", "grill"],
    "dedicated workspace": ["dedicated workspace", "workspace", "work desk", "desk"],
    "ev charger": ["ev charger", "electric vehicle charger", "tesla charger"],
    "game console": ["game console", "ps5", "playstation", "xbox", "nintendo switch"],
    "movie theater": ["movie theater", "home theater", "cinema room"],
    "fire pit": ["fire pit", "fire-pit"],
    "indoor fireplace": ["indoor fireplace", "fireplace"],
}

def canon_amenity(a: str) -> str:
    a0 = normalize_ws(a).lower()
    for canon, syns in AMENITY_SYNONYMS.items():
        if a0 == canon or a0 in [s.lower() for s in syns]:
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
        escaped = re.escape(s).replace("\\ ", r"[\s\-]+")
        pat = re.compile(r"(?i)\b" + escaped + r"\b")
        for m in pat.finditer(t):
            hits.append((m.start(), m.end(), t[m.start():m.end()]))
    hits.sort(key=lambda x: x[0])
    return hits


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


# =========================
# CORE CHECKER (LLM + REQUIRED DETERMINISTIC AMENITY RULES)
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
    amenities_selected: List[str],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    display, high_set, low_set = all_known_amenities()

    selected = {canon_amenity(a) for a in amenities_selected}
    corpus = {"title": title, **texts}

    for canon in sorted(high_set | low_set):
        mentioned = False
        max_count = None
        max_field = None
        max_evidence = None

        for field, text in corpus.items():
            hits = find_amenity_hits(text, canon)
            if hits:
                mentioned = True
                for s, e, _ in hits:
                    n = extract_number_near(text, s)
                    if n is not None and n >= 2:
                        if max_count is None or n > max_count:
                            max_count = n
                            max_field = field
                            max_evidence = (text[max(0, s-35):min(len(text), e+35)]).strip()

        if mentioned and canon not in selected:
            sev = "high" if canon in high_set else "low"
            issues.append({
                "issue_type": "Amenity mentioned but not selected",
                "severity": sev,
                "field": "title" if find_amenity_hits(title, canon) else "text",
                "claim": display.get(canon, canon),
                "ground_truth": "Not in Amenities list",
                "evidence": display.get(canon, canon),
                "reason": f"Text mentions '{display.get(canon, canon)}' but it is not present in the Amenities ground truth."
            })

        if max_count is not None and canon in selected:
            issues.append({
                "issue_type": "Amenity count mismatch",
                "severity": "medium" if canon in high_set else "low",
                "field": max_field or "text",
                "claim": f"{max_count}× {display.get(canon, canon)}",
                "ground_truth": f"Amenities includes '{display.get(canon, canon)}' (single selection)",
                "evidence": max_evidence or display.get(canon, canon),
                "reason": f"{max_field or 'Text'} suggests {max_count} {display.get(canon, canon)}(s), but the amenities ground truth only indicates the amenity is selected (no multiple units)."
            })

    combined_text = " ".join([title] + list(texts.values())).lower()
    for canon in sorted(selected):
        if not find_amenity_hits(combined_text, canon):
            sev = "medium" if canon in high_set else "low"
            issues.append({
                "issue_type": "Amenity selected but not mentioned",
                "severity": sev,
                "field": "amenities",
                "claim": display.get(canon, canon),
                "ground_truth": "Selected in Amenities list",
                "evidence": display.get(canon, canon),
                "reason": f"Amenities includes '{display.get(canon, canon)}' but it is not mentioned in the title/text fields."
            })

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
    amenities_selected: List[str],
    exclusive_keys: List[str],
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode_client = openai_client(api_key)

    gt_pairs = build_ground_truth_pairs(flat)
    exclusive_payload = [{"key": k, "value": str(flat.get(k, ""))} for k in exclusive_keys]

    payload = {
        "ground_truth_pairs": gt_pairs,
        "amenities_selected": amenities_selected,
        "text_fields": {"title": title, **texts},
        "high_intent_amenities": HIGH_INTENT_AMENITIES,
        "low_priority_amenities": LOW_PRIORITY_AMENITIES,
        "exclusive_fields": exclusive_payload,
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
        "Ignore numbers that refer to parking capacity or hot tub capacity when assessing max guests.\n"
        "Exclusive rule: house_rules content should only exist in house_rules and not in other text fields.\n"
        "Each issue must include: issue_type, severity(high/medium/low), field, reason, evidence, claim(optional), ground_truth(optional)."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    out = llm_json(mode_client, model, messages)
    llm_issues = out.get("issues") or []
    mapping = out.get("field_mapping") or {}

    det_issues = deterministic_required_amenity_checks(title, texts, amenities_selected)

    merged = []
    seen = set()
    for it in (llm_issues + det_issues):
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
# STREAMLIT UI
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

def as_display_value(v: Any) -> str:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return "" if v is None else str(v)

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
            "title": "Cozy villa with 2 hot tubs, private pool, sauna, and EV charger — sleeps 4",
            "max_guests": 6,
            "bedrooms": 2,
            "bathrooms": 1,
            "property_type": "apartment",
            "Amenities": ["Hot tub", "Shared pool", "Wifi"],
            "house_rules": "No pets. No parties. Quiet hours after 10pm.",
            "description": "Sleeps 4. Two hot tubs and a private pool. Pet friendly! Perfect villa getaway.",
            "summary": "A stylish apartment with shared pool. Extra guest fee applies.",
            "reviews": [{"text": "Great place!"}, {"text": "Loved the pool."}]
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

    # Auto-detect (no Field Selection UI)
    title_key = detect_title_key(flat)
    amenities_key = detect_amenities_key(flat)
    house_rules_key = detect_house_rules_key(flat)
    exclusive_keys = [house_rules_key] if house_rules_key else []
    text_keys = choose_editable_text_keys(flat, title_key=title_key, house_rules_key=house_rules_key)
    readonly_reviews = build_readonly_reviews(flat)

    title_val = str(flat.get(title_key, "")) if title_key else ""
    amenities_val = flat.get(amenities_key, []) if amenities_key else []
    if not isinstance(amenities_val, list):
        amenities_val = []
    amenities_val = [str(x) for x in amenities_val if isinstance(x, (str, int, float))]

    st.header("Editable listing text")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Title")
        title_edit = st.text_input("Title", value=title_val, key="__title")

        st.subheader("Amenities (ground truth)")
        st.write(amenities_val if amenities_val else "—")

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

    all_texts_for_checking = {**edited_texts, **readonly_reviews}

    def run_once():
        api_key = OPENAI_API_KEY
        model = OPENAI_MODEL

        issues, mapping = run_llm_checker(
            flat=flat,
            title=title_edit,
            texts=all_texts_for_checking,
            amenities_selected=amenities_val,
            exclusive_keys=exclusive_keys,
            api_key=api_key,
            model=model,
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
    mapping = current["mapping"]

    st.header("Results")

    counts = {"high": 0, "medium": 0, "low": 0}
    for it in issues:
        counts[(it.get("severity") or "low").lower()] = counts.get((it.get("severity") or "low").lower(), 0) + 1
    m1, m2, m3 = st.columns(3)
    m1.metric("High", counts.get("high", 0))
    m2.metric("Medium", counts.get("medium", 0))
    m3.metric("Low", counts.get("low", 0))

    st.subheader("Field mapping (LLM best-effort)")
    map_rows = []
    if isinstance(mapping, dict):
        for canon, d in mapping.items():
            if isinstance(d, dict):
                map_rows.append({
                    "canonical": str(canon),
                    "json_key": as_display_value(d.get("key")),
                    "value": as_display_value(d.get("value")),
                    "confidence": as_display_value(d.get("confidence")),
                })
            else:
                map_rows.append({"canonical": str(canon), "json_key": "", "value": as_display_value(d), "confidence": ""})
    safe_df(pd.DataFrame(map_rows))

    df = issues_to_df(issues).copy()
    df = df.sort_values(by=["severity", "issue_type"], key=lambda s: s.map(sev_rank), ascending=True)
    st.subheader("Issues table")
    safe_df(df)

    st.download_button("Download issues (CSV)", df.to_csv(index=False).encode("utf-8"), "issues.csv", "text/csv")
    st.download_button("Download issues (JSON)", json.dumps(issues, ensure_ascii=False, indent=2).encode("utf-8"), "issues.json", "application/json")

    st.subheader("Issue details (highlights + reasons)")
    render_issue_cards(issues, title_edit, all_texts_for_checking)

    st.sidebar.header("Runs")
    st.sidebar.write(f"{len(st.session_state.get('runs', []))} run(s) this session")
    if st.sidebar.button("Clear runs"):
        st.session_state["runs"] = []
        st.session_state["current"] = None
        st.rerun()


if __name__ == "__main__":
    main()
