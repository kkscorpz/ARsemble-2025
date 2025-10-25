import math
from typing import Any, Dict, Tuple, List
import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from components_data import data
from storage import load_json, save_json

import re
import unicodedata

# --------------------------
# Setup
# --------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Fallback")


# --- add: dataset summary for server startup debugging ---
try:
    logger.info("Dataset summary: cpu=%d gpu=%d motherboard=%d ram=%d storage=%d psu=%d cpu_cooler=%d",
                len(data.get("cpu", {})),
                len(data.get("gpu", {})),
                len(data.get("motherboard", {})),
                len(data.get("ram", {})),
                len(data.get("storage", {})),
                len(data.get("psu", {})),
                len(data.get("cpu_cooler", {})))
except Exception:
    logger.exception("Failed to log dataset summary")
# --- end add ---

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Fast, low-latency Gemini model


def slugify(text: str) -> str:
    """Simple safe slugify — replaces spaces and special chars with hyphens."""
    return re.sub(r'[^a-z0-9]+', '-', (text or "").lower().strip()).strip('-')


def _tokens(s: str) -> List[str]:
    # Use the global slugify helper to normalize text for tokenization
    s = slugify(s or "")
    return [t for t in s.split('-') if t]


def _has_numeric_token(tokens: List[str]) -> bool:
    return any(re.search(r"\d", t) for t in tokens)


def _normalize_text_for_match(s: str) -> str:
    """Normalize text for matching: lowercase, ascii-fold, remove non-alnum, collapse spaces."""
    if not s:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)  # replace punctuation with space
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _best_match_in_dataset(query: str, data: dict) -> Tuple[Dict[str, Any], str]:
    """
    Stricter fuzzy matcher with brand & numeric-model token agreement.

    Rules:
    - If the user query contains a brand/series token (asus, rog, tuf, msi, gigabyte, etc.),
      require the candidate's name to contain that brand/series token.
    - If the query contains numeric/model tokens (e.g., b550, z790, 5800x, 4070),
      require the candidate to share at least one numeric/model token (exact token or
      token-prefix match like 'b550-f' vs 'b550m' allowed).
    - Prefer exact/slug matches, then normalized substring, then token-subset, then scored overlap,
      but only accept scored fallback when score >= MIN_SCORE.
    - Adds debug logging for why a candidate was chosen.
    """
    try:
        if not query:
            return None, None

        q_slug = slugify(query)
        q_norm = _normalize_text_for_match(query)
        q_tokens = [t for t in re.split(r"\s+", q_norm) if t]

        # Identify numeric/model tokens and brand/series tokens in the query
        q_numeric_tokens = set(
            [t for t in q_tokens if re.search(r"\d", t)])  # e.g., b550, 5800x
        # short list of brand/series tokens we care about (extend as needed)
        brand_tokens = {
            "asus": ("asus", "rog", "strix", "tuf"),
            "msi": ("msi", "mpg", "carbon", "tomahawk", "mag"),
            "gigabyte": ("gigabyte", "aorus"),
            "asrock": ("asrock", "steel legend", "taichi"),
            "intel": ("intel", "lga"),
            "amd": ("amd", "ryzen", "am4", "am5")
        }
        # flatten brand token lookup for fast check
        q_brand_tokens = set()
        for k, toks in brand_tokens.items():
            for tok in toks:
                if tok in q_norm:
                    q_brand_tokens.add(tok)

        def candidate_tokens(name: str):
            n_norm = _normalize_text_for_match(name or "")
            n_tokens = [t for t in re.split(r"\s+", n_norm) if t]
            n_numeric = set([t for t in n_tokens if re.search(r"\d", t)])
            n_brand = set(
                t for t in n_tokens for group in brand_tokens.values() if t in group)
            return n_tokens, n_numeric, n_brand

        def numeric_matches(q_nums: set, n_nums: set) -> bool:
            if not q_nums:
                return True
            if not n_nums:
                return False
            # allow prefix matches like "b550-f" vs "b550"
            for qn in q_nums:
                for nn in n_nums:
                    if qn == nn:
                        return True
                    # allow 'b550-f' and 'b550plus' style variants if they share base token
                    if nn.startswith(qn) or qn.startswith(nn):
                        return True
            return False

        # Fast-path: slug substring (strict)
        for comp_type, comps in data.items():
            for obj in comps.values():
                name = obj.get("name") or ""
                name_slug = slugify(name)
                if name_slug and name_slug in q_slug:
                    # enforce brand/numeric if present in query
                    n_tokens, n_numeric, n_brand = candidate_tokens(name)
                    if q_brand_tokens and not (q_brand_tokens & n_brand):
                        continue
                    if not numeric_matches(q_numeric_tokens, n_numeric):
                        continue
                    logger.debug(
                        "best_match: slug match -> %s (%s)", name, comp_type)
                    return obj, comp_type

        # Normalized substring match
        for comp_type, comps in data.items():
            for obj in comps.values():
                name = obj.get("name") or ""
                name_norm = _normalize_text_for_match(name)
                if name_norm and name_norm in q_norm:
                    n_tokens, n_numeric, n_brand = candidate_tokens(name)
                    if q_brand_tokens and not (q_brand_tokens & n_brand):
                        continue
                    if not numeric_matches(q_numeric_tokens, n_numeric):
                        continue
                    logger.debug(
                        "best_match: normalized substring -> %s (%s)", name, comp_type)
                    return obj, comp_type

        # Token-subset (all name tokens must appear in query) — high confidence
        for comp_type, comps in data.items():
            for obj in comps.values():
                name = obj.get("name") or ""
                name_norm = _normalize_text_for_match(name)
                name_tokens = [t for t in re.split(r"\s+", name_norm) if t]
                if name_tokens and all(tok in q_tokens for tok in name_tokens):
                    n_tokens, n_numeric, n_brand = candidate_tokens(name)
                    if q_brand_tokens and not (q_brand_tokens & n_brand):
                        continue
                    if not numeric_matches(q_numeric_tokens, n_numeric):
                        continue
                    logger.debug(
                        "best_match: token subset -> %s (%s)", name, comp_type)
                    return obj, comp_type

        # Scoring fallback: token overlap + numeric bonus + brand bonus
        best_obj = None
        best_type = None
        best_score = 0.0
        MIN_SCORE = 3.0  # raise threshold to reduce bad matches

        q_token_set = set(q_tokens)
        for comp_type, comps in data.items():
            for obj in comps.values():
                name = obj.get("name") or ""
                name_norm = _normalize_text_for_match(name)
                name_tokens = [t for t in re.split(r"\s+", name_norm) if t]
                if not name_tokens:
                    continue
                n_tokens, n_numeric, n_brand = candidate_tokens(name)

                # brand enforcement: if query has brand tokens, require overlap
                if q_brand_tokens and not (q_brand_tokens & n_brand):
                    continue

                # numeric enforcement: if query has numeric tokens, require at least one match
                if q_numeric_tokens and not numeric_matches(q_numeric_tokens, n_numeric):
                    continue

                common = len(q_token_set & set(name_tokens))
                # numeric bonus: exact numeric token matches are important
                numeric_bonus = len(q_numeric_tokens & n_numeric) * 2
                brand_bonus = 1.5 if (q_brand_tokens & n_brand) else 0.0
                # prefer multi-token matches
                length_bonus = min(len(name_tokens), 3) * 0.2
                score = common + numeric_bonus + brand_bonus + length_bonus

                # penalize single-token name matches when query is multi-token (to avoid generic hits)
                if len(name_tokens) == 1 and len(q_tokens) > 1:
                    score -= 0.75

                if score > best_score:
                    best_score = score
                    best_obj = obj
                    best_type = comp_type

        if best_obj and best_score >= MIN_SCORE:
            logger.debug("best_match: scoring fallback -> %s (%s) score=%.2f",
                         best_obj.get("name"), best_type, best_score)
            return best_obj, best_type

        # Last resort: case-insensitive substring on raw name (with numeric/brand guard)
        lower_q = query.lower()
        for comp_type, comps in data.items():
            for obj in comps.values():
                name = (obj.get("name") or "")
                if lower_q in name.lower():
                    n_tokens, n_numeric, n_brand = candidate_tokens(name)
                    if q_brand_tokens and not (q_brand_tokens & n_brand):
                        continue
                    if not numeric_matches(q_numeric_tokens, n_numeric):
                        continue
                    logger.debug(
                        "best_match: raw substring last-resort -> %s (%s)", name, comp_type)
                    return obj, comp_type

        logger.debug("best_match: no confident match for query %r", query)
        return None, None

    except Exception:
        logger.exception("_best_match_in_dataset failed:")
        return None, None


def _parse_price_to_int(price_str: str) -> int:
    """Safe parse of price-like strings into integer (e.g., '₱12,000' -> 12000)."""
    try:
        if not price_str:
            return 0
        s = str(price_str)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0


# --------------------------
# Budget-based build recommendation utilities
# --------------------------


def parse_price(price_str: str) -> int:
    """Convert '₱12,000' or '12000' to integer 12000 for numeric comparisons."""
    return _parse_price_to_int(price_str)


def compatible(motherboard: Dict, cpu: Dict, ram: Dict, gpu: Dict, psu: Dict) -> bool:
    """Quick compatibility checks (defensive)."""
    try:
        if cpu and motherboard:
            cpu_socket = (cpu.get("socket") or "").lower()
            mobo_socket = (motherboard.get("socket") or "").lower()
            if cpu_socket and mobo_socket and cpu_socket not in mobo_socket:
                return False

        if ram and motherboard:
            ram_type = (ram.get("ram_type") or "").lower()
            mobo_ram_type = (motherboard.get("ram_type") or "").lower()
            if ram_type and mobo_ram_type and ram_type not in mobo_ram_type:
                return False

        if psu and gpu:
            psu_w = parse_price(psu.get("wattage", "0"))
            gpu_w = parse_price(gpu.get("power", "0"))
            if psu_w < gpu_w + 100:
                return False

        return True
    except Exception:
        return False


def score_build(cpu, gpu, motherboard, ram, storage, psu, usage: str = "general") -> int:
    """Basic scoring system to rank builds."""
    score = 0
    try:
        cores = int((cpu.get("cores", "0") or "0").split()[0])
        score += cores * 10
    except Exception:
        pass
    try:
        vram = int("".join(ch for ch in (
            gpu.get("vram", "") or "") if ch.isdigit()) or 0)
        score += vram * 5
    except Exception:
        pass
    sock = (cpu.get("socket") or "").lower()
    if "am5" in sock or "lga1700" in sock:
        score += 10
    if "ddr5" in (ram.get("ram_type") or "").lower():
        score += 8
    total_price = sum(parse_price(x.get("price", "0"))
                      for x in [cpu, gpu, motherboard, ram, storage, psu] if x)
    score -= int(total_price / 2000)
    return score


# --------------------------
# Helper: filter items by shop availability
# --------------------------
def _filter_items_by_shop(items: List[Dict], shop_id: str, min_keep: int = 1) -> List[Dict]:
    """
    Return items that are marked as available at shop_id first.
    If too few items are marked for that shop (less than min_keep),
    return the original items list (fallback).
    Each item is expected to optionally contain a "stores": ["smfp_computer", ...] list.
    """
    if not shop_id or not isinstance(items, list):
        return items
    prefer = [it for it in items if isinstance(
        it, dict) and shop_id in (it.get("stores") or [])]
    if len(prefer) >= min_keep:
        return prefer
    # fallback: prefer items that explicitly mention any store, then global items
    # if shop-specific not found, still prefer items that are known-to-be-in-shops
    known_store = [it for it in items if isinstance(
        it, dict) and it.get("stores")]
    if known_store and len(known_store) >= min_keep:
        return known_store
    # final fallback: return full list unfiltered
    return items

# --------------------------
# Updated budget_builds: shop-aware and returns structured builds
# --------------------------


def budget_builds(budget: int, usage: str = "general", top_n: int = 3) -> List[Dict]:
    """
    Generate up to top_n builds under the given budget.

    Improvements:
     - Defensive price parsing (missing price fields tolerated)
     - Uses the higher-level `compatible()` check rather than brittle substring socket checks
     - Samples a small set of cheapest parts to limit combinations
     - Returns an empty list only if genuinely no combination fits budget/compatibility
    """
    builds = []

    # defensive getters (empty lists if missing)
    cpus = list(data.get("cpu", {}).values())[:12]
    gpus = list(data.get("gpu", {}).values())[:12]
    mobos = list(data.get("motherboard", {}).values())[:12]
    rams = list(data.get("ram", {}).values())[:8]
    storages = list(data.get("storage", {}).values())[:6]
    psus = list(data.get("psu", {}).values())[:6]

    # safe price extractor
    def safe_price(x):
        try:
            return parse_price(x.get("price", "0"))
        except Exception:
            return 0

    # sort each category by price (cheapest first)
    cpus = sorted(cpus, key=safe_price)
    gpus = sorted(gpus, key=safe_price)
    mobos = sorted(mobos, key=safe_price)
    rams = sorted(rams, key=safe_price)
    storages = sorted(storages, key=safe_price)
    psus = sorted(psus, key=safe_price)

    # sample slices to keep combinations reasonable
    cpus_slice = cpus[:6] if len(cpus) > 6 else cpus
    gpus_slice = gpus[:6] if len(gpus) > 6 else gpus
    mobos_slice = mobos[:6] if len(mobos) > 6 else mobos
    rams_slice = rams[:6] if len(rams) > 6 else rams
    storages_slice = storages[:4] if len(storages) > 4 else storages
    psus_slice = psus[:4] if len(psus) > 4 else psus

    for cpu in cpus_slice:
        for gpu in gpus_slice:
            for mobo in mobos_slice:
                for ram in rams_slice:
                    for storage in storages_slice:
                        for psu in psus_slice:
                            try:
                                total = sum(parse_price(x.get("price", "0")) for x in [
                                    cpu, gpu, mobo, ram, storage, psu] if x)
                            except Exception:
                                # treat parse failure as over budget to skip this combo
                                continue
                            if total > budget:
                                continue
                            # final compatibility gate using your compatible() helper
                            if not compatible(mobo, cpu, ram, gpu, psu):
                                continue
                            builds.append({
                                "cpu": cpu,
                                "gpu": gpu,
                                "motherboard": mobo,
                                "ram": ram,
                                "storage": storage,
                                "psu": psu,
                                "total_price": total,
                                "score": score_build(cpu, gpu, mobo, ram, storage, psu, usage)
                            })

    # sort by score desc then price asc
    builds_sorted = sorted(
        builds, key=lambda x: (-x.get("score", 0), x.get("total_price", 0)))
    return builds_sorted[:top_n]


def build_to_tap_response(build: Dict, build_id: str) -> Dict:
    """Format build results into tapable chip responses."""
    chips = []
    for comp in ["cpu", "gpu", "motherboard", "ram", "storage", "psu", "cpu_cooler"]:
        item = build.get(comp)
        if not item:
            continue
        name = item.get("name", "") or ""
        s = slugify(name)
        chips.append({
            "id": f"{comp}:{s}",
            "text": item.get("name", ""),
            "type": comp,
            "price": item.get("price", ""),
            "meta": item
        })
    summary = f"Estimated total: ₱{build['total_price']} — Score {build['score']}"
    return {"message_id": build_id, "message": summary, "chips": chips}


def build_chips(items: List[dict], ctype: str, limit: int = 6) -> List[dict]:
    """Top-level helper to format a list of component dicts into chips."""
    chips = []
    try:
        items_sorted = sorted(
            items, key=lambda v: _parse_price_to_int(v.get("price", "") or "0"))
    except Exception:
        items_sorted = items
    for v in items_sorted[:limit]:
        safe_name = (v.get("name") or "").strip() or "Unknown"
        chips.append({
            "id": f"{ctype}:{slugify(safe_name)}",

            "text": safe_name,
            "price": v.get("price", ""),
            "type": ctype,
            "meta": v
        })
    return chips


def get_compatible_components(query: str, data: dict) -> dict:
    """
    Compatibility resolver that returns a dict including:
      - found (bool)
      - target (str)
      - compatible_type (str)
      - reason (optional)
      - chips (list)  -- tapable items
      - text (optional) -- friendly conversational message for UI
    """
    result = {"found": False}
    try:
        q = (query or "").lower()
        q_norm = _normalize_text_for_match(query)
        component_types = ["cpu", "gpu", "motherboard",
                           "ram", "storage", "psu", "cpu_cooler"]

        # detect what the user is asking FOR (asked_type), e.g. "motherboard" or "cpu"
        asked_type = None
        for comp_type in component_types:
            if re.search(rf"\b{comp_type}s?\b", q):
                asked_type = comp_type
                break

        # robustly detect a named component mentioned in query
        target_obj, target_type = _best_match_in_dataset(query, data)
        logger.debug("DEBUG: _best_match_in_dataset -> %s (%s)", getattr(target_obj,
                     "get", lambda k: None)("name") if target_obj else None, target_type)

        # If we found a target but confidence is low (None returned), ensure no nonsense
        if target_obj:
            # ensure target_type is one of our known types
            if target_type not in component_types:
                target_type = None

        # ---------- If user asked for a TYPE and mentioned a specific component ----------
        if asked_type and target_obj:
            # sanity: if asked_type==motherboard but target_type==motherboard it's ambiguous.
            # handle common pairs explicitly:
            if asked_type == "cpu" and target_type == "motherboard":
                mobo_socket = (target_obj.get("socket") or "").lower()
                if not mobo_socket:
                    return {"found": False, "message": f"No socket information available for {target_obj.get('name', 'that motherboard')}."}
                matches = [
                    c for c in data.get("cpu", {}).values()
                    if mobo_socket and mobo_socket in (c.get("socket") or "").lower()
                ]
                if not matches:
                    return {"found": False, "message": f"I couldn’t find CPUs compatible with {target_obj.get('name')}."}
                chips = build_chips(matches, "cpu")
                text = f"Here are CPUs that fit the socket {target_obj.get('socket', '')} on {target_obj.get('name')}. Tap any item to see details."
                return {"found": True, "target": target_obj.get("name"), "compatible_type": "cpu", "reason": f"CPUs that fit socket {target_obj.get('socket', '')}", "chips": chips, "text": text}

            if asked_type == "motherboard" and target_type == "cpu":
                cpu_socket = (target_obj.get("socket") or "").lower()
                if not cpu_socket:
                    return {"found": False, "message": f"No socket information available for {target_obj.get('name', 'that CPU')}."}
                matches = [
                    m for m in data.get("motherboard", {}).values()
                    if cpu_socket and cpu_socket in (m.get("socket") or "").lower()
                ]
                if not matches:
                    return {"found": False, "message": f"I couldn’t find any motherboards compatible with {target_obj.get('name')}."}
                chips = build_chips(matches, "motherboard")
                text = f"These motherboards support {target_obj.get('name')} (socket {target_obj.get('socket', '')}). Tap any to view specs."
                return {"found": True, "target": target_obj.get("name"), "compatible_type": "motherboard", "reason": f"Motherboards with socket {target_obj.get('socket', '')}", "chips": chips, "text": text}

            if asked_type == "motherboard" and target_type == "ram":
                ram_type = (target_obj.get("ram_type") or "").lower()
                if not ram_type:
                    return {"found": False, "message": f"No RAM type information available for {target_obj.get('name', 'that RAM')}."}
                matches = [
                    m for m in data.get("motherboard", {}).values()
                    if ram_type and ram_type in (m.get("ram_type") or "").lower()
                ]
                if not matches:
                    return {"found": False, "message": f"I couldn’t find any motherboards compatible with {target_obj.get('name')}."}
                chips = build_chips(matches, "motherboard")
                text = f"Motherboards that support {target_obj.get('ram_type', '').upper()} RAM (matching {target_obj.get('name')}). Tap any to see details."
                return {"found": True, "target": target_obj.get("name"), "compatible_type": "motherboard", "reason": f"Motherboards that support {target_obj.get('ram_type', '')}", "chips": chips, "text": text}

            if asked_type == "psu" and target_type == "gpu":
                matches = list(data.get("psu", {}).values())
                if not matches:
                    return {"found": False, "message": f"I couldn’t find PSUs compatible with {target_obj.get('name')}."}
                chips = build_chips(matches, "psu")
                text = f"Suggested PSUs for {target_obj.get('name')}. Verify wattage and connectors before buying — tap an item for details."
                return {"found": True, "target": target_obj.get("name"), "compatible_type": "psu", "reason": "Suggested PSUs (check wattage vs GPU power draw)", "chips": chips, "text": text}

            if asked_type == "gpu" and target_type == "motherboard":
                matches = list(data.get("gpu", {}).values())
                chips = build_chips(matches, "gpu")
                text = f"Most GPUs fit a PCIe x16 slot — here are GPUs to consider for {target_obj.get('name')}. Tap to view each GPU."
                return {"found": True, "target": target_obj.get("name"), "compatible_type": "gpu", "reason": "GPUs that fit PCIe x16 slots", "chips": chips, "text": text}

            # fallback for other combos
            return {"found": False, "message": f"I don’t have a rule for matching {asked_type} with {target_obj.get('name')} in the dataset."}

        # ---------- asked_type but no specific named target (type-to-type) ----------
        if asked_type and not target_obj:
            # DDRx-specific motherboard query
            if asked_type == "motherboard" and (("ddr4" in q) or ("ddr5" in q) or re.search(r"\bddr\d\b", q)):
                desired = "ddr5" if "ddr5" in q else (
                    "ddr4" if "ddr4" in q else None)
                matches = [m for m in data.get("motherboard", {}).values(
                ) if desired and desired in (m.get("ram_type") or "").lower()]
                if matches:
                    chips = build_chips(matches, "motherboard")
                    text = f"Here are motherboards that support {desired.upper()}. Tap any to view specs."
                    return {"found": True, "target": f"Motherboards supporting {desired.upper()}", "compatible_type": "motherboard", "reason": f"Motherboards that support {desired.upper()} RAM", "chips": chips, "text": text}

            # socket-specific CPU query: e.g., "Which CPUs work with AM4?"
            socket_match = re.search(r"\b(am\d+|lga\s*\d+)\b", q_norm or q)
            if asked_type == "cpu" and socket_match:
                sock_token = socket_match.group(1).replace(" ", "").lower()
                matches = [c for c in data.get("cpu", {}).values() if sock_token and sock_token in (
                    c.get("socket") or "").lower().replace(" ", "")]
                if matches:
                    chips = build_chips(matches, "cpu")
                    text = f"CPUs for socket {sock_token.upper()}. Tap any to see details."
                    return {"found": True, "target": f"CPUs for socket {sock_token.upper()}", "compatible_type": "cpu", "reason": f"CPUs that fit socket {sock_token.upper()}", "chips": chips, "text": text}

            # generic lists (cpu/gpu/ram)
            if asked_type in ("gpu", "cpu", "ram"):
                matches = list(data.get(asked_type, {}).values())
                chips = build_chips(matches, asked_type)
                text = f"Showing available {asked_type.upper()} options — tap an item to view specs."
                return {"found": True, "target": f"{asked_type.upper()}s", "compatible_type": asked_type, "reason": "Available options", "chips": chips, "text": text}

            return {"found": False, "message": f"I can show typical compatible components for {asked_type}, but for precise matching I need a specific model."}

        # ---------- only a specific component mentioned but user didn't say what they want ----------
        if target_obj and not asked_type:
            # default to suggesting motherboards for CPUs
            if target_type == "cpu":
                cpu_socket = (target_obj.get("socket") or "").lower()
                matches = [m for m in data.get("motherboard", {}).values(
                ) if cpu_socket and cpu_socket in (m.get("socket") or "").lower()]
                if matches:
                    chips = build_chips(matches, "motherboard")
                    text = f"I found {target_obj.get('name')}. These motherboards match its socket — tap to view details."
                    return {"found": True, "target": target_obj.get("name"), "compatible_type": "motherboard", "reason": f"Motherboards with socket {target_obj.get('socket', '')}", "chips": chips, "text": text}
            return {"found": False, "message": "I found that component but not sure what compatibility you want—ask e.g. 'What motherboard is compatible with <name>?'"}

        # ---------- nothing matched ----------
        return {"found": False, "message": "I couldn’t find any component in the dataset that matches what you mentioned."}

    except Exception as e:
        logger.exception("get_compatible_components failed:")
        return {"found": False, "error": str(e), "message": "Compatibility resolver error."}


# --------------------------
# Gemini Fallback with Data
# --------------------------
# --------------------------
# Replace this: gemini_fallback_with_data
# --------------------------
def gemini_fallback_with_data(user_input: str, context_data: dict, chat_history: List[Dict] = None) -> str:
    """
    Gemini fallback tuned for concise, casual+academic tone and PH-currency/range rules.

    Post-conditions enforced:
      - If the model mentions prices, it MUST use Philippine peso (₱) and provide ranges (e.g., ₱4,000–₱6,000).
      - Do not print exact client-only prices. If unsure, give approximate ranges and a suggestion to "check local retailers".
      - Decode escaped unicode sequences (so '\u20b1' becomes '₱').
      - Remove forbidden phrases and redundant greetings.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # safe context snippet (trim to keep prompt small)
        try:
            context_snippet = json.dumps(context_data or {}, indent=2)[:3000]
        except Exception:
            context_snippet = "{}"

        user_lower = (user_input or "").lower()

        is_latest_query = bool(re.search(
            r"\b(latest|newest|what's new|what are the latest|recently released|released in 2025|2025|now 2025)\b",
            user_lower,
            flags=re.IGNORECASE,
        ))
        is_specs_query = any(k in user_lower for k in [
                             "details", "specs", "specifications", "show me", "what are the specs", "how much", "price", "cost"])
        is_build_query = any(k in user_lower for k in [
                             "build", "recommend", "pc build", "₱", "php", "peso", "budget", "recommendation"])

        base = (
            "You are ARsemble AI — a concise, factual PC-building assistant. "
            "Reply in a casual-but-academic tone (no greetings). "
            "Do NOT use phrases like 'dataset', 'I don't have information', 'not in my dataset', or 'as of my last update'. "
            "If uncertain, give a brief likely answer and a practical check the user can perform."
        )

        # Extra strict price & privacy rules we always want the model to follow
        price_instructions = (
            "\n\nIMPORTANT PRICING RULES (MUST FOLLOW):\n"
            "- If you mention prices, ALWAYS use Philippine peso and the ₱ symbol (e.g., ₱4,000). Do NOT use $ or USD.\n"
            "- When giving prices, provide only approximate ranges (e.g., ₱4,000–₱6,000). Do NOT output a precise single client price.\n"
            "- Round to the nearest 100 or 500 and keep ranges short (example: ₱4,500–₱6,000).\n"
            "- If you are unsure about local pricing, say 'Check local retailers for current pricing' (but still give a short rounded range).\n"
        )

        if is_latest_query:
            prompt = base + f"""

User asked: "{user_input}"

Task:
- Provide direct lists of current mainstream desktop CPU and GPU families/series and 2–4 notable example models per vendor (Intel, AMD, NVIDIA) where applicable.
- Include one short sentence summarizing the generation's key benefit.
- If you mention pricing at all here, follow the pricing rules below.

Context (if relevant):
{context_snippet}

{price_instructions}
"""
        elif is_specs_query or is_build_query:
            prompt = base + f"""

User asked: "{user_input}"

Task:
- If user requested specs or price, return a short factual block or compact build summary.
- If mentioning prices, follow the pricing rules below.
Context:
{context_snippet}

{price_instructions}
"""
        else:
            prompt = base + f"""

User asked: "{user_input}"

Task:
- Provide a concise (1–3 sentence) factual answer. Prefer the context if clearly relevant; otherwise answer from general knowledge without disclaimers about internal data.
- If you provide price guidance for any component, follow the pricing rules below.
Context:
{context_snippet}

{price_instructions}
"""

        # attach short chat history optionally
        if chat_history:
            try:
                recent = chat_history[-6:]
                hist_lines = ["\nChat history (most recent last):"]
                for m in recent:
                    hist_lines.append(
                        f"{m.get('role', 'user').upper()}: {m.get('text') or m.get('message', '')}")
                prompt += "\n" + "\n".join(hist_lines)
            except Exception:
                logger.exception("Failed to attach chat history.")

        try:
            response = model.generate_content(
                prompt, temperature=0.15, max_output_tokens=600)
        except TypeError:
            try:
                response = model.generate_content(prompt)
            except Exception:
                logger.exception("Gemini call failed.")
                return "Sorry — I'm unable to generate that right now."

        raw = getattr(response, "text", None) or str(response)
        clean = raw.strip()

        # remove accidental greetings/forbidden phrases
        clean = re.sub(r'^(hi|hello|hey)[\s,\!\.]+',
                       '', clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(dataset|as of my last update|not in my database|not in my dataset|i don't have information|i do not have information)\b",
                       "", clean, flags=re.IGNORECASE)

        # --- Remove Markdown asterisks and convert list markers to clean bullets ---
        try:
            # remove bold/italic markers like **bold** or *italic* -> keep inner text
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean, flags=re.DOTALL)
            clean = re.sub(r'\*(.*?)\*', r'\1', clean, flags=re.DOTALL)

            # convert markdown list markers at start of line ('* ' or '- ') into a single bullet char
            clean = re.sub(r'(?m)^[\*\-\+]\s+', '• ', clean)

            # remove any remaining stray asterisks (e.g., inline '*' not used as emphasis)
            clean = clean.replace('*', '')

            # collapse repeated bullets or leading spaces that might remain
            clean = re.sub(r'(?m)^\s*•\s*•+\s*', '• ', clean)
            clean = re.sub(r'\n{3,}', '\n\n', clean)
        except Exception:
            logger.exception(
                "Markdown cleanup postprocessing failed; continuing.")

        # --- Postprocessing for currency / unicode escapes ---
        try:
            # decode common unicode escapes like \u20b1 -> ₱
            # First, ensure backslash-u sequences are decoded if present literally
            if "\\u" in clean:
                try:
                    clean = clean.encode("utf-8").decode("unicode_escape")
                except Exception:
                    # fallback: replace common escaped sequences
                    clean = clean.replace("\\u20b1", "₱")
            # Replace any "$1234" or "$ 1,234" -> "₱1,234" (simple heuristic)
            clean = re.sub(r"\$\s?([0-9][0-9,\.]*)", r"₱\1", clean)
            # Replace 'USD' or 'usd' near numbers or alone with '₱'
            clean = re.sub(r"\bUSD\b", "₱", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\bdollars?\b", "pesos",
                           clean, flags=re.IGNORECASE)

            # If model produced exact single prices like "₱12000", attempt to convert to a short range:
            # heuristic: detect lone price token and expand to ±15% rounded to nearest 100
            def expand_exact_to_range(match):
                num = match.group(1)
                digits = re.sub(r"[^\d]", "", num)
                if not digits:
                    return match.group(0)
                v = int(digits)
                lo = int(round(v * 0.85 / 100.0) * 100)
                hi = int(round(v * 1.15 / 100.0) * 100)
                return f"₱{lo:,d}–₱{hi:,d}"

            # Convert single-price occurrences like "₱12000" or "₱12,000" but not ranges already
            clean = re.sub(
                r"₱\s?([0-9]{3,7}(?:,[0-9]{3})?)\b(?!\s*[-–—])", expand_exact_to_range, clean)

            # collapse multiple blank lines
            clean = re.sub(r"\n{3,}", "\n\n", clean)
        except Exception:
            logger.exception(
                "Postprocessing of Gemini text failed; returning raw cleaned text.")

        # final trim
        clean = clean.strip()
        return clean
    except Exception:
        logger.exception("Gemini fallback failed.")
        return "Sorry — I ran into an issue generating that info."


# --------------------------
# Unified Handler
# --------------------------


def primary_model_response(user_input: str) -> str:
    """
    Placeholder primary model response. Replace with your primary model call.
    Currently raises NotImplementedError so the fallback path is exercised cleanly.
    """
    raise NotImplementedError("primary_model_response not implemented")


# --- SHOP and CLIENT management ---
PERSIST_DIR = os.getenv("ARSSEMBLE_PERSIST_DIR", ".arsemble_data")
CLIENTS_FILE = os.path.join(PERSIST_DIR, "clients.json")
SHOPS_FILE = os.path.join(PERSIST_DIR, "shops.json")

_clients_cache = None
_shops_cache = None


def _load_clients():
    global _clients_cache
    if _clients_cache is None:
        _clients_cache = load_json(CLIENTS_FILE, {})
    return _clients_cache


def _load_shops():
    global _shops_cache
    if _shops_cache is None:
        _shops_cache = load_json(SHOPS_FILE, {})
    return _shops_cache


def _save_clients():
    save_json(CLIENTS_FILE, _clients_cache or {})


def _save_shops():
    save_json(SHOPS_FILE, _shops_cache or {})


def add_shop(shop_id: str, info: dict):
    shops = _load_shops()
    shops[shop_id] = info
    _save_shops()
    return info


def list_shops(only_public=True):
    shops = _load_shops()
    if only_public:
        return {k: v for k, v in shops.items() if v.get("public", True)}
    return shops


def make_public_data(full_data: dict) -> dict:
    """
    Create a public-safe copy of the dataset for use with Gemini fallback.
    Converts exact prices into approximate ranges like ₱5,000–₱9,000 and
    keeps only non-sensitive, helpful fields so Gemini can answer generically.
    """
    def to_range(price_str: str) -> str:
        if not price_str:
            return "₱4K–₱15K"
        try:
            digits = int(''.join(ch for ch in str(price_str) if ch.isdigit()))
            low = max(4000, int(digits * 0.8))
            high = int(digits * 1.2)
            # round to nearest thousand for cleaner presentation
            low = int(round(low / 1000.0) * 1000)
            high = int(round(high / 1000.0) * 1000)
            # ensure low < high
            if low >= high:
                high = low + 1000
            return f"₱{low:,}–₱{high:,}"
        except Exception:
            return "₱4K–₱15K"

    public = {}
    for ctype, comps in (full_data or {}).items():
        public[ctype] = {}
        for key, comp in (comps or {}).items():
            if not isinstance(comp, dict):
                continue
            public[ctype][key] = {
                "name": comp.get("name"),
                "type": comp.get("type") or ctype,
                "price": to_range(comp.get("price", "")),
                # helpful spec snippets (non-sensitive)
                "socket": comp.get("socket"),
                "ram_type": comp.get("ram_type"),
                "cores": comp.get("cores"),
                "clock": comp.get("clock"),
                "vram": comp.get("vram") or comp.get("memory"),
                "tdp": comp.get("tdp") or comp.get("power") or comp.get("wattage"),
                # keep vendor/model identifiers if present (non-exact)
                "model": comp.get("model") or comp.get("sku"),
            }
    return public


def get_component_details(raw: str):
    """
    Flexible lookup used by the server. Returns a dict with at least:
      - found: bool
      - name, type, price, specs (when found)
      - suggestions: list (when not found)
      - chips: optional tapable chips list (when a direct match)
      - debug: diagnostic string
    """
    try:
        if not raw or not isinstance(raw, str):
            return {"found": False, "error": "No component name provided."}

        query_raw = raw.strip()
        # 1) try a direct chip-id helper if present
        lookup_fn = globals().get("lookup_component_by_chip_id")
        if callable(lookup_fn):
            try:
                direct = lookup_fn(query_raw)
            except Exception:
                direct = None
            if direct:
                # build a minimal 'specs' dict
                specs = {k: v for k, v in direct.items(
                ) if k not in ("name", "price", "type")}
                return {
                    "found": True,
                    "name": direct.get("name"),
                    "type": direct.get("type"),
                    "price": direct.get("price", ""),
                    "specs": specs,
                    "chips": [{"id": f"{direct.get('type')}:{slugify(direct.get('name', ''))}", "text": direct.get("name"), "price": direct.get("price", ""), "type": direct.get("type"), "meta": direct}],
                    "debug": "direct lookup_component_by_chip_id"
                }

        # 2) use fuzzy best-match helper from your module
        best_obj, best_type = _best_match_in_dataset(query_raw, data)
        if best_obj and best_type:
            specs = {k: v for k, v in best_obj.items(
            ) if k not in ("name", "price")}
            chip = {
                "id": f"{best_type}:{slugify(best_obj.get('name', ''))}",
                "text": best_obj.get("name"),
                "price": best_obj.get("price", ""),
                "type": best_type,
                "meta": best_obj
            }
            return {
                "found": True,
                "name": best_obj.get("name"),
                "type": best_type,
                "price": best_obj.get("price", ""),
                "specs": specs,
                "chips": [chip],
                "debug": "best_match_in_dataset"
            }

        # 3) suggestions: token-overlap scoring
        q_norm = _normalize_text_for_match(query_raw)
        q_tokens = set(t for t in re.split(r"\s+", q_norm) if t)
        scores = []
        for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu", "cpu_cooler"):
            for comp in (data.get(ctype, {}) or {}).values():
                name = (comp.get("name") or "")
                if not name:
                    continue
                name_norm = _normalize_text_for_match(name)
                name_tokens = set(t for t in re.split(r"\s+", name_norm) if t)
                overlap = len(q_tokens & name_tokens)
                if overlap > 0:
                    scores.append((overlap, ctype, comp.get("name"), comp))
        if scores:
            scores_sorted = sorted(scores, key=lambda x: -x[0])[:8]
            suggestions = []
            chips = []
            seen = set()
            for sc, ctype, name, comp in scores_sorted:
                key = (ctype, (name or "").strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(
                    {"name": name, "type": ctype, "score": int(sc)})
                chips.append({
                    "id": f"{ctype}:{slugify(name)}",
                    "text": name,
                    "price": comp.get("price", ""),
                    "type": ctype,
                    "meta": comp
                })
            return {
                "found": False,
                "error": f"No exact match for '{raw}', but here are close matches.",
                "suggestions": suggestions,
                "chips": chips,
                "debug": "suggestions returned"
            }

        # nothing found at all
        return {"found": False, "error": f"No match found for '{raw}' among available parts.", "debug": "no matches at all"}

    except Exception as e:
        logger.exception("get_component_details failed:")
        return {"found": False, "error": "Lookup failed due to an internal error.", "exception": str(e)}


def _format_shops_list(shops: dict, client_id: str = "smfp_computer", heading: str = "Here are computer shops I can suggest:") -> str:
    """
    Return a nicely formatted multi-line string listing shops.
    Client shop (if present) is shown first and marked as CLIENT SHOP.
    Each shop shows: name (CLIENT SHOP), address, region (only once).
    """
    if not shops:
        return "I don't have any shop information available right now."

    # keep order: client first if exists, then other public shops sorted by name
    items = []
    client = shops.get(client_id)
    if client:
        items.append((client_id, client))

    # other shops: public ones excluding the client_id
    others = [
        (sid, s) for sid, s in shops.items()
        if sid != client_id and s.get("public", True)
    ]
    # sort by name for stable order
    others_sorted = sorted(others, key=lambda x: (
        x[1].get("name") or "").lower())
    items.extend(others_sorted)

    lines = [heading, ""]
    for sid, info in items:
        name = info.get("name", "Unknown")
        addr = info.get("address", "").strip()
        region = info.get("region", "").strip()
        notes = info.get("notes", "").strip()

        # mark client shop
        client_tag = " — CLIENT SHOP" if sid == client_id else ""

        # build shop block (2–3 lines)
        # avoid repeating region if it's already in address
        addr_line = addr
        if region and region.lower() not in (addr or "").lower():
            addr_line = f"{addr} — {region}" if addr else region

        # short one-line name + tag
        lines.append(f"• {name}{client_tag}")
        if addr_line:
            lines.append(f"  {addr_line}")
        if notes:
            lines.append(f"  {notes}")
        lines.append("")  # spacer between shops

    lines.append(
        "You can ask 'show me details of <shop-id>' to get more, or 'list shops' to see all shops.")
    return "\n".join(lines)


def get_ai_response(user_input: str) -> dict:
    """
    Unified AI handler with:
     - primary_model (if available) attempt
     - conversational/latest routing -> Gemini (public_data)
     - PSU detection
     - compatibility detection (local resolver, fallback to Gemini public_data)
     - budget builds
     - local lookup / lists
     - final fallback -> Gemini (public_data)
    """
    try:
        # Try primary model first (if implemented)
        primary_text = primary_model_response(user_input)
        return {"source": "primary", "text": primary_text, "used_fallback": False}
    except Exception as e:
        logger.warning("[PRIMARY ERROR] %s — switching to fallback.", str(e))

    try:
        q = (user_input or "").strip()
        lower_q = q.lower()

        # ---------- LOCAL exact lookup for specs/pricing ----------
        # If user asks for details/specs/price, prefer returning exact client dataset price
        try:
            # treat queries asking for "details", "specs", "how much", "price", "cost", or "details about X"
            if any(k in lower_q for k in ("details", "specs", "how much", "price", "cost")) or re.search(r"\bdetails about\b", lower_q):
                local_lookup = get_component_details(q)
                # If we found a component and it has a numeric/explicit price in the client data, return it directly.
                if local_lookup and local_lookup.get("found"):
                    price = (local_lookup.get("price") or "").strip()
                    # simple numeric check — contains any digit
                    if price and re.search(r"\d", price):
                        # format a concise text response (frontend expects "text")
                        specs = local_lookup.get("specs", {}) or {}
                        specs_lines = []
                        for k, v in specs.items():
                            specs_lines.append(f"{k}: {v}")
                        specs_text = "\n".join(specs_lines).strip()
                        text_lines = [
                            f"{local_lookup.get('name')} — {local_lookup.get('type', '').upper()}",
                            f"Price: {price}",
                        ]
                        if specs_text:
                            text_lines.append("")
                            text_lines.append("Specs:")
                            text_lines.append(specs_text)
                        # also include a hint to check local retailers if user wants confirmation
                        text_lines.append("")
                        text_lines.append(
                            "Check local retailers for current pricing and stock.")
                        text = "\n".join(text_lines).strip()
                        return {
                            "source": "local-exact",
                            "text": text,
                            "found": True,
                            "details": local_lookup
                        }
        except Exception:
            logger.exception(
                "Local exact lookup for specs/pricing failed; falling through to other handlers.")

        # ---------- Conversational/latest -> Gemini (use public data) ----------
        try:
            conversational_patterns = [
                r"\bwhat is\b", r"\bwhat does\b", r"\bwhy\b", r"\bhow\b",
                r"\bdifference\b", r"\bexplain\b", r"\bcompare\b", r"\b vs\b", r"\bversus\b",
                r"\blatest\b", r"\bnewest\b", r"\bnew release\b", r"\breleased\b", r"\bannounced\b",
                r"\breview\b", r"\bbenchmark\b", r"\bhow good\b", r"\bshould i buy\b", r"\bworth it\b",
                r"\brelease date\b"
            ]
            if any(re.search(pat, lower_q, flags=re.IGNORECASE) for pat in conversational_patterns):
                logger.info(
                    "Routing conversational/latest question to Gemini fallback.")
                gem_text = gemini_fallback_with_data(
                    user_input, make_public_data(data))
                return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}
        except Exception:
            logger.exception(
                "Error in conversational routing check; continuing to local handlers.")

        # ---------- 0) Bottleneck natural-language detection ----------
        if "bottleneck" in lower_q:
            patterns = [
                r"bottleneck(?:\s+of|\s+between)?\s+(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+)$",
                r"which.*bottleneck.*(?:between)?\s+(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+)$",
                r"(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+?)\s+.*bottleneck"
            ]
            found_a = None
            found_b = None
            for pat in patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    a_raw = m.group(1).strip(" \"'")
                    b_raw = m.group(2).strip(" \"'")
                    a_obj = lookup_component_by_chip_id(a_raw) or None
                    b_obj = lookup_component_by_chip_id(b_raw) or None
                    if not a_obj:
                        a_obj, _ = _best_match_in_dataset(a_raw, data)
                    if not b_obj:
                        b_obj, _ = _best_match_in_dataset(b_raw, data)
                    found_a = a_obj
                    found_b = b_obj
                    break

            if not (found_a and found_b):
                if "+" in q:
                    parts = [p.strip() for p in q.split("+", 1)]
                elif " and " in lower_q:
                    parts = [p.strip() for p in re.split(
                        r"\band\b", q, flags=re.IGNORECASE)[:2]]
                else:
                    parts = []
                if len(parts) >= 2 and not (found_a and found_b):
                    a_raw, b_raw = parts[0], parts[1]
                    a_obj = lookup_component_by_chip_id(a_raw) or None
                    b_obj = lookup_component_by_chip_id(b_raw) or None
                    if not a_obj:
                        a_obj, _ = _best_match_in_dataset(a_raw, data)
                    if not b_obj:
                        b_obj, _ = _best_match_in_dataset(b_raw, data)
                    found_a = found_a or a_obj
                    found_b = found_b or b_obj

            if not (found_a and found_b):
                scores = []
                q_norm = _normalize_text_for_match(q)
                q_tokens = set([t for t in re.split(r"\s+", q_norm) if t])
                for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu"):
                    for comp in (data.get(ctype, {}) or {}).values():
                        name = (comp.get("name") or "")
                        name_norm = _normalize_text_for_match(name)
                        name_tokens = set(
                            [t for t in re.split(r"\s+", name_norm) if t])
                        overlap = len(q_tokens & name_tokens)
                        if overlap > 0:
                            scores.append((overlap, comp))
                scores_sorted = sorted(scores, key=lambda x: -x[0])
                unique = []
                for sc, comp in scores_sorted:
                    if comp not in unique:
                        unique.append(comp)
                    if len(unique) >= 2:
                        break
                if unique:
                    if not found_a and len(unique) >= 1:
                        found_a = unique[0]
                    if not found_b and len(unique) >= 2:
                        found_b = unique[1]

            if found_a or found_b:
                if not found_a or not found_b:
                    missing = "CPU" if not found_a else "GPU"
                    msg = f"Could not confidently identify both components from your query. Missing: {missing}. Try 'bottleneck of <cpu> and <gpu>'."
                    return {"source": "local-bottleneck", "text": msg}

                a_type = None
                b_type = None
                for t in ("cpu", "gpu", "motherboard", "ram", "storage", "psu"):
                    if found_a in data.get(t, {}).values():
                        a_type = t
                    if found_b in data.get(t, {}).values():
                        b_type = t

                cpu_obj = None
                gpu_obj = None
                if a_type == "gpu" or ("gpu" in (found_a.get("type") or "").lower()):
                    gpu_obj = found_a
                if a_type == "cpu" or ("cpu" in (found_a.get("type") or "").lower()):
                    cpu_obj = found_a
                if b_type == "gpu" or ("gpu" in (found_b.get("type") or "").lower()):
                    gpu_obj = found_b
                if b_type == "cpu" or ("cpu" in (found_b.get("type") or "").lower()):
                    cpu_obj = found_b

                def looks_like_cpu(name):
                    return bool(re.search(r"\b(ryzen|intel|core|i3|i5|i7|i9|xeon|athlon)\b", (name or "").lower()))

                def looks_like_gpu(name):
                    return bool(re.search(r"\b(rtx|gtx|radeon|rx|vga|titan)\b", (name or "").lower()))

                if not cpu_obj and looks_like_cpu(found_a.get("name", "")):
                    cpu_obj = found_a
                if not cpu_obj and looks_like_cpu(found_b.get("name", "")):
                    cpu_obj = found_b
                if not gpu_obj and looks_like_gpu(found_a.get("name", "")):
                    gpu_obj = found_a
                if not gpu_obj and looks_like_gpu(found_b.get("name", "")):
                    gpu_obj = found_b

                if not cpu_obj:
                    cpu_obj = found_a
                if not gpu_obj:
                    gpu_obj = found_b

                try:
                    analyzer = globals().get("analyze_bottleneck_for_build") or analyze_bottleneck_text
                    if analyzer == analyze_bottleneck_text:
                        cpu_to_pass = cpu_obj if isinstance(
                            cpu_obj, dict) else {}
                        gpu_to_pass = gpu_obj if isinstance(
                            gpu_obj, dict) else {}
                        text = analyze_bottleneck_text(
                            cpu_to_pass, gpu_to_pass)
                    else:
                        text = analyze_bottleneck_for_build(cpu_obj, gpu_obj)
                    return {"source": "local-bottleneck", "text": text}
                except Exception:
                    logger.exception(
                        "Bottleneck analyzer failed; falling through.")

        # ---------- 1) PSU / power-supply natural language detection ----------
        if re.search(r"\b(psu|power supply|power-supply|power supply i need|psu do i need|psu needed|what power supply|how much power)\b", lower_q):
            try:
                psu_resp = recommend_psu_for_query_with_chips(
                    user_input, data, headroom_percent=30)
                if psu_resp and not psu_resp.get("error"):
                    return {"source": "local-psu", "type": "psu_recommendation", **psu_resp}
            except Exception:
                logger.exception(
                    "PSU recommendation failed; falling through to other handlers.")

        # ---------- 2) Compatibility question detection ----------
        if "compatible" in lower_q or "works with" in lower_q:
            try:
                compat = get_compatible_components(user_input, data)
                if compat and compat.get("found"):
                    target_name = compat.get("target", "This component")
                    comp_type_label = (compat.get(
                        "compatible_type") or "component").upper()
                    raw_chips = compat.get("chips", []) or []

                    normalized = []
                    for c in raw_chips:
                        if not isinstance(c, dict):
                            continue
                        text = (c.get("text") or c.get("name") or c.get(
                            "meta", {}).get("name") or "").strip()
                        ctype = (c.get("type") or c.get("meta", {}).get("type") or (
                            comp_type_label.lower() if comp_type_label else "")).lower()
                        price = c.get("price") or (
                            c.get("meta") or {}).get("price") or ""
                        meta = c.get("meta") or {}

                        existing_id = c.get("id") or ""
                        slug_part = slugify(text) or slugify(
                            meta.get("name", ""))
                        if existing_id:
                            prefix = f"{ctype}:{ctype}-"
                            if existing_id.startswith(prefix) and slug_part:
                                cid = f"{ctype}:{slug_part}"
                            else:
                                cid = existing_id
                        else:
                            cid = f"{ctype}:{slug_part}" if slug_part else f"{ctype}:unknown"

                        norm = {
                            "id": cid,
                            "text": text or (meta.get("name") or "Unknown"),
                            "type": ctype or "component",
                            "price": price,
                            "meta": meta
                        }
                        normalized.append(norm)

                    if normalized:
                        preview_lines = []
                        for c in normalized[:6]:
                            text_label = c.get("text") or c.get("id")
                            price_str = f" ({c.get('price')})" if c.get(
                                "price") else ""
                            preview_lines.append(f"- {text_label}{price_str}")
                        listed = "\n".join(preview_lines)
                        friendly_text = f"Sure — {target_name} works well with these {comp_type_label} options:\n{listed}\n\n(Tap any item to view details.)"
                    else:
                        friendly_text = f"{target_name} is compatible with several {comp_type_label.lower()} options."

                    return {
                        "source": "local-compatibility",
                        "type": "compatibility",
                        "target": target_name,
                        "compatible_type": compat.get("compatible_type"),
                        "results": normalized,
                        "chips": normalized,
                        "text": friendly_text
                    }
                else:
                    # No local compatibility result -> use Gemini fallback (public-safe) with a focused prompt
                    try:
                        logger.info(
                            "No local compatibility result — using Gemini fallback with public data.")

                        # Attempt to extract two component mentions from the user's query
                        comp_a = None
                        comp_b = None
                        # patterns / separators
                        if " and " in lower_q:
                            parts = [p.strip() for p in re.split(
                                r"\band\b", q, flags=re.IGNORECASE) if p.strip()]
                            if len(parts) >= 2:
                                comp_a, comp_b = parts[0], parts[1]
                        if not (comp_a and comp_b) and "+" in q:
                            parts = [p.strip() for p in q.split("+", 1)]
                            if len(parts) >= 2:
                                comp_a, comp_b = parts[0], parts[1]
                        if not (comp_a and comp_b):
                            m = re.search(
                                r"(.+?)\s+(?:compatible|compatibility|works with|work with)\s+(.+)", q, flags=re.IGNORECASE)
                            if m:
                                comp_a = comp_a or m.group(1).strip(" \"'")
                                comp_b = comp_b or m.group(2).strip(" \"'")

                        # fallback: take top two token-overlap matches from dataset names (if present)
                        if not (comp_a and comp_b):
                            q_norm = _normalize_text_for_match(q)
                            q_tokens = set(
                                t for t in re.split(r"\s+", q_norm) if t)
                            scores = []
                            for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu", "cpu_cooler"):
                                for comp in (data.get(ctype, {}) or {}).values():
                                    name = (comp.get("name") or "")
                                    if not name:
                                        continue
                                    name_norm = _normalize_text_for_match(name)
                                    name_tokens = set(
                                        t for t in re.split(r"\s+", name_norm) if t)
                                    overlap = len(q_tokens & name_tokens)
                                    if overlap > 0:
                                        scores.append((overlap, name))
                            scores_sorted = sorted(scores, key=lambda x: -x[0])
                            uniq = []
                            for sc, nm in scores_sorted:
                                if nm not in uniq:
                                    uniq.append(nm)
                                if len(uniq) >= 2:
                                    break
                            if uniq:
                                comp_a = comp_a or uniq[0]
                                comp_b = comp_b or (
                                    uniq[1] if len(uniq) > 1 else None)

                        comp_a_label = comp_a or "Component A"
                        comp_b_label = comp_b or "Component B"
                        fallback_prompt = (
                            "You are ARsemble AI — a precise PC-building assistant. Answer in 1–3 short sentences.\n\n"
                            f"The user asked: \"{user_input}\"\n\n"
                            f"Task: Determine whether the following two components are compatible: \"{comp_a_label}\" and \"{comp_b_label}\".\n"
                            "- Provide a short verdict line (one of: Compatible, Not compatible, Likely compatible - check X).\n"
                            "- Then list 2–4 practical checks the user can perform to verify compatibility (socket, chipset, RAM type, PSU wattage, PCIe slot, GPU length).\n"
                            "- If relevant, list the exact socket/chipset names or connectors to check (e.g., AM5, LGA1700, PCIe x16, 8-pin GPU power).\n"
                            "Do NOT mention internal dataset or say you lack local data. Keep it actionable and concise."
                        )

                        gem_text = gemini_fallback_with_data(
                            fallback_prompt, make_public_data(data))
                        if gem_text:
                            return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}
                        else:
                            return {"source": "local-compatibility", "type": "compatibility", "message": "I can't determine compatibility from local data — try giving both model names explicitly."}
                    except Exception:
                        logger.exception(
                            "Conversational Gemini fallback failed; returning default message.")
                        return {"source": "local-compatibility", "type": "compatibility", "message": compat.get("message", "No compatible parts found.") if compat else "No compatible parts found."}
            except Exception:
                logger.exception(
                    "Compatibility resolver failed; falling through.")

        # ---------- 3) Budget build detection ----------
        gpu_model_present = bool(
            re.search(r"\b(?:rtx|gtx|rx|radeon)\s*\d{3,4}\b", lower_q))
        currency_match = re.search(
            r"(?:₱|\bphp\b|\bpeso\b)\s?([0-9,]{3,7})", user_input, flags=re.IGNORECASE)
        plain_number_match = re.search(r"\b([0-9]{4,6})\b", user_input)
        is_budget_asked = any(k in lower_q for k in [
            "budget", "build", "recommend", "pc build", "how much", "how much do i need", "i have", "i've got", "i've got a budget"
        ])

        budget = None
        if currency_match:
            digits = re.sub(r"[^\d]", "", currency_match.group(1))
            if digits:
                budget = int(digits)
        elif is_budget_asked and plain_number_match and not gpu_model_present:
            budget = int(plain_number_match.group(1))

        if budget:
            logger.info(
                "Budget build requested: budget=%s, query=%s", budget, user_input)
            builds = budget_builds(budget)
            if builds:
                return {
                    "source": "local-recommendation",
                    "type": "budget_builds",
                    "budget": budget,
                    "results": [
                        build_to_tap_response_with_watts(b, f"build_{i+1}") for i, b in enumerate(builds)
                    ]
                }
            else:
                return {
                    "source": "local-recommendation",
                    "type": "budget_builds",
                    "budget": budget,
                    "message": f"No compatible builds found within that budget. Try increasing the budget or check the dataset prices/specs."
                }

         # ---------- 3.5.5) Shops / retailers suggestion (client-first) ----------
        if re.search(r"\b(shop|store|computer shop|pc shop|retailer|where to buy|suggest shops|suggest any computer shops)\b", lower_q):
            try:
                # Load only public shops but prefer client if present
                shops_dict = list_shops(only_public=True) or {}
                # try to place client shop first if present
                client_key = "smfp_computer"  # change this if your client id differs
                ordered = []

                # if client exists and public, put it first and mark as client
                if client_key in shops_dict:
                    c = dict(shops_dict[client_key])
                    c["_is_client"] = True
                    ordered.append(c)

                # then append other public shops (skip duplicate)
                for k, v in shops_dict.items():
                    if k == client_key:
                        continue
                    item = dict(v)
                    item["_is_client"] = False
                    ordered.append(item)

                # fallback: if no public shops found, include non-public as last resort
                if not ordered:
                    all_shops = _load_shops() or {}
                    for k, v in all_shops.items():
                        item = dict(v)
                        item["_is_client"] = (k == client_key)
                        ordered.append(item)

                # Build friendly text: client first with "Client shop" note, then others
                lines = []
                if ordered:
                    lines.append(
                        "Here are computer shops I can suggest:\n")
                    for s in ordered:
                        name = s.get("name") or "Unknown"
                        addr = s.get("address") or ""
                        region = s.get("region") or ""
                        note = " — CLIENT SHOP" if s.get("_is_client") else ""
                        lines.append(
                            f"• {name}{note}\n    {addr} {('— ' + region) if region else ''}".strip())
                else:
                    lines.append(
                        "I don’t have any shops listed right now. You can add one with the add_shop(...) helper on the server.")

                friendly_text = "\n".join(lines)
                return {
                    "source": "local-list",
                    "type": "shops_list",
                    "target": "shops",
                    "results": ordered,
                    "text": friendly_text
                }
            except Exception:
                logger.exception(
                    "Shops suggestion handler failed; falling through to standard handlers.")

        # ---------- 3.6) Dataset list queries (text-only lists, no tapables for large lists) ----------
        list_pattern = re.search(
            r"\b(list|show me|available|do you have|all the|give me the list|give me a list)\b", lower_q)
        if list_pattern:
            component_types = {
                "cpu cooler": "cpu_cooler",
                "cpu_cooler": "cpu_cooler",
                "cpu": "cpu",
                "cpus": "cpu",
                "gpu": "gpu",
                "gpus": "gpu",
                "motherboard": "motherboard",
                "motherboards": "motherboard",
                "ram": "ram",
                "memory": "ram",
                "storage": "storage",
                "ssd": "storage",
                "hdd": "storage",
                "psu": "psu",
                "power supply": "psu",
                "power-supply": "psu"
            }
            matched_type = None
            for key, norm in component_types.items():
                if key in lower_q:
                    matched_type = norm
                    break

            brand_tokens = {
                "ryzen": ("amd", "ryzen"),
                "amd": ("amd", "ryzen"),
                "intel": ("intel", "core", "i3", "i5", "i7", "i9"),
                "nvidia": ("nvidia", "rtx", "gtx", "geforce"),
                "radeon": ("radeon", "rx"),
                "amd-gpu": ("amd", "radeon", "rx"),
                "intel-gpu": ("intel", "arc", "iris")
            }
            requested_brands = []
            for tok in brand_tokens.keys():
                if tok in lower_q:
                    requested_brands.append(tok)

            if matched_type:
                items = list(data.get(matched_type, {}).values()) or []

                if items:
                    def matches_brand(name: str, brand_keys: list) -> bool:
                        n = (name or "").lower()
                        for k in brand_keys:
                            if k in n:
                                return True
                        return False

                    filtered = items
                    if requested_brands and matched_type in ("cpu", "gpu"):
                        brand_keys = []
                        for rb in requested_brands:
                            brand_keys.extend(brand_tokens.get(rb, ()))
                        brand_keys = list(set(brand_keys))
                        filtered = [it for it in items if matches_brand(
                            it.get("name", ""), brand_keys)]

                    lines = []
                    limit = 200
                    for comp in filtered[:limit]:
                        name = (comp.get("name") or "").strip()
                        price = (comp.get("price") or "").strip()
                        price_str = f" — {price}" if price else ""

                        if matched_type == "cpu":
                            brand = "Intel" if re.search(r"\b(intel|core|i3|i5|i7|i9)\b", name, re.I) else (
                                "AMD" if re.search(
                                    r"\b(ryzen|athlon|threadripper)\b", name, re.I) else "Other"
                            )
                            model = re.sub(
                                r"(?i)\b(intel|amd|amd ryzen|ryzen|core|processor|cpu)\b", "", name).strip()
                            model = model if model else name
                            lines.append(f"{brand} — {model}{price_str}")
                        elif matched_type == "gpu":
                            brand = "NVIDIA" if re.search(r"\b(rtx|gtx|geforce|nvidia)\b", name, re.I) else (
                                "AMD" if re.search(r"\b(radeon|rx)\b", name, re.I) else "Intel" if re.search(
                                    r"\b(arc|iris)\b", name, re.I) else "Other"
                            )
                            model = re.sub(
                                r"(?i)\b(nvidia|amd|intel|gpu|graphics|vga|geforce|radeon)\b", "", name).strip()
                            model = model if model else name
                            lines.append(f"{brand} — {model}{price_str}")
                        elif matched_type == "psu":
                            watt = ""
                            m = re.search(r"(\d{3,4})\s*w", name, re.I)
                            if m:
                                watt = f"{m.group(1)} W"
                            brand = name.split()[0] if name else "Unknown"
                            lines.append(
                                f"{brand} — {watt or 'Unknown wattage'}{price_str}")
                        else:
                            lines.append(f"{name}{price_str}")

                    if not lines and requested_brands and matched_type in ("cpu", "gpu"):
                        brand_label = ", ".join(requested_brands)
                        return {"source": "local-list", "type": "component_list_textonly", "target": matched_type, "results": [], "text": f"I couldn't find any {matched_type.replace('_', ' ')} entries matching '{brand_label}'. Try a broader query like 'list all CPUs' or check spelling."}

                    joined_lines = "\n".join(lines)
                    friendly_text = (
                        f"Here’s a quick look at the available {matched_type.replace('_', ' ')}s:\n\n"
                        f"{joined_lines}\n\n"
                        f"(List trimmed to the first {limit} items for readability. Ask 'show me specs of <name>' for details.)"
                    )

                    return {"source": "local-list", "type": "component_list_textonly", "target": matched_type, "results": [], "text": friendly_text}
                else:
                    # No local entries -> ask Gemini (public data) for latest/available info
                    try:
                        logger.info(
                            "No local entries for %s — routing to Gemini fallback.", matched_type)
                        fallback_prompt = (
                            f"The user asked about the latest {matched_type.replace('_', ' ')}s. "
                            f"Give a helpful summary of currently available and recent {matched_type.replace('_', ' ')}s "
                            f"(2023–2025 era) including relevant CPU or GPU families if needed. "
                            f"List them clearly with bullet points per vendor and mention common sockets/chipsets. Keep it concise."
                        )
                        gem_text = gemini_fallback_with_data(
                            fallback_prompt, make_public_data(data))
                        if gem_text:
                            return {"source": "gemini-fallback", "type": "component_list_textonly", "target": matched_type, "results": [], "text": gem_text, "used_fallback": True}
                    except Exception:
                        logger.exception(
                            "Gemini fallback for missing list failed.")
                        friendly_text = (
                            f"There are no available {matched_type.replace('_', ' ')}s right now. Compatibility and socket information may still be available.")
                        return {"source": "local-list", "type": "component_list_missing", "target": matched_type, "results": [], "text": friendly_text}
            else:
                categories = [k for k, v in data.items() if v]
                friendly_text = ("I have data available for: " +
                                 ", ".join(categories) + ". Ask 'list CPUs' or 'list GPUs'.")
                return {"source": "local-list", "type": "available_categories", "results": [], "text": friendly_text}

        # ---------- 4) Final fallback to Gemini (public data) ----------
        gemini_text = gemini_fallback_with_data(
            user_input, make_public_data(data))
        return {"source": "gemini-fallback", "text": gemini_text, "used_fallback": True}

    except Exception as e:
        logger.exception("get_ai_response failed unexpectedly:")
        return {"source": "error", "text": "Internal error processing request."}


# --------------------------
# PSU suggestions + tapable wattage chips
# --------------------------

def _psu_chip_from_obj(psu_obj: dict) -> dict:
    """Return a tapable chip dict for a PSU object (id, text, price, meta)."""
    if not psu_obj or not isinstance(psu_obj, dict):
        return {}
    name = (psu_obj.get("name") or "Unknown PSU").strip()
    watt = _parse_psu_wattage(psu_obj) or 0
    slug = slugify(name)
    return {
        "id": f"psu:{slug}-{watt}w",
        "text": f"{name} — {watt} W",
        "price": psu_obj.get("price", ""),
        "type": "psu",
        "meta": psu_obj
    }


def suggest_psus_for_wattage_varied(req_watt: int, data: dict, limit: int = 6) -> List[dict]:
    """
    Return PSUs across varied watt tiers (under, near, above requirement).
    """
    psus = list(data.get("psu", {}).values())
    annotated = [(p, _parse_psu_wattage(p)) for p in psus]

    adequate = sorted([t for t in annotated if t[1] >= req_watt],
                      key=lambda x: (x[1] - req_watt, x[1]))
    under = sorted([t for t in annotated if t[1] < req_watt],
                   key=lambda x: (req_watt - x[1], -x[1]))

    result = []
    # half from adequate (at least 1)
    result.extend([p for p, w in adequate[:max(1, limit // 2)]])
    # add one smaller option if available
    if under:
        result.append(under[0][0])
    # fill with next adequate ones
    result.extend([p for p, w in adequate[max(1, limit // 2):limit]])
    # if still short, add top-tier
    if len(result) < limit:
        sorted_by_w = sorted(annotated, key=lambda x: -x[1])
        if sorted_by_w:
            top = sorted_by_w[0][0]
            result.append(top)

    # unique by name + watt
    seen = set()
    unique = []
    for p in result:
        name = (p.get("name") or "").strip()
        w = _parse_psu_wattage(p)
        key = (name.lower(), w)
        if key not in seen and name:
            seen.add(key)
            unique.append(p)
        if len(unique) >= limit:
            break
    return unique[:limit]


def recommend_psu_for_query(query: str, data: dict, headroom_percent: int = 30) -> dict:
    """
    Improved free-text PSU estimator:
    - matches GPUs/CPUs as before
    - tries to detect explicit motherboard names via _best_match_in_dataset
    - detects DDR4/DDR5 mentions and picks a representative RAM item
    - returns same shape as before
    """
    try:
        if not query or not isinstance(query, str):
            return {"error": "empty query"}

        q = query.lower()
        found = {"cpu": [],
                 "gpu": [],
                 "motherboard": [],
                 "ram": [],
                 "storage": [],
                 "psu": [],
                 "cpu_cooler": []
                 }

        # 1) direct name/slug matches across categories
        for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu", "cpu_cooler"):
            bucket = data.get(ctype, {}) or {}
            for obj in bucket.values():
                name = (obj.get("name") or "").lower()
                if not name:
                    continue
                # match on slug or substring
                if slugify(name) in slugify(q) or name in q:
                    found[ctype].append(obj)

        # 2) fallback token-based GPU/CPU patterns (as before)
        gpu_tokens = re.findall(r"\b(?:rtx|gtx|rx|radeon)\s*\d{3,4}\b", q)
        for gt in gpu_tokens:
            obj, t = _best_match_in_dataset(gt, data)
            if obj and t == "gpu" and obj not in found["gpu"]:
                found["gpu"].append(obj)

        cpu_tokens = re.findall(
            r"\b(?:ryzen\s*\d|\bi\d\b|\bi3\b|\bi5\b|\bi7\b|\bi9\b|\bcore\b)\s*[\w\d-]*", q)
        for ct in cpu_tokens:
            obj, t = _best_match_in_dataset(ct, data)
            if obj and t == "cpu" and obj not in found["cpu"]:
                found["cpu"].append(obj)

        # 3) Try matching motherboard explicitly from the full query (helps detect "h610m k")
        mobo_obj, mobo_type = _best_match_in_dataset(query, data)
        if mobo_obj and mobo_type == "motherboard" and mobo_obj not in found["motherboard"]:
            found["motherboard"].append(mobo_obj)

        # 4) Detect DDR4 / DDR5 mentions and pick a representative RAM module if available
        if "ddr4" in q or "ddr5" in q:
            desired = "ddr5" if "ddr5" in q else "ddr4"
            # find first RAM in dataset that matches ram_type
            for r in data.get("ram", {}).values():
                if desired in (r.get("ram_type") or "").lower():
                    if r not in found["ram"]:
                        found["ram"].append(r)
                        break

        # 5) if still missing a motherboard but query contains common mobo tokens, pick a cheap one as representative
        if not found["motherboard"]:
            common_mobo_tokens = ["b450", "b550", "x570",
                                  "h610", "h510", "z690", "a520", "b460", "b660"]
            if any(tok in q for tok in common_mobo_tokens):
                for m in data.get("motherboard", {}).values():
                    name = (m.get("name") or "").lower()
                    if any(tok in name for tok in common_mobo_tokens):
                        found["motherboard"].append(m)
                        break

        # 6) Aggressive substring scan as last resort (helps if user typed many short tokens)
        if not any(found.values()):
            lower_q = q
            for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu"):
                for obj in (data.get(ctype, {}) or {}).values():
                    name = (obj.get("name") or "").lower()
                    if name and name in lower_q:
                        found[ctype].append(obj)

        # Build human summary
        detected_parts = []
        for t in ("cpu", "gpu", "motherboard", "ram", "storage"):
            arr = found.get(t) or []
            if arr:
                names = ", ".join(a.get("name", "") for a in arr[:3])
                detected_parts.append(f"{t.upper()}: {names}")
        detected_str = "; ".join(detected_parts) if detected_parts else ""

        if not detected_parts:
            return {"error": "No components detected in query. Try specifying exact GPU/CPU/motherboard names."}

        # Compute watt estimates (use estimate_component_wattage)
        component_watts = {}
        total = 0
        for t in ("cpu", "gpu", "motherboard", "ram", "storage"):
            arr = found.get(t) or []
            wsum = 0
            if arr:
                for comp in arr:
                    w = estimate_component_wattage(comp)
                    try:
                        wsum += int(w)
                    except Exception:
                        wsum += 0
            component_watts[t] = int(wsum)
            total += int(wsum)

        extras = 15
        component_watts["extras"] = extras
        total += extras

        target = total * (1 + float(headroom_percent) / 100.0)
        recommended = int((target + 49) // 50 * 50)

        return {
            "detected": found,
            "detected_str": detected_str,
            "component_watts": component_watts,
            "total_draw": int(total),
            "recommended_psu": int(recommended),
            "headroom_percent": int(headroom_percent),
        }

    except Exception as e:
        logger.exception("recommend_psu_for_query failed:")
        return {"error": str(e)}


def recommend_psu_for_query_with_chips(query: str, data: dict, headroom_percent: int = 30) -> dict:
    """Compute total wattage from components mentioned and suggest PSUs (with chips)."""
    base = recommend_psu_for_query(
        query, data, headroom_percent=headroom_percent)
    if base.get("error"):
        return base

    recommended = base.get("recommended_psu") or 0
    suggested_raw = suggest_psus_for_wattage_varied(recommended, data, limit=6)
    suggested_chips = [_psu_chip_from_obj(p) for p in suggested_raw]

    # dedupe suggested chips by id (just in case)
    seen_ids = set()
    deduped_chips = []
    for ch in suggested_chips:
        cid = ch.get("id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            deduped_chips.append(ch)

    # add tapable recommended watt chip (always present)
    recommended_chip = {
        "id": f"psu:recommended-{int(recommended)}w",
        "text": f"Recommended PSU: {int(recommended)} W",
        "type": "psu-recommendation",
        "meta": {"recommended_watt": int(recommended)}
    }

    base.update({
        "suggested_psu_chips": deduped_chips,
        "recommended_chip": recommended_chip,
        "detected_str": base.get("detected_str", "")
    })
    return base

# --------------------------
# Minimal watt & PSU helpers (drop in before build_to_tap_response_with_watts)
# --------------------------


def _parse_watt_value(v: str) -> int:
    """Parse the first integer-looking group from a string like '650W' -> 650"""
    try:
        if v is None:
            return 0
        s = str(v)
        m = re.search(r"(\d{2,4})", s.replace(",", ""))
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def _parse_psu_wattage(psu_obj: dict) -> int:
    """Extract numeric wattage from a PSU object (looks for 'wattage','w','power','rating' or in name)."""
    if not psu_obj or not isinstance(psu_obj, dict):
        return 0
    for k in ("wattage", "w", "power", "rating"):
        if k in psu_obj and psu_obj.get(k) is not None:
            w = _parse_watt_value(psu_obj.get(k))
            if w:
                return w
    # fallback: try to parse from name
    name = (psu_obj.get("name") or "").lower()
    m = re.search(r"(\d{3,4})\s*w", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d{3,4})", name)
    if m2 and any(tok in name for tok in ("rm", "psu", "w", "watt")):
        return int(m2.group(1))
    return 0


def estimate_component_wattage(comp: dict) -> int:
    """
    Heuristic estimate for component wattage.
    Uses direct fields first (power/tdp/wattage), else name/type heuristics.
    Improved tier mapping and more conservative defaults.
    """
    if not comp or not isinstance(comp, dict):
        return 0

    # direct keys to check (highest priority)
    for key in ("power", "tdp", "wattage", "tdp_w", "power_draw", "w"):
        if key in comp and comp.get(key) is not None:
            v = _parse_watt_value(comp.get(key))
            if v:
                return v

    name = (comp.get("name") or "").lower()
    ctype = (comp.get("type") or "").lower()

    # --- GPU heuristics (conservative but realistic) ---
    # Map common tokens to approximate typical whole-system GPU draw
    if ctype == "gpu" or "gpu" in name or name.startswith("rtx") or name.startswith("gtx") or "radeon" in name:
        # more granular estimates for modern cards
        if re.search(r"(4090|rx\s*7900|titan)", name):
            return 450
        if re.search(r"(4080|rx\s*7900\s*xt|rx\s*7800)", name):
            return 320
        if re.search(r"(4070|4070 ti|rx\s*7800\s*xt|rx\s*7700)", name):
            return 250
        if re.search(r"(4060|3060|6600|3060ti|3060 ti)", name):
            return 170
        if re.search(r"(3050|1650|1050|gtx\s*1650)", name):
            return 95
        # mid-range conservative default
        return 180

    # --- CPU heuristics ---
    if ctype == "cpu" or "ryzen" in name or re.search(r"\bi\d\b|\bi3\b|\bi5\b|\bi7\b|\bi9\b", name):
        # try to deduce cores -> rough TDP mapping
        cores_field = comp.get("cores") or comp.get("core_count") or ""
        cores = 0
        if isinstance(cores_field, str):
            m = re.search(r"(\d+)", cores_field)
            if m:
                cores = int(m.group(1))
        elif isinstance(cores_field, (int, float)):
            cores = int(cores_field)
        if cores >= 12:
            return 140
        if cores >= 8:
            return 95
        if cores >= 6:
            return 85
        if cores >= 4:
            return 65
        return 55

    # Motherboard
    if ctype == "motherboard" or any(k in name for k in ("mobo", "prime", "b450", "b550", "x570", "z690", "z790")):
        return 50

    # RAM
    if ctype == "ram" or "ddr" in name:
        # assume a stick draws ~4-8W; estimate module count if possible
        # fallback: 8W per module assumption
        modules = 1
        # try to detect "16GB x2" style
        m = re.search(r"(\b\d+)\s*x\s*(\d+)", name)
        if m:
            try:
                modules = int(m.group(2))
            except Exception:
                modules = 1
        return max(4, modules * 8)

    # Storage
    if ctype == "storage" or "ssd" in name or "nvme" in name:
        # NVMe ~5W, SATA SSD ~3-5W
        if "nvme" in name or "m.2" in name:
            return 5
        if "ssd" in name:
            return 4
        if "hdd" in name:
            return 8
        return 5

    # fans/peripherals
    if "fan" in name or "rgb" in name:
        return 4

    # fallback conservative estimate
    return 10


def compute_total_wattage_from_build(build: dict, include_headroom: bool = True, headroom_percent: int = 30) -> dict:
    """
    Compute components' watt estimates, total draw, and recommended PSU (rounded to 50W).
    Resolves component IDs -> dicts if necessary.
    Adds conservative minimums and ensures reasonable recommended PSU for systems with discrete GPUs.
    """
    comp_watts = {}
    total = 0

    # helper to get resolved component dict
    def _res(x):
        return _resolve_component_maybe(x, data)

    # storage may be list or ids
    storages = []
    if isinstance(build.get("storage"), list):
        storages = [_res(s) for s in build.get("storage")]
    elif build.get("storage"):
        storages = [_res(build.get("storage"))]

    keys = ["cpu", "gpu", "motherboard", "ram", "storage"]
    for k in keys:
        if k == "storage":
            s_total = 0
            for s in storages:
                s_total += estimate_component_wattage(s)
            comp_watts["storage"] = int(s_total)
            total += s_total
            continue

        comp_obj = _res(build.get(k))
        w = estimate_component_wattage(comp_obj)
        comp_watts[k] = int(w)
        total += w

    # extras (fans, peripherals, case, pumps, USB)
    # raise baseline extras to be more realistic
    extras = 60  # previous 15 -> more sensible baseline for a typical desktop rig
    comp_watts["extras"] = extras
    total += extras

    recommended = None
    if include_headroom:
        target = total * (1 + headroom_percent / 100.0)
        # round up to nearest 50
        recommended = int(math.ceil(target / 50.0) * 50)

        # Ensure reasonable minimum PSUs:
        # - If a discrete GPU is present, minimum recommended should be at least 450W
        # - For CPU-focused systems, at least 350W
        gpu_present = bool(build.get("gpu") or comp_watts.get("gpu", 0) >= 120)
        if gpu_present and recommended < 450:
            recommended = 450
        elif not gpu_present and recommended < 300:
            recommended = 300

    # ensure recommended is int
    return {
        "component_watts": comp_watts,
        "total_draw": int(total),
        "recommended_psu": int(recommended) if recommended is not None else None,
        "headroom_percent": headroom_percent,
    }


def _resolve_component_maybe(obj_or_id: Any, data: dict) -> dict:
    """
    If passed a dict, return it. If passed a chip-id (string like 'cpu:amd-ryzen-5-3600')
    or short name, attempt to resolve to the full component dict using dataset helpers.
    Returns {} if not resolvable.
    """
    if not obj_or_id:
        return {}
    if isinstance(obj_or_id, dict):
        return obj_or_id
    if isinstance(obj_or_id, str):
        # try chip id format 'type:slug' -> lookup by slug match
        parts = obj_or_id.split(":", 1)
        if len(parts) == 2:
            _type, slug = parts[0], parts[1]
            bucket = data.get(_type, {}) or {}
            for k, v in bucket.items():
                if slugify(k) == slug or slugify(v.get("name", "")) == slug:
                    return v
        # fallback: fuzzy best match
        best, _ = _best_match_in_dataset(obj_or_id, data)
        return best or {}
    # unknown type
    return {}


def build_to_tap_response_with_watts(build: Dict, build_id: str, include_headroom: bool = True, headroom_percent: int = 30) -> Dict:
    """Convert build into tapable chips with wattage + recommended PSU chips."""
    base_resp = build_to_tap_response(build, build_id)

    # Ensure top-level total_price & score are present and correct
    base_resp["total_price"] = build.get(
        "total_price") or build.get("total") or None
    base_resp["score"] = build.get("score")

    # compute watt info
    watt_info = compute_total_wattage_from_build(
        build, include_headroom=include_headroom, headroom_percent=headroom_percent)
    total_draw = watt_info.get("total_draw", 0)
    recommended = watt_info.get("recommended_psu", 0)

    # watt chip (don't use its "price" to represent currency)
    watt_chip = {
        "id": f"build-watt:{build_id}-{total_draw}w",
        "text": f"Estimated draw: {total_draw} W",
        "price": "",  # IMPORTANT: leave empty so frontend won't treat as ₱value
        "type": "build-watt",
        "meta": {"total_draw": total_draw, "recommended_psu": recommended}
    }

    recommended_chip = {
        "id": f"build-psu-reco:{build_id}-{recommended}w",
        "text": f"Recommended PSU: {recommended} W",
        "type": "psu-recommendation",
        "meta": {"recommended_watt": recommended}
    }

    suggested_raw = suggest_psus_for_wattage_varied(recommended, data, limit=4)
    suggested_chips = [_psu_chip_from_obj(p) for p in suggested_raw]

    # Append chips but ensure watt/reco chips appended AFTER component chips
    base_resp["chips"].append(watt_chip)
    base_resp["chips"].append(recommended_chip)
    base_resp["chips"].extend(suggested_chips)

    # also attach watt_info for frontend clarity
    base_resp["watt_info"] = watt_info

    return base_resp


# --------------------------
# Bottleneck analysis (CPU vs GPU) - short and plain-text output
# --------------------------

def _safe_int_from_field(val, default=0):
    try:
        if val is None:
            return default
        s = str(val)
        m = re.search(r"(\d+)", s.replace(",", ""))
        return int(m.group(1)) if m else default
    except Exception:
        return default


def estimate_cpu_capacity(cpu: dict) -> float:
    """
    Heuristic CPU capacity score (higher is better).
    Uses cores, clock (GHz), and an IPC factor guess from model name.
    Returns a positive float (arbitrary units).
    """
    if not cpu or not isinstance(cpu, dict):
        return 0.0

    # cores
    cores = _safe_int_from_field(
        cpu.get("cores", "") or cpu.get("core_count", "") or 0)
    if cores == 0:
        # try parsing strings like "6 cores"
        cores = _safe_int_from_field(cpu.get("cores", "0"))

    # clock in GHz (try to extract first floating number)
    clock = 0.0
    clock_field = cpu.get("clock") or cpu.get(
        "base_clock") or cpu.get("frequency") or ""
    try:
        if isinstance(clock_field, (int, float)):
            clock = float(clock_field)
        else:
            # find first float or int (e.g. "3.6 GHz / 4.2 GHz Boost")
            m = re.search(r"(\d+(?:\.\d+)?)", str(clock_field))
            if m:
                clock = float(m.group(1))
    except Exception:
        clock = 0.0

    # short name heuristics: newer architectures tend to have slightly better IPC
    name = (cpu.get("name") or "").lower()
    ipc_bonus = 1.0
    if any(tok in name for tok in ("ryzen 7", "ryzen 9", "i7", "i9", "14700", "13700")):
        ipc_bonus = 1.15
    elif any(tok in name for tok in ("ryzen 5", "i5", "5600", "10400")):
        ipc_bonus = 1.0
    elif any(tok in name for tok in ("i3", "celeron", "athlon")):
        ipc_bonus = 0.85

    # fallback if no cores info: assume 4
    if cores <= 0:
        cores = 4

    # capacity formula (arbitrary units)
    capacity = cores * max(0.8, clock) * ipc_bonus * 10.0
    return float(capacity)


def estimate_gpu_capacity(gpu: dict) -> float:
    """
    Heuristic GPU capacity score (higher is better).
    Uses vram in GB, a tier guess from model name, and a power/tdp hint.
    Returns positive float.
    """
    if not gpu or not isinstance(gpu, dict):
        return 0.0

    name = (gpu.get("name") or "").lower()
    # VRAM
    vram_field = gpu.get("vram") or gpu.get("memory") or ""
    vram_gb = _safe_int_from_field(vram_field, default=0)

    # tier hint by model tokens
    tier = 1.0
    if any(tok in name for tok in ("ti", "super", "rx 6800", "3080", "3090", "4070", "4080", "4090")):
        tier = 2.0
    elif any(tok in name for tok in ("3070", "3070ti", "6700", "3060ti", "3070")):
        tier = 1.6
    elif any(tok in name for tok in ("3060", "6600", "1660", "2060", "3050")):
        tier = 1.2
    elif any(tok in name for tok in ("1650", "1050", "gtx 1050", "gtx 960")):
        tier = 0.7

    # power hint
    power = _safe_int_from_field(
        gpu.get("power") or gpu.get("tdp") or gpu.get("wattage") or 0)

    # compute capacity: combine tier, vram and power
    capacity = (tier * 100.0) + (vram_gb * 8.0) + (min(power, 350) * 0.25)
    # ensure positive
    return float(max(10.0, capacity))


def estimate_workload_demand(resolution: str = "1080p", settings: str = "high", target_fps: int = 60) -> dict:
    """
    Returns a simple demand dict: {'cpu': X, 'gpu': Y}
    Units are same arbitrary scale as capacity heuristics so comparison works.
    - resolution: "1080p", "1440p", "4k"
    - settings: "low","medium","high","ultra"
    - target_fps: desired frames per second (int)
    """
    # base multipliers
    res_mult = {"1080p": 1.0, "1440p": 1.5, "4k": 2.4}
    set_mult = {"low": 0.7, "medium": 1.0, "high": 1.3, "ultra": 1.6}

    r = res_mult.get(resolution.lower(), 1.0)
    s = set_mult.get(settings.lower(), 1.0)
    fps = max(30, min(240, int(target_fps or 60)))

    # baseline demands (arbitrary)
    base_cpu = 120.0  # baseline CPU units for 1080p/medium/60fps
    base_gpu = 130.0  # baseline GPU units for 1080p/medium/60fps

    # GPU scales strongly with resolution & settings; CPU scales with fps & settings moderately
    demand_cpu = base_cpu * (0.8 + 0.5 * s) * (fps / 60.0)
    demand_gpu = base_gpu * r * s * (fps / 60.0)

    return {"cpu": float(demand_cpu), "gpu": float(demand_gpu)}


def analyze_bottleneck_text(cpu: dict, gpu: dict, resolution: str = "1080p", settings: str = "high", target_fps: int = 60) -> str:
    """
    Clean, plain-text bottleneck analysis with readable formatting.
    Example:
        [ Bottleneck Analysis ]
        CPU: Ryzen 5 5600X
        GPU: RTX 3060
        Resolution: 1080p (High, 60 FPS)
        → CPU Load: 78%
        → GPU Load: 82%
        Verdict: ✅ Balanced — both components perform similarly.
        Summary: Your CPU and GPU are well-matched for this workload.
    """
    # --- Estimate performance values ---
    cpu_cap = estimate_cpu_capacity(cpu)
    gpu_cap = estimate_gpu_capacity(gpu)
    demand = estimate_workload_demand(
        resolution=resolution, settings=settings, target_fps=target_fps)

    cpu_load_pct = int(
        round((demand["cpu"] / cpu_cap) * 100)) if cpu_cap > 0 else 0
    gpu_load_pct = int(
        round((demand["gpu"] / gpu_cap) * 100)) if gpu_cap > 0 else 0

    # --- Decide verdict ---
    diff = gpu_load_pct - cpu_load_pct
    abs_diff = abs(diff)

    if cpu_load_pct >= 110 and cpu_load_pct - gpu_load_pct >= 10:
        verdict = "⚠️ CPU Bottleneck — Processor is overloaded."
        summary = "The CPU limits overall performance. Consider upgrading your CPU or lowering CPU-heavy settings/FPS target."
    elif gpu_load_pct >= 110 and gpu_load_pct - cpu_load_pct >= 10:
        verdict = "⚠️ GPU Bottleneck — Graphics card is overloaded."
        summary = "The GPU limits performance. Consider lowering resolution or upgrading your GPU."
    elif abs_diff <= 10:
        verdict = "✅ Balanced — both components perform similarly."
        summary = "Your CPU and GPU are well-matched for this workload."
    elif diff > 10:
        verdict = "⚠️ GPU Bottleneck — GPU more utilized."
        summary = "The GPU is the limiting factor. Lower graphical settings or upgrade your GPU."
    else:
        verdict = "⚠️ CPU Bottleneck — CPU more utilized."
        summary = "The CPU is the limiting factor. Lower CPU-heavy settings or upgrade your CPU."

    # --- Build clean formatted output ---
    cpu_name = cpu.get("name", "Unknown CPU").strip()
    gpu_name = gpu.get("name", "Unknown GPU").strip()
    settings_label = settings.capitalize()

    result = f"""


CPU: {cpu_name}
GPU: {gpu_name}
Resolution: {resolution} ({settings_label} Settings, {target_fps} FPS Target)

→ CPU Load: {cpu_load_pct}%
→ GPU Load: {gpu_load_pct}%

Verdict: {verdict}
Summary: {summary}
""".strip()

    return result


# --------------------------
# CLI Tester
# --------------------------
if __name__ == "__main__":
    print("💡 ARsemble AI Assistant with Gemini 2.5 Flash-Lite fallback\n")
    print("CLI shortcuts: type 'tap <chip-name>' or 'lookup <chip-name>' to view component details.")
    print("Commands: wattage <chip...>, wattage auto <free-text>, and ask 'What PSU do I need for ...?'\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ["exit", "quit"]:
            break

        # -----------------------
        # tap / lookup
        # -----------------------
        if user.lower().startswith(("tap ", "lookup ")):
            parts = user.split(maxsplit=1)
            if len(parts) < 2:
                print(
                    "Usage: tap <chip_id_or_name> (e.g. tap amd-ryzen-5-3600 or tap AMD Ryzen 5 3600)\n")
                continue
            raw = parts[1].strip()
            details = get_component_details(raw)
            if not details.get("found"):
                print(f"[LOOKUP] {details.get('error')}\n")
                continue
            # Pretty-print for CLI
            print(
                f"[COMPONENT] {details['name']} — {details.get('price', 'N/A')}")
            for k, v in details["specs"].items():
                print(f"  {k}: {v}")
            print()
            continue

        # -----------------------
        # wattage <chip ids or names>
        # -----------------------
        if user.lower().startswith(("wattage ", "watt ")):
            parts = user.split()[1:]
            if not parts:
                print(
                    "Usage: wattage <chip-id-or-name> [<chip-id-or-name> ...]\n")
                continue
            build = build_from_chip_ids(parts)
            result = compute_total_wattage_from_build(
                build, include_headroom=True, headroom_percent=30)
            print("[WATTAGE] Component estimates:")
            for k, v in result["component_watts"].items():
                print(f"  {k}: {v} W")
            print(f"Estimated total system draw: {result['total_draw']} W")
            print(
                f"Recommended PSU (with {result['headroom_percent']}% headroom): {result['recommended_psu']} W\n")
            continue

# --------------------------
# DEBUG: list all PSUs in dataset
# --------------------------
        print("\n--- DEBUG: listing all PSU entries ---")
        for k, v in data.get("psu", {}).items():
            print("PSU:", v.get("name"), "-> wattage:", v.get("wattage",
                  v.get("power", "")), "price:", v.get("price"))
            print("--- END of PSU list ---\n")

        # -----------------------
        # wattage auto <free text>
        # -----------------------
        if user.lower().startswith(("wattage auto ", "wattage-auto ", "wattageauto ")):
            desc_parts = user.split(None, 2)
            if len(desc_parts) < 3:
                print("Usage: wattage auto <free-form build description>\nExample: wattage auto 'Ryzen 5 3600 with RTX 3060, MSI B450, 16GB Corsair, 1TB NVMe'\n")
                continue
            free_text = desc_parts[2]
            parsed_build = parse_build_from_text(
                free_text, data, verbose=False)
            if not parsed_build:
                print("Could not auto-parse any components from that description.\n")
                continue
            print("[AUTO-PARSE] Detected components:")
            for k, v in parsed_build.items():
                if k == "storage" and isinstance(v, list):
                    for s in v:
                        print(f"  storage: {s.get('name')}")
                else:
                    print(
                        f"  {k}: {v.get('name') if isinstance(v, dict) else v}")
            result = compute_total_wattage_from_build(
                parsed_build, include_headroom=True, headroom_percent=30)
            print("\n[WATTAGE] Component estimates:")
            for k, v in result["component_watts"].items():
                print(f"  {k}: {v} W")
            print(f"Estimated total system draw: {result['total_draw']} W")
            print(
                f"Recommended PSU (with {result['headroom_percent']}% headroom): {result['recommended_psu']} W\n")
            continue

        # -----------------------
        # Natural-language PSU question handler (e.g., "What PSU do I need for RTX 3060 + Ryzen 5 5600X?")
        # -----------------------
        if re.search(r"\b(psu|power supply|psu do i need|psu needed|what power supply)\b", user.lower()):
            res = recommend_psu_for_query_with_chips(
                user, data, headroom_percent=30)
            if res.get("error"):
                print(f"[PSU] {res['error']}\n")
                continue

            print(f"[PSU] Detected: {res.get('detected_str', '')}")
            for k, v in (res.get('component_watts') or {}).items():
                print(f"  {k}: {v} W")
            print(f"Total Draw: {res.get('total_draw')} W")
            print(f"Recommended PSU: {res.get('recommended_psu')} W\n")

            # suggested chips (tapable)
            if res.get("suggested_psu_chips"):
                print("Suggested PSUs (tapable chips):")
                for chip in res.get("suggested_psu_chips", []):
                    print(
                        f"  {chip['id']} -> {chip['text']} — {chip.get('price', '')}")
            # recommended chip
            if res.get("recommended_chip"):
                rc = res.get("recommended_chip")
                print(f"  {rc['id']} -> {rc['text']}")
            print()
            continue

        # -----------------------
        # Normal request path (recommendations / fallback)
        # -----------------------
        reply = get_ai_response(user)

        # Print handling that supports multiple response types
        src = reply.get("source", "unknown")

        # Simple text responses (primary / gemini)
        if src in ("primary", "gemini-fallback"):
            print(f"[{src.upper()}] {reply.get('text', '')}\n")
            continue

        # Local budget recommendation (tapable chips) — cleaned, spaced formatting
        if src == "local-recommendation":
            print(f"[{src.upper()}] budget: ₱{reply.get('budget')}\n")
            for res in reply.get("results", []):
                # Header line (Estimated total + score)
                print(f"  [{res.get('message_id')}] {res.get('message')}\n")

                chips = res.get("chips", []) or []
                if not chips:
                    print("   (no component chips)\n")
                    continue

                # Group chips by type
                from collections import defaultdict
                grouped = defaultdict(list)
                for c in chips:
                    ctype = (c.get("type") or "").lower()
                    grouped[ctype].append(c)

                # Print main components in friendly order with spacing
                main_order = [("CPU", "cpu"), ("GPU", "gpu"), ("Motherboard", "motherboard"),
                              ("RAM", "ram"), ("Storage", "storage")]
                for label, key in main_order:
                    items = grouped.get(key, [])
                    for it in items:
                        text = it.get("text") or it.get("id")
                        price = it.get("price", "")
                        print(f"   {label}: {text} — {price}")

                # Print selected PSU (if any) once
                psu_items = grouped.get("psu", [])
                main_psu = psu_items[0] if psu_items else None
                if main_psu:
                    print(
                        f"   PSU: {main_psu.get('text')} — {main_psu.get('price', '')}")

                # Blank line before watt summary
                print()

                # Extract watt & recommendation chips
                watt_chip = None
                reco_chip = None
                for c in chips:
                    ctype = (c.get("type") or "").lower()
                    cid = (c.get("id") or "").lower()
                    if ctype == "build-watt" or cid.startswith("build-watt"):
                        watt_chip = c
                    if ctype == "psu-recommendation" or cid.startswith("psu:recommended") or cid.startswith("build-psu-reco"):
                        reco_chip = c

                # Watt summary block
                if watt_chip:
                    meta = watt_chip.get("meta", {}) or {}
                    total_draw = meta.get(
                        "total_draw") or watt_chip.get("text")
                    recommended_w = meta.get("recommended_psu") or (
                        reco_chip and reco_chip.get("meta", {}).get("recommended_watt"))
                    print(f"   Estimated draw: {total_draw} W")
                    if recommended_w:
                        print(
                            f"   Recommended PSU (with headroom): {recommended_w} W")
                    print()  # spacing after watt summary

                # Explicit recommended chip text (if present)
                if reco_chip:
                    print(f"   {reco_chip.get('text')}\n")

                # Suggested PSUs: collect and dedupe (exclude main_psu)
                suggested_raw = []
                for p in (grouped.get("psu", []) or []):
                    if p is not main_psu:
                        suggested_raw.append(p)
                for c in chips:
                    if (c.get("type") or "").lower() == "psu" and c not in suggested_raw and c is not main_psu:
                        suggested_raw.append(c)

                # dedupe by (name, watt)
                seen = set()
                suggested = []
                for p in suggested_raw:
                    meta = p.get("meta") or {}
                    name = (meta.get("name") or p.get("text") or "").strip()
                    watt = _parse_psu_wattage(
                        meta) or _parse_watt_value(p.get("text") or "")
                    key = (name.lower(), int(watt) if watt else 0)
                    if key not in seen and name:
                        seen.add(key)
                        suggested.append(p)
                    if len(suggested) >= 6:
                        break

                if suggested:
                    print("   Suggested PSUs:")
                    for p in suggested:
                        print(f"     • {p.get('text')} — {p.get('price', '')}")
                    print()

                print()  # blank line between builds
            continue

        # Component lookup / other structured responses (if returned directly)
        if reply.get("found") is True and reply.get("component"):
            comp = reply["component"]
            print(f"[COMPONENT] {comp.get('name')}")
            for k, v in comp.items():
                if k == "name":
                    continue
                print(f"  {k}: {v}")
            print()
            continue

        # Local compatibility structured response
        if reply.get("source") == "local-compatibility":
            if reply.get("results"):
                print(
                    f"[LOCAL-COMPATIBILITY] {reply.get('target')} -> {reply.get('compatible_type')}")
                for chip in reply.get("results", []):
                    print(f"  • {chip.get('text')} — {chip.get('price')}")
                print()
            else:
                print(f"[LOCAL-COMPATIBILITY] {reply.get('message')}\n")
            continue

        # Fallback: dump the whole reply so you can debug
        try:
            print(json.dumps(reply, ensure_ascii=False, indent=2))
        except Exception:
            print(str(reply))
        print()

# --- Initialize default shops if not existing ---
default_shops = {
    "smfp_computer": {
        "name": "SMFP Computer",
        "address": "594 J Nepomuceno St, Quiapo, Manila, 1001 Metro Manila",
        "region": "Metro Manila",
        "public": True,
        "notes": "Client shop provided via chat."
    },
    "pc_express": {
        "name": "PC Express",
        "address": "Gilmore Avenue, New Manila, Quezon City, Metro Manila",
        "region": "Metro Manila",
        "public": True,
        "notes": "Major national PC retailer with multiple branches."
    },
    "dynquestpc": {
        "name": "DynaQuest PC",
        "address": "SM City North EDSA Cyberzone, Quezon City, Metro Manila",
        "region": "Metro Manila",
        "public": True,
        "notes": "Well-known tech store offering gaming parts and peripherals."
    },
    "easypc": {
        "name": "EasyPC",
        "address": "Gilmore IT Center, Aurora Blvd, Quezon City, Metro Manila",
        "region": "Metro Manila",
        "public": True,
        "notes": "Popular chain known for affordable PC parts and online ordering."
    },
    "pchub": {
        "name": "PC Hub",
        "address": "Gilmore Ave, Quezon City, Metro Manila",
        "region": "Metro Manila",
        "public": True,
        "notes": "Trusted PC parts shop in Gilmore with custom build services."
    }
}

shops = _load_shops()
if not shops:
    for sid, info in default_shops.items():
        add_shop(sid, info)
