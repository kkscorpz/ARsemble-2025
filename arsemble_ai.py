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
            "amd": ("amd", "ryzen", "am4", "am5"),
            # GPU vendor / series tokens (important for GPU matching)
            "nvidia": ("nvidia", "geforce", "gtx", "rtx"),
            "amd_gpu": ("radeon", "rx", "vega")
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


def score_build(cpu, gpu, motherboard, ram, storage, psu, usage: str = "general", budget: int = None) -> int:
    """Enhanced scoring that considers budget appropriateness."""
    score = 0
    try:
        # Base performance scoring (existing logic)
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

    # Budget efficiency bonus
    if budget:
        total_price = sum(parse_price(x.get("price", "0"))
                          for x in [cpu, gpu, motherboard, ram, storage, psu] if x)
        budget_ratio = total_price / budget
        if 0.8 <= budget_ratio <= 1.0:  # Ideal: 80-100% of budget
            score += 20
        elif 0.6 <= budget_ratio < 0.8:  # Good: 60-80% of budget
            score += 10
        elif budget_ratio > 1.0:  # Over budget penalty
            score -= 15

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
    Generate budget-appropriate builds with intelligent component selection.
    """
    builds = []

    # Get all components
    all_cpus = list(data.get("cpu", {}).values())
    all_gpus = list(data.get("gpu", {}).values())
    all_mobos = list(data.get("motherboard", {}).values())
    all_rams = list(data.get("ram", {}).values())
    all_storages = list(data.get("storage", {}).values())
    all_psus = list(data.get("psu", {}).values())

    def safe_price(x):
        try:
            return parse_price(x.get("price", "0"))
        except Exception:
            return 0

    # Sort all components by price
    all_cpus.sort(key=safe_price)
    all_gpus.sort(key=safe_price)
    all_mobos.sort(key=safe_price)
    all_rams.sort(key=safe_price)
    all_storages.sort(key=safe_price)
    all_psus.sort(key=safe_price)

    def select_components_by_budget_range(components, budget_range, min_items=3, max_items=8):
        """Select components appropriate for the budget range"""
        if not components:
            return []

        low, high = budget_range
        filtered = []

        for comp in components:
            price = safe_price(comp)
            if low <= price <= high:
                filtered.append(comp)

        # If not enough in range, expand selection
        if len(filtered) < min_items:
            # Add cheaper components
            cheaper = [c for c in components if safe_price(c) < low]
            needed = min_items - len(filtered)
            filtered.extend(cheaper[:needed])

            # If still not enough, add more expensive
            if len(filtered) < min_items:
                expensive = [c for c in components if safe_price(c) > high]
                needed = min_items - len(filtered)
                filtered.extend(expensive[:needed])

        return filtered[:max_items]

    # Define budget ranges for different budget levels
    if budget <= 25000:
        # Ultra budget builds
        ranges = {
            'cpu': (1000, 6000),
            'gpu': (2000, 10000),
            'mobo': (1000, 5000),
            'ram': (1000, 3000),
            'storage': (1000, 4000),
            'psu': (1000, 3000)
        }
        sample_sizes = {'cpu': 6, 'gpu': 6, 'mobo': 4,
                        'ram': 4, 'storage': 3, 'psu': 3}

    elif budget <= 40000:
        # Mid-range budget
        ranges = {
            'cpu': (3000, 12000),
            'gpu': (8000, 20000),
            'mobo': (3000, 8000),
            'ram': (2000, 6000),
            'storage': (2000, 6000),
            'psu': (2000, 5000)
        }
        sample_sizes = {'cpu': 8, 'gpu': 8, 'mobo': 6,
                        'ram': 5, 'storage': 4, 'psu': 4}

    elif budget <= 60000:
        # Higher budget
        ranges = {
            'cpu': (6000, 20000),
            'gpu': (15000, 35000),
            'mobo': (5000, 12000),
            'ram': (3000, 8000),
            'storage': (3000, 8000),
            'psu': (3000, 7000)
        }
        sample_sizes = {'cpu': 6, 'gpu': 6, 'mobo': 5,
                        'ram': 4, 'storage': 3, 'psu': 3}

    else:
        # Premium budget
        ranges = {
            'cpu': (10000, 30000),
            'gpu': (25000, 60000),
            'mobo': (8000, 20000),
            'ram': (5000, 12000),
            'storage': (5000, 15000),
            'psu': (5000, 10000)
        }
        sample_sizes = {'cpu': 5, 'gpu': 5, 'mobo': 4,
                        'ram': 3, 'storage': 3, 'psu': 3}

    # Select components for this budget range
    cpus = select_components_by_budget_range(all_cpus, ranges['cpu'],
                                             min_items=3, max_items=sample_sizes['cpu'])
    gpus = select_components_by_budget_range(all_gpus, ranges['gpu'],
                                             min_items=3, max_items=sample_sizes['gpu'])
    mobos = select_components_by_budget_range(all_mobos, ranges['mobo'],
                                              min_items=3, max_items=sample_sizes['mobo'])
    rams = select_components_by_budget_range(all_rams, ranges['ram'],
                                             min_items=3, max_items=sample_sizes['ram'])
    storages = select_components_by_budget_range(all_storages, ranges['storage'],
                                                 min_items=2, max_items=sample_sizes['storage'])
    psus = select_components_by_budget_range(all_psus, ranges['psu'],
                                             min_items=3, max_items=sample_sizes['psu'])

    # Debug logging
    logger.info(
        f"Budget {budget}: Selected {len(cpus)} CPUs, {len(gpus)} GPUs, {len(mobos)} Mobos")

    # Generate builds with compatibility check
    build_count = 0
    max_builds = 500  # Reasonable limit

    for cpu in cpus:
        for gpu in gpus:
            cpu_price = safe_price(cpu)
            gpu_price = safe_price(gpu)

            # Quick price check before deeper iteration
            if cpu_price + gpu_price > budget * 0.8:  # CPU+GPU shouldn't consume >80%
                continue

            for mobo in mobos:
                for ram in rams:
                    for storage in storages:
                        for psu in psus:
                            if build_count >= max_builds:
                                break

                            try:
                                total = sum(safe_price(x) for x in [
                                            cpu, gpu, mobo, ram, storage, psu])
                            except Exception:
                                continue

                            # Strict budget enforcement with 5% tolerance
                            if total > budget * 1.05:
                                continue

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
                                "score": score_build(cpu, gpu, mobo, ram, storage, psu, usage, budget)
                            })
                            build_count += 1

    # Enhanced scoring that considers budget utilization
    def build_sort_key(build):
        total = build.get("total_price", 0)
        score = build.get("score", 0)

        # Prefer builds that use 85-100% of budget (good value)
        budget_utilization = total / budget if budget > 0 else 0
        utilization_score = 0

        if 0.85 <= budget_utilization <= 1.0:
            utilization_score = 20
        elif 0.7 <= budget_utilization < 0.85:
            utilization_score = 10
        elif budget_utilization > 1.0:
            utilization_score = -10

        # Higher score first, then cheaper
        return (-(score + utilization_score), -total)

    builds_sorted = sorted(builds, key=build_sort_key)

    # Log results for debugging
    if builds_sorted:
        logger.info(
            f"Generated {len(builds_sorted)} builds for budget {budget}")
        for i, build in enumerate(builds_sorted[:3]):
            logger.info(
                f"Build {i+1}: ₱{build['total_price']} (score: {build['score']})")
    else:
        logger.warning(f"No builds generated for budget {budget}")

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

    Special behavior:
      - If the user query appears to ask a "bottleneck" question, send a short, strict prompt
        that forces Gemini to output the compact bottleneck format:
          → CPU Load: ~<N>%  |  GPU Load: ~<M>%
          Verdict: <one-line verdict>
          Explanation: <1-2 short sentences, plus 1 suggestion>
      - Otherwise use the previous general fallback behaviour (concise factual answers,
        pricing rules, context snippet, and postprocessing).
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # safe context snippet (trim to keep prompt small)
        try:
            context_snippet = json.dumps(context_data or {}, indent=2)[:3000]
        except Exception:
            context_snippet = "{}"

        user_lower = (user_input or "").lower()

        # Detect bottleneck-specific queries
        is_bottleneck_q = bool(re.search(
            r"\bbottleneck\b|\bwill .* bottleneck\b|\bbetween\b.*\b(and|vs|vs\.)\b", user_lower))

        # Shared base instructions (no greetings, concise, avoid forbidden phrases)
        base = (
            "You are ARsemble AI — a concise, factual PC-building assistant. "
            "Reply in a casual-but-academic tone (no greetings). "
            "Do NOT use phrases like 'dataset', 'I don't have information', 'not in my dataset', or 'as of my last update'. "
            "If uncertain, give a brief likely answer and a practical check the user can perform."
        )

        # Pricing rules (enforced everywhere)
        price_instructions = (
            "\n\nIMPORTANT PRICING RULES (MUST FOLLOW):\n"
            "- If you mention prices, ALWAYS use Philippine peso and the ₱ symbol (e.g., ₱4,000). Do NOT use $ or USD.\n"
            "- When giving prices, provide only approximate ranges (e.g., ₱4,000–₱6,000). Do NOT output a precise single client price.\n"
            "- Round to the nearest 100 or 500 and keep ranges short (example: ₱4,500–₱6,000).\n"
            "- If you are unsure about local pricing, say 'Check local retailers for current pricing' (but still give a short rounded range).\n"
        )

        if is_bottleneck_q:
            # Strict bottleneck prompt — forces compact, predictable format
            bottleneck_prompt = (
                base
                + "\n\nBottleneck Task:\n"
                f"- The user asked: \"{user_input}\"\n"
                "- Determine which component (CPU or GPU) is the bottleneck between the two components mentioned.\n"
                "- Output EXACTLY the following compact format (no extra commentary):\n"
                "  1) One line with estimated loads: \"→ CPU Load: ~<N>%  |  GPU Load: ~<M>%\"\n"
                "  2) One short verdict line starting with \"Verdict:\" and one of: \"⚠️ CPU Bottleneck\", \"⚠️ GPU Bottleneck\", or \"✅ Balanced\". You may add a very short explanation after the verdict separated by ' — '.\n"
                "  3) One short Explanation line (1–2 sentences) beginning with \"Explanation:\" that explains concisely why and includes a single quick suggestion (e.g., upgrade CPU, upgrade GPU, lower settings).\n"
                "- Use reasonable approximate percentages for gaming at 1080p / High / 60 FPS unless the user specified a different resolution/settings; round percentages to nearest 5 or 10. Keep it short.\n"
                "- Do NOT mention datasets, model limitations, or internal state. Do NOT ask for clarification. If component names are ambiguous, make a best-effort assumption.\n"
                f"{price_instructions}\n"
                "Context (if relevant):\n"
                f"{context_snippet}\n"
            )

            # attach short chat history optionally
            if chat_history:
                try:
                    recent = chat_history[-6:]
                    hist_lines = ["\nChat history (most recent last):"]
                    for m in recent:
                        hist_lines.append(
                            f"{m.get('role', 'user').upper()}: {m.get('text') or m.get('message', '')}")
                    bottleneck_prompt += "\n" + "\n".join(hist_lines)
                except Exception:
                    logger.exception(
                        "Failed to attach chat history to bottleneck prompt.")

            try:
                response = model.generate_content(
                    bottleneck_prompt, temperature=0.05, max_output_tokens=160)
            except TypeError:
                try:
                    response = model.generate_content(bottleneck_prompt)
                except Exception:
                    logger.exception("Gemini bottleneck call failed.")
                    return "Sorry — I'm unable to generate that right now."

            raw = getattr(response, "text", None) or str(response)
            clean = raw.strip()

            # Minimal cleanup: strip markdown, remove greeting words, ensure lines are short
            clean = re.sub(
                r'^(hi|hello|hey)[\s,\!\.]+', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean, flags=re.DOTALL)
            clean = re.sub(r'\*(.*?)\*', r'\1', clean, flags=re.DOTALL)
            clean = re.sub(r'(?m)^[\*\-\+]\s+', '', clean)

            # Ensure the output begins with the load line. If not, heuristically extract numbers and reconstruct.
            lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
            if not lines:
                return "Sorry — I couldn't produce a bottleneck summary."

            # If first line doesn't look like load line, try to find a load-like substring
            if not re.search(r"cpu load[:\s]", lines[0], flags=re.IGNORECASE) and not re.search(r"→\s*cpu", lines[0], flags=re.IGNORECASE):
                # search for a line containing 'cpu' and a percent, or two percents
                found_idx = None
                for i, ln in enumerate(lines):
                    if re.search(r"\bcpu\b.*\d+%|\d+%.*\bcpu\b", ln, flags=re.IGNORECASE) or len(re.findall(r"\d+%", ln)) >= 2:
                        found_idx = i
                        break
                if found_idx is not None:
                    # rotate lines to put found_idx first
                    lines = lines[found_idx:] + lines[:found_idx]

            # join first 3 meaningful lines (truncate excessive content)
            compact = "\n".join(lines[:3])

            # Post-process currency/unicode rules (same logic as general path)
            try:
                if "\\u" in compact:
                    try:
                        compact = compact.encode(
                            "utf-8").decode("unicode_escape")
                    except Exception:
                        compact = compact.replace("\\u20b1", "₱")
                compact = re.sub(r"\$\s?([0-9][0-9,\.]*)", r"₱\1", compact)
                compact = re.sub(r"\bUSD\b", "₱", compact, flags=re.IGNORECASE)
                compact = re.sub(r"\bdollars?\b", "pesos",
                                 compact, flags=re.IGNORECASE)

                # expand exact single prices into ranges if present
                def expand_exact_to_range(match):
                    num = match.group(1)
                    digits = re.sub(r"[^\d]", "", num)
                    if not digits:
                        return match.group(0)
                    v = int(digits)
                    lo = int(round(v * 0.85 / 100.0) * 100)
                    hi = int(round(v * 1.15 / 100.0) * 100)
                    return f"₱{lo:,d}–₱{hi:,d}"

                compact = re.sub(
                    r"₱\s?([0-9]{3,7}(?:,[0-9]{3})?)\b(?!\s*[-–—])", expand_exact_to_range, compact)
            except Exception:
                logger.exception(
                    "Postprocessing of bottleneck text failed; returning raw cleaned content.")

            return compact.strip()

        # Non-bottleneck flow: general concise prompt (existing behavior)
        is_latest_query = bool(re.search(
            r"\b(latest|newest|what's new|what are the latest|recently released|released in 2025|2025|now 2025)\b",
            user_lower,
            flags=re.IGNORECASE,
        ))
        is_specs_query = any(k in user_lower for k in [
                             "details", "specs", "specifications", "show me", "what are the specs", "how much", "price", "cost"])
        is_build_query = any(k in user_lower for k in [
                             "build", "recommend", "pc build", "₱", "php", "peso", "budget", "recommendation"])

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
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean, flags=re.DOTALL)
            clean = re.sub(r'\*(.*?)\*', r'\1', clean, flags=re.DOTALL)
            clean = re.sub(r'(?m)^[\*\-\+]\s+', '• ', clean)
            clean = clean.replace('*', '')
            clean = re.sub(r'(?m)^\s*•\s*•+\s*', '• ', clean)
            clean = re.sub(r'\n{3,}', '\n\n', clean)
        except Exception:
            logger.exception(
                "Markdown cleanup postprocessing failed; continuing.")

        # --- Postprocessing for currency / unicode escapes ---
        try:
            if "\\u" in clean:
                try:
                    clean = clean.encode("utf-8").decode("unicode_escape")
                except Exception:
                    clean = clean.replace("\\u20b1", "₱")
            clean = re.sub(r"\$\s?([0-9][0-9,\.]*)", r"₱\1", clean)
            clean = re.sub(r"\bUSD\b", "₱", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\bdollars?\b", "pesos",
                           clean, flags=re.IGNORECASE)

            def expand_exact_to_range(match):
                num = match.group(1)
                digits = re.sub(r"[^\d]", "", num)
                if not digits:
                    return match.group(0)
                v = int(digits)
                lo = int(round(v * 0.85 / 100.0) * 100)
                hi = int(round(v * 1.15 / 100.0) * 100)
                return f"₱{lo:,d}–₱{hi:,d}"

            clean = re.sub(
                r"₱\s?([0-9]{3,7}(?:,[0-9]{3})?)\b(?!\s*[-–—])", expand_exact_to_range, clean)
            clean = re.sub(r"\n{3,}", "\n\n", clean)
        except Exception:
            logger.exception(
                "Postprocessing of Gemini text failed; returning raw cleaned text.")

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
     - PSU detection (improved: headroom parsing, structured output, Gemini fallback)
     - compatibility detection (local resolver, fallback to Gemini public_data)
     - bottleneck analysis (robust parsing + Gemini fallback)
     - budget builds / lists / local lookup
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
        try:
            if any(k in lower_q for k in ("details", "specs", "how much", "price", "cost")) or re.search(r"\bdetails about\b", lower_q):
                local_lookup = get_component_details(q)
                if local_lookup and local_lookup.get("found"):
                    price = (local_lookup.get("price") or "").strip()
                    if price and re.search(r"\d", price):
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

        # ---------- 0) Bottleneck natural-language detection (robust replacement) ----------
        if "bottleneck" in lower_q:
            try:
                logger.info("🔍 [BOTTLENECK] analyzing query: %s", q)

                def normalize_joined_tokens(s: str) -> str:
                    if not s:
                        return s
                    s = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", s)
                    s = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", s)
                    s = re.sub(r"[_\-]+", " ", s)
                    return re.sub(r"\s+", " ", s).strip()

                def looks_like_cpu_obj(obj):
                    if not obj:
                        return False
                    name = (obj.get("name") or "").lower()
                    ctype = (obj.get("type") or "").lower()
                    if "cpu" in ctype:
                        return True
                    return bool(re.search(r"\b(ryzen|intel|core\s*i|i[0-9]{1,2}\b|xeon|athlon|processor)\b", name))

                def looks_like_gpu_obj(obj):
                    if not obj:
                        return False
                    name = (obj.get("name") or "").lower()
                    ctype = (obj.get("type") or "").lower()
                    if "gpu" in ctype or "graphics" in ctype:
                        return True
                    return bool(re.search(r"\b(rtx|gtx|geforce|radeon|rx|vga|titan|nvidia|amd|arc|iris)\b", name))

                # 1) Extract two components
                a_raw = b_raw = None
                for pat in [
                    r"bottleneck(?:\s+of)?\s+(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+)$",
                    r"which.*bottleneck.*(?:between)?\s+(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+)$",
                    r"(.+?)\s+(?:and|\+|vs|vs\.)\s+(.+?)\s+.*bottleneck"
                ]:
                    m = re.search(pat, q, flags=re.IGNORECASE)
                    if m:
                        a_raw = m.group(1).strip(" \"'")
                        b_raw = m.group(2).strip(" \"'")
                        break

                # fallback split
                if not (a_raw and b_raw):
                    if "+" in q:
                        parts = [p.strip() for p in q.split("+", 1)]
                        if len(parts) >= 2:
                            a_raw, b_raw = parts[0], parts[1]
                    elif " and " in lower_q:
                        parts = [p.strip() for p in re.split(
                            r"\band\b", q, flags=re.IGNORECASE) if p.strip()]
                        if len(parts) >= 2:
                            a_raw, b_raw = parts[0], parts[1]

                a_raw = normalize_joined_tokens(a_raw) if a_raw else None
                b_raw = normalize_joined_tokens(b_raw) if b_raw else None

                logger.info(
                    "✅ [BOTTLENECK] extracted raw tokens: a_raw=%r, b_raw=%r", a_raw, b_raw)

                if not a_raw or not b_raw:
                    # forward to Gemini for free-form handling
                    logger.warning(
                        "⚠️ [BOTTLENECK] Could not extract two components — forwarding to Gemini.")
                    gem_text = gemini_fallback_with_data(
                        user_input, make_public_data(data))
                    return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

                # 2) direct chip-id lookup
                try:
                    a_obj = lookup_component_by_chip_id(a_raw) or None
                except Exception:
                    a_obj = None
                try:
                    b_obj = lookup_component_by_chip_id(b_raw) or None
                except Exception:
                    b_obj = None

                # 3) fuzzy best-match in dataset
                if not a_obj:
                    a_obj, _ = _best_match_in_dataset(a_raw, data)
                if not b_obj:
                    b_obj, _ = _best_match_in_dataset(b_raw, data)

                logger.info("🔎 [BOTTLENECK] best-match results: A=%s, B=%s",
                            (a_obj.get("name") if a_obj else None), (b_obj.get("name") if b_obj else None))

                # 4) assign CPU/GPU with heuristics
                def assign_cpu_gpu(candidate1, candidate2):
                    if looks_like_cpu_obj(candidate1) and looks_like_gpu_obj(candidate2):
                        return candidate1, candidate2
                    if looks_like_cpu_obj(candidate2) and looks_like_gpu_obj(candidate1):
                        return candidate2, candidate1

                    if looks_like_cpu_obj(candidate1) and not looks_like_cpu_obj(candidate2):
                        return candidate1, candidate2
                    if looks_like_cpu_obj(candidate2) and not looks_like_cpu_obj(candidate1):
                        return candidate2, candidate1

                    if looks_like_gpu_obj(candidate1) and not looks_like_gpu_obj(candidate2):
                        if looks_like_cpu_obj(candidate2):
                            return candidate2, candidate1
                        return candidate1, candidate2
                    if looks_like_gpu_obj(candidate2) and not looks_like_gpu_obj(candidate1):
                        if looks_like_cpu_obj(candidate1):
                            return candidate1, candidate2
                        return candidate2, candidate1

                    n1 = (candidate1.get("name") if candidate1 else "") or ""
                    n2 = (candidate2.get("name") if candidate2 else "") or ""
                    cnt_cpu_1 = len(re.findall(
                        r"\b(i\d|core|ryzen|xeon|athlon|processor)\b", n1.lower()))
                    cnt_cpu_2 = len(re.findall(
                        r"\b(i\d|core|ryzen|xeon|athlon|processor)\b", n2.lower()))
                    cnt_gpu_1 = len(re.findall(
                        r"\b(rtx|gtx|geforce|radeon|rx|vga|nvidia|amd)\b", n1.lower()))
                    cnt_gpu_2 = len(re.findall(
                        r"\b(rtx|gtx|geforce|radeon|rx|vga|nvidia|amd)\b", n2.lower()))

                    score1 = cnt_cpu_1 * 2 + cnt_gpu_1
                    score2 = cnt_cpu_2 * 2 + cnt_gpu_2
                    if score1 > score2:
                        return candidate1, candidate2
                    if score2 > score1:
                        return candidate2, candidate1

                    if candidate1 and 'cpu' in ((candidate1.get('type') or '').lower()):
                        if candidate2 and 'gpu' in ((candidate2.get('type') or '').lower()):
                            return candidate1, candidate2
                    if candidate2 and 'cpu' in ((candidate2.get('type') or '').lower()):
                        if candidate1 and 'gpu' in ((candidate1.get('type') or '').lower()):
                            return candidate2, candidate1

                    return candidate1, candidate2

                cpu_obj, gpu_obj = assign_cpu_gpu(a_obj, b_obj)
                if not cpu_obj or not gpu_obj:
                    cpu_obj, gpu_obj = assign_cpu_gpu(b_obj, a_obj)

                if not cpu_obj or not gpu_obj:
                    logger.warning(
                        "⚠️ [BOTTLENECK] Missing components — forwarding to Gemini fallback.")
                    try:
                        gem_text = gemini_fallback_with_data(
                            user_input, make_public_data(data))
                        return {"source": "gemini-fallback", "text": gem_text.strip(), "used_fallback": True}
                    except Exception as e:
                        logger.error("Gemini fallback failed: %s", e)
                        return {"source": "local-bottleneck", "text": "Sorry — I couldn't analyze the bottleneck right now. Please try again later."}

                logger.info("✅ [BOTTLENECK] Assigned CPU=%s, GPU=%s",
                            cpu_obj.get("name"), gpu_obj.get("name"))

                # produce final analysis
                text = analyze_bottleneck_text(cpu_obj, gpu_obj)
                return {"source": "local-bottleneck", "text": text}

            except Exception:
                logger.exception(
                    "Bottleneck analysis failed; using Gemini fallback.")
                gem_text = gemini_fallback_with_data(
                    user_input, make_public_data(data))
                return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

        # --- 1) PSU / power-supply natural language detection (improved) ---
        if re.search(r"\b(psu|power supply|power-supply|power supply i need|psu do i need|psu needed|what power supply|how much power|what psu)\b", lower_q):
            try:
                # parse headroom percent if specified (default 30%)
                headroom_percent = 30
                m_head = re.search(
                    r"(\d{1,2})\s*%.*headroom", user_input, flags=re.IGNORECASE)
                if not m_head:
                    m_head = re.search(
                        r"headroom\s*(?:of\s*)?(\d{1,2})\s*%", user_input, flags=re.IGNORECASE)
                if m_head:
                    try:
                        v = int(m_head.group(1))
                        if 5 <= v <= 100:
                            headroom_percent = v
                    except Exception:
                        pass

                psu_resp = recommend_psu_for_query_with_chips(
                    user_input, data, headroom_percent=headroom_percent)

                # If helper failed or has no meaningful output -> forward to Gemini for PSU advice
                if not psu_resp or psu_resp.get("error") or (not psu_resp.get("component_watts") and not psu_resp.get("suggested_psu_chips")):
                    logger.info(
                        "PSU helper returned no clear result — forwarding to Gemini fallback for PSU advice.")
                    fallback_prompt = (
                        "You are ARsemble AI — a concise PC-building assistant.\n\n"
                        f"The user asked: \"{user_input}\"\n\n"
                        "Task: Recommend an appropriate PSU wattage and explain briefly why. "
                        "If the user included CPU and GPU models, mention them in a Detected: line. "
                        "Return a short formatted answer with these sections: Detected (if applicable), Wattage breakdown (CPU / GPU / System), Estimated total draw, Recommended PSU (with headroom), Suggested PSUs (2 items). "
                        "Keep it short (3–6 lines). If you cannot determine exact models, give a safe general recommendation and say 'Check local retailers for exact models/pricing.'"
                    )
                    gem_text = gemini_fallback_with_data(
                        fallback_prompt, make_public_data(data))
                    return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

                # Build user-facing PSU response
                response_lines = ["🔌 PSU Recommendation"]

                # Detected components (if provided)
                detected = psu_resp.get("detected_components") or {}
                det_lines = []
                if detected:
                    for k in ("CPU", "GPU", "Motherboard"):
                        v = detected.get(k)
                        if v:
                            det_lines.append(f"  {k}: {v}")
                else:
                    maybe_cpu = psu_resp.get(
                        "cpu_name") or psu_resp.get("detected_cpu")
                    maybe_gpu = psu_resp.get(
                        "gpu_name") or psu_resp.get("detected_gpu")
                    if maybe_cpu:
                        det_lines.append(f"  CPU: {maybe_cpu}")
                    if maybe_gpu:
                        det_lines.append(f"  GPU: {maybe_gpu}")

                if det_lines:
                    response_lines.append("")
                    response_lines.append("Detected:")
                    response_lines.extend(det_lines)

                # Wattage breakdown
                component_watts = psu_resp.get("component_watts") or {}
                if component_watts:
                    response_lines.append("")
                    for comp, w in component_watts.items():
                        try:
                            w_i = int(w)
                        except Exception:
                            w_i = w
                        response_lines.append(f"  {comp}: {w_i} W")

                total_draw = psu_resp.get("total_draw", 0)
                headroom = psu_resp.get("headroom_percent", headroom_percent)
                recommended_val = psu_resp.get(
                    "recommended_psu") or psu_resp.get("recommended_rounded") or 0
                recommended_name = psu_resp.get("recommended_psu_name") or ""

                response_lines.extend([
                    "",
                    f"Estimated total draw: {int(total_draw)} W",
                    f"Recommended PSU (with {headroom}% headroom): {int(recommended_val)} W" + (
                        f" — {recommended_name}" if recommended_name else "")
                ])

                suggested = psu_resp.get("suggested_psu_chips") or []
                if suggested:
                    response_lines.append("")
                    response_lines.append("Suggested PSUs:")
                    for s in suggested[:4]:
                        price_str = f" — {s.get('price')}" if s.get(
                            "price") else ""
                        response_lines.append(f" • {s.get('text')}{price_str}")

                response_lines.append("")
                response_lines.append(
                    "Tip: Check local retailers for exact pricing and availability.")

                response_text = "\n".join(response_lines)
                return {
                    "source": "local-psu",
                    "text": response_text,
                    "type": "psu_recommendation",
                    "total_draw": total_draw,
                    "recommended_psu": recommended_val,
                    "component_watts": component_watts
                }

            except Exception as e:
                logger.exception(
                    "PSU recommendation failed, using Gemini fallback: %s", e)
                fallback_prompt = (
                    "You are ARsemble AI — a concise PC-building assistant.\n\n"
                    f"The user asked: \"{user_input}\"\n\n"
                    "Task: Provide a short PSU recommendation (3–6 short lines). If possible list detected CPU/GPU, give wattage breakdown, total draw and a recommended PSU wattage with headroom. If you cannot detect components, give a reasonable conservative PSU recommendation and advise to 'check local retailers'."
                )
                gem_text = gemini_fallback_with_data(
                    fallback_prompt, make_public_data(data))
                return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

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

                        norm = {"id": cid, "text": text or (meta.get(
                            "name") or "Unknown"), "type": ctype or "component", "price": price, "meta": meta}
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

                    return {"source": "local-compatibility", "type": "compatibility", "target": target_name, "compatible_type": compat.get("compatible_type"), "results": normalized, "chips": normalized, "text": friendly_text}
                else:
                    # No local compatibility result -> use Gemini fallback (public-safe) with a focused prompt
                    try:
                        logger.info(
                            "No local compatibility result — using Gemini fallback with public data.")

                        # Attempt to extract two component mentions from the user's query
                        comp_a = None
                        comp_b = None
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

                        # fallback: top two token-overlap matches from dataset names (if present)
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
        is_budget_asked = any(k in lower_q for k in ["budget", "build", "recommend", "pc build",
                              "how much", "how much do i need", "i have", "i've got", "i've got a budget, suggest"])

        budget = None
        if currency_match:
            digits = re.sub(r"[^\d]", "", currency_match.group(1))
            if digits:
                budget = int(digits)
        elif is_budget_asked and plain_number_match and not gpu_model_present:
            budget = int(plain_number_match.group(1))

        if budget:
            logger.info(
                f"Budget build requested: ₱{budget} for query: {user_input}")
            if "gaming" in user_input.lower():
                usage_type = "gaming"
            elif "work" in user_input.lower() or "office" in user_input.lower():
                usage_type = "work"
            else:
                usage_type = "general"

            builds = budget_builds(budget, usage=usage_type, top_n=3)
            if builds:
                return {
                    "source": "local-recommendation",
                    "type": "budget_builds",
                    "budget": budget,
                    "usage": usage_type,
                    "results": [build_to_tap_response_with_watts(b, f"build_{i+1}") for i, b in enumerate(builds)]
                }
            else:
                return {
                    "source": "local-recommendation",
                    "type": "budget_builds",
                    "budget": budget,
                    "message": f"No compatible builds found within ₱{budget:,}. Try increasing your budget or check component availability."
                }

        # ---------- 3.5.5) Shops / retailers suggestion (client-first) ----------
        shop_patterns = [
            r"\bshop\b", r"\bstore\b", r"\bretailer\b", r"\bwhere to buy\b",
            r"\bwhere can i buy\b", r"\bpurchase\b", r"\bbuy\b", r"\bcomputer shop\b",
            r"\bpc shop\b", r"\bcomputer store\b", r"\bsuggest shops\b", r"\bsuggest any computer shops\b"
        ]
        is_shop_query = any(re.search(pattern, lower_q)
                            for pattern in shop_patterns)
        is_parts_purchase = any(term in lower_q for term in [
                                "buy parts", "purchase components", "where to get"])

        if is_shop_query or is_parts_purchase:
            try:
                shops_dict = list_shops(only_public=True) or {}
                if shops_dict:
                    lines = ["Here are computer shops I can suggest:\n"]
                    client_key = "smfp_computer"
                    if client_key in shops_dict:
                        client_shop = shops_dict[client_key]
                        lines.append(
                            f"• {client_shop.get('name', 'SMFP Computer')} — CLIENT SHOP")
                        lines.append(f"  {client_shop.get('address', '')}")
                        if client_shop.get('region'):
                            lines.append(
                                f"  Region: {client_shop.get('region')}")
                        lines.append("")
                    other_shops = {k: v for k,
                                   v in shops_dict.items() if k != client_key}
                    for shop_id, shop_info in other_shops.items():
                        lines.append(f"• {shop_info.get('name', 'Unknown')}")
                        lines.append(f"  {shop_info.get('address', '')}")
                        if shop_info.get('region'):
                            lines.append(
                                f"  Region: {shop_info.get('region')}")
                        lines.append("")
                    friendly_text = "\n".join(lines)
                    return {"source": "local-list", "type": "shops_list", "text": friendly_text}
                else:
                    return {"source": "local-list", "type": "shops_list", "text": "I don't have shop information available right now. You can check major retailers like PC Express, DynaQuest, or online platforms like Lazada and Shopee."}
            except Exception as e:
                logger.exception("Shops suggestion failed:")
                gem_text = gemini_fallback_with_data(
                    user_input, make_public_data(data))
                return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

        # ---------- 3.6) Dataset list queries ----------
        list_keywords = ["list", "show me", "available",
                         "all the", "give me the list", "give me a list"]
        is_list_query = any(keyword in lower_q for keyword in list_keywords)

        component_types = {
            "cpu cooler": "cpu_cooler", "cpu_cooler": "cpu_cooler", "cooler": "cpu_cooler",
            "cpu": "cpu", "cpus": "cpu", "processor": "cpu", "processors": "cpu",
            "gpu": "gpu", "gpus": "gpu", "graphics card": "gpu", "graphics cards": "gpu",
            "video card": "gpu", "video cards": "gpu",
            "motherboard": "motherboard", "motherboards": "motherboard", "mobo": "motherboard",
            "ram": "ram", "memory": "ram", "ddr": "ram",
            "storage": "storage", "ssd": "storage", "hdd": "storage", "hard drive": "storage",
            "psu": "psu", "power supply": "psu", "power-supply": "psu"
        }

        brand_patterns = {
            "amd": ("amd", "ryzen"),
            "intel": ("intel", "core", "i3", "i5", "i7", "i9"),
            "nvidia": ("nvidia", "rtx", "gtx", "geforce"),
            "radeon": ("radeon", "rx"),
        }

        if is_list_query:
            matched_type = None
            requested_brands = []
            for key, norm_type in component_types.items():
                if re.search(r'\b' + key + r'\b', lower_q):
                    matched_type = norm_type
                    break

            if not matched_type:
                if any(term in lower_q for term in ["cpu", "processor", "intel", "amd", "ryzen"]):
                    matched_type = "cpu"
                elif any(term in lower_q for term in ["gpu", "graphics", "video", "rtx", "gtx", "radeon"]):
                    matched_type = "gpu"
                elif any(term in lower_q for term in ["motherboard", "mobo"]):
                    matched_type = "motherboard"
                elif any(term in lower_q for term in ["ram", "memory", "ddr"]):
                    matched_type = "ram"
                elif any(term in lower_q for term in ["storage", "ssd", "hdd"]):
                    matched_type = "storage"
                elif any(term in lower_q for term in ["psu", "power supply"]):
                    matched_type = "psu"

            for brand, patterns in brand_patterns.items():
                if any(pattern in lower_q for pattern in patterns):
                    requested_brands.append(brand)

            if matched_type:
                items = list(data.get(matched_type, {}).values()) or []
                if items:
                    if requested_brands and matched_type in ("cpu", "gpu"):
                        def matches_brand(name: str, brand_keys: list) -> bool:
                            n = (name or "").lower()
                            for k in brand_keys:
                                if k in n:
                                    return True
                            return False

                        brand_keys = []
                        for rb in requested_brands:
                            brand_keys.extend(brand_patterns.get(rb, ()))
                        brand_keys = list(set(brand_keys))

                        filtered = [it for it in items if matches_brand(
                            it.get("name", ""), brand_keys)]
                    else:
                        filtered = items

                    if filtered:
                        lines = []
                        limit = min(20, len(filtered))
                        for comp in filtered[:limit]:
                            name = (comp.get("name") or "").strip()
                            price = (comp.get("price") or "").strip()
                            price_str = f" — {price}" if price else ""
                            if matched_type == "cpu":
                                brand = "Intel" if re.search(r"\b(intel|core|i3|i5|i7|i9)\b", name, re.I) else (
                                    "AMD" if re.search(r"\b(ryzen|athlon|threadripper)\b", name, re.I) else "Other")
                                model = re.sub(
                                    r"(?i)\b(intel|amd|amd ryzen|ryzen|core|processor|cpu)\b", "", name).strip()
                                model = model if model else name
                                lines.append(f"{brand} — {model}{price_str}")
                            elif matched_type == "gpu":
                                brand = "NVIDIA" if re.search(r"\b(rtx|gtx|geforce|nvidia)\b", name, re.I) else ("AMD" if re.search(
                                    r"\b(radeon|rx)\b", name, re.I) else "Intel" if re.search(r"\b(arc|iris)\b", name, re.I) else "Other")
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

                        joined_lines = "\n".join(lines)
                        type_label = matched_type.replace('_', ' ').upper()
                        if requested_brands:
                            brand_label = " ".join(
                                [b.upper() for b in requested_brands])
                            friendly_text = f"Available {brand_label} {type_label}s:\n\n{joined_lines}"
                        else:
                            friendly_text = f"Available {type_label}s:\n\n{joined_lines}"

                        if len(filtered) > limit:
                            friendly_text += f"\n\n(Showing {limit} of {len(filtered)} items. Ask for specific models for more details.)"

                        return {"source": "local-list", "type": "component_list_textonly", "target": matched_type, "results": [], "text": friendly_text}

                type_label = matched_type.replace('_', ' ').upper()
                return {"source": "local-list", "type": "component_list_missing", "target": matched_type, "results": [], "text": f"No {type_label} entries found in the current dataset."}

            else:
                categories = [k for k, v in data.items() if v]
                friendly_text = f"I have data available for: {', '.join(categories)}. Ask 'list CPUs', 'list GPUs', 'list AMD processors', etc."
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
    SIMPLE PSU estimator with direct string matching
    """
    try:
        q = query.lower()
        logger.info(f"🔍 PSU Query: '{q}'")

        component_watts = {}
        total = 0

        # DIRECT CPU MATCHING - SIMPLE AND RELIABLE
        if "3200g" in q:
            detected_components.append("CPU: AMD Ryzen 3 3200G")
            cpu_watts = 65
            logger.info("✅ DIRECT CPU MATCH: Ryzen 3 3200G - 65W")
        elif "3600" in q:
            detected_components.append("CPU: AMD Ryzen 5 3600")
            cpu_watts = 65
            logger.info("✅ DIRECT CPU MATCH: Ryzen 5 3600 - 65W")
        elif "5800x" in q:
            detected_components.append("CPU: AMD Ryzen 7 5800X")
            cpu_watts = 105
            logger.info("✅ DIRECT CPU MATCH: Ryzen 7 5800X - 105W")
        elif any(term in q for term in ["cpu", "processor", "ryzen", "amd"]):
            detected_components.append("CPU: Generic CPU")
            cpu_watts = 65
            logger.info("🔧 Generic CPU fallback: 65W")
        else:
            cpu_watts = 0

        component_watts["cpu"] = cpu_watts
        total += cpu_watts

        # DIRECT GPU MATCHING - SIMPLE AND RELIABLE
        if "4060" in q:
            detected_components.append("GPU: RTX 4060")
            gpu_watts = 115
            logger.info("✅ DIRECT GPU MATCH: RTX 4060 - 115W")
        elif "3060" in q:
            detected_components.append("GPU: RTX 3060")
            gpu_watts = 170
            logger.info("✅ DIRECT GPU MATCH: RTX 3060 - 170W")
        elif "4070" in q:
            detected_components.append("GPU: RTX 4070")
            gpu_watts = 200
            logger.info("✅ DIRECT GPU MATCH: RTX 4070 - 200W")
        elif any(term in q for term in ["gpu", "graphics", "rtx", "gtx"]):
            detected_components.append("GPU: Generic Gaming GPU")
            gpu_watts = 200
            logger.info("🔧 Generic GPU fallback: 200W")
        else:
            gpu_watts = 0

        component_watts["gpu"] = gpu_watts
        total += gpu_watts

        # Base system components
        component_watts["motherboard"] = 50
        component_watts["ram"] = 10
        component_watts["storage"] = 5
        component_watts["extras"] = 60
        total += 125  # motherboard + ram + storage + extras

        detected_str = "; ".join(
            detected_components) if detected_components else "Generic PC components"

        # Calculate PSU
        if total > 125:
            target = total * (1 + headroom_percent / 100.0)
            recommended = int((target + 49) // 50 * 50)

            # Minimum PSU requirements
            if gpu_watts > 0:
                recommended = max(recommended, 500)  # Minimum for any GPU
        else:
            recommended = 550
            total = 350

        logger.info(
            f"📈 PSU Calculation: total={total}W, recommended={recommended}W")

        return {
            "detected_str": detected_str,
            "component_watts": component_watts,
            "total_draw": total,
            "recommended_psu": recommended,
            "headroom_percent": headroom_percent,
        }

    except Exception as e:
        logger.exception("❌ PSU function failed:")
        return {"error": str(e)}


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


def suggest_psus_for_wattage_varied(req_watt: int, data: dict, limit: int = 6) -> List[dict]:
    """
    Return appropriate PSUs for the required wattage - filter out obviously underpowered units.
    """
    psus = list(data.get("psu", {}).values())
    annotated = [(p, _parse_psu_wattage(p)) for p in psus]

    # Filter out PSUs that are significantly underpowered
    adequate = sorted([t for t in annotated if t[1] >= req_watt * 0.8],  # Allow 20% below for budget options
                      # Prefer closest match
                      key=lambda x: (abs(x[1] - req_watt), x[1]))

    # If no adequate PSUs, get the highest available
    if not adequate:
        adequate = sorted(annotated, key=lambda x: -x[1])[:limit]

    # Take the best matches
    result = [p for p, w in adequate[:limit]]

    # Remove duplicates by name+wattage
    seen = set()
    unique = []
    for p in result:
        name = (p.get("name") or "").strip()
        w = _parse_psu_wattage(p)
        key = (name.lower(), w)
        if key not in seen and name:
            seen.add(key)
            unique.append(p)

    return unique[:limit]


def recommend_psu_for_query_with_chips(user_input: str, dataset: dict, headroom_percent: int = 30) -> dict:
    """
    Estimate PSU needs from a freeform query mentioning CPU and/or GPU.
    Returns a dict:
      {
        "component_watts": {"CPU": int, "GPU": int, "System": int, ...},
        "total_draw": int,                 # sum of component draws (W)
        "recommended_psu": int,            # wattage after headroom (rounded up)
        "headroom_percent": int,
        "suggested_psu_chips": [ { "text": name, "price": str }, ... ],
        "error": None or "reason"
      }
    Uses dataset['cpu'], dataset['gpu'], dataset['psu'].
    """
    try:
        q = (user_input or "").lower()

        # --- helpers ---
        def parse_named_components(s: str):
            # naive extraction: look for known gpu/cpu tokens in the query
            # returns (cpu_query, gpu_query) as raw strings or None
            cpu_q = None
            gpu_q = None
            # separators
            parts = re.split(
                r"\band\b|\bwith\b|\bplus\b|\b/\b|\bvs\b|\bvs.\b|\b,\b", s)
            # try to find tokens that look like GPUs or CPUs
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if re.search(r"\b(rt[x|x ]|gtx|geforce|radeon|rx|vga|titan)\b", p):
                    gpu_q = p
                elif re.search(r"\b(ryzen|core|i\d|xeon|athlon)\b", p):
                    cpu_q = p
            # fallback: if query explicitly says "PSU for <gpu>" or starts with GPU first
            m = re.search(r"psu\s+for\s+(.+)", s)
            if m and not gpu_q:
                gpu_q = m.group(1).strip()
            return cpu_q, gpu_q

        def parse_tdp_from_obj(obj):
            # try to pull tdp/power numeric from object fields in dataset
            if not obj:
                return None
            # common fields: 'tdp', 'power'
            t = obj.get("tdp") or obj.get("power") or ""
            if isinstance(t, str):
                m = re.search(r"(\d{2,3})\s*w", t, flags=re.IGNORECASE)
                if m:
                    return int(m.group(1))
                m = re.search(r"(\d{2,3})", t)
                if m:
                    return int(m.group(1))
            if isinstance(t, (int, float)):
                return int(t)
            return None

        def approximate_gpu_draw_by_name(name: str) -> int:
            # reasonable defaults (approximate)
            name = (name or "").lower()
            if "rtx 4090" in name or "4090" in name and "rtx" in name:
                return 450
            if "rtx 4080" in name or "4080" in name:
                return 320
            if "rtx 4070" in name or "4070" in name:
                return 200
            if "rtx 4060" in name or "4060" in name:
                return 115
            if "rtx 3050" in name or "3050" in name:
                return 130
            if "rtx 3060" in name or "3060" in name:
                return 170
            if "gtx 750" in name:
                return 60
            if "rx 6700" in name or "6700" in name:
                return 170
            # generic fallback for other modern GPUs
            if "rtx" in name or "radeon" in name or "rx" in name or "gtx" in name:
                return 200
            # unknown -> return None
            return None

        # --- extraction & matching ---
        cpu_q, gpu_q = parse_named_components(q)

        # Try to best-match against dataset using your helper _best_match_in_dataset if available
        cpu_obj = None
        gpu_obj = None
        try:
            if cpu_q:
                cpu_obj, _ = _best_match_in_dataset(
                    cpu_q, {"cpu": dataset.get("cpu", {})})
        except Exception:
            cpu_obj = None
        try:
            if gpu_q:
                gpu_obj, _ = _best_match_in_dataset(
                    gpu_q, {"gpu": dataset.get("gpu", {})})
        except Exception:
            gpu_obj = None

        # If no match found but raw token exists, create a minimal guessed object (name field)
        if not cpu_obj and cpu_q:
            cpu_obj = {"name": cpu_q}
        if not gpu_obj and gpu_q:
            gpu_obj = {"name": gpu_q}

        # --- determine wattage ---
        cpu_draw = parse_tdp_from_obj(cpu_obj) if cpu_obj else None
        gpu_draw = None
        if gpu_obj:
            gpu_draw = parse_tdp_from_obj(gpu_obj)
            # try gpu 'power' numeric if present (e.g., "~200 Watts")
            if not gpu_draw:
                # try to parse numbers in 'power' field string
                power_field = (gpu_obj.get("power") or "")
                m = re.search(r"(\d{2,3})", str(power_field))
                if m:
                    gpu_draw = int(m.group(1))
        # fallback approximations
        if gpu_draw is None and gpu_obj:
            gpu_draw = approximate_gpu_draw_by_name(gpu_obj.get("name", ""))

        # If CPU draw missing, try defaults: low-end 65W, mid/high 95-125W depending on name
        if cpu_draw is None and cpu_obj:
            name = (cpu_obj.get("name") or "").lower()
            if "ryzen 9" in name or "i9" in name or "14900" in name or "14700" in name:
                cpu_draw = 125
            elif "ryzen 7" in name or "i7" in name or re.search(r"\b(5800x|7700x|7900x)\b", name):
                cpu_draw = 105
            elif "i5" in name or "ryzen 5" in name:
                cpu_draw = 65 if "5600" in name or "7600" in name else 95 if "12400" in name else 65
            else:
                cpu_draw = 65

        # If both missing, return helpful error
        if not cpu_obj and not gpu_obj:
            return {"error": "no_components", "text": "No CPU or GPU detected in query. Please mention CPU and/or GPU (e.g., 'PSU for RTX 4070 and Ryzen 7 5800X')."}

        # System baseline overhead (motherboard, storage, fans, etc.)
        # Use slightly larger baseline for high-end GPUs
        if gpu_draw and gpu_draw >= 300:
            system_draw = 150
        else:
            system_draw = 125

        # Fill unknown draws conservatively
        cpu_draw = int(cpu_draw or 65)
        gpu_draw = int(gpu_draw or 200)

        # compute totals
        component_watts = {}
        if cpu_obj:
            component_watts["CPU"] = cpu_draw
        if gpu_obj:
            component_watts["GPU"] = gpu_draw
        component_watts["System"] = system_draw

        total_draw = sum(component_watts.values())  # base draw
        recommended = int(
            math.ceil(total_draw * (1 + headroom_percent / 100.0)))

        # Round recommended to nearest 50 for nicer numbers
        recommended_rounded = int(math.ceil(recommended / 50.0) * 50)

        # pick candidate PSUs from dataset (smallest that >= recommended_rounded)
        psu_list = []
        for k, p in (dataset.get("psu") or {}).items():
            try:
                watt = int(re.sub(r"[^\d]", "", str(
                    p.get("wattage", p.get("wattage", "")) or "")))
            except Exception:
                watt = None
            if watt:
                psu_list.append({"id": k, "name": p.get(
                    "name"), "watt": watt, "price": p.get("price", "")})
        psu_list_sorted = sorted(psu_list, key=lambda x: x["watt"])

        recommended_psu = None
        suggested = []
        for ps in psu_list_sorted:
            if ps["watt"] >= recommended_rounded:
                recommended_psu = ps
                break
        # if none large enough, pick the largest available
        if not recommended_psu and psu_list_sorted:
            recommended_psu = psu_list_sorted[-1]

        # Gather a few suggestions (up to 3) >= recommended_rounded
        for ps in psu_list_sorted:
            if ps["watt"] >= recommended_rounded:
                suggested.append(
                    {"text": f"{ps['name']} ({ps['watt']}W)", "price": ps.get("price", "")})
        # if none, include top 3 highest available
        if not suggested:
            suggested = [{"text": f"{ps['name']} ({ps['watt']}W)", "price": ps.get(
                "price", "")} for ps in psu_list_sorted[-3:]]

        return {
            "component_watts": component_watts,
            "total_draw": total_draw,
            "recommended_psu": recommended_psu["watt"] if recommended_psu else recommended_rounded,
            "recommended_psu_name": recommended_psu["name"] if recommended_psu else None,
            "recommended_psu_id": recommended_psu["id"] if recommended_psu else None,
            "recommended_rounded": recommended_rounded,
            "headroom_percent": headroom_percent,
            "suggested_psu_chips": suggested,
            "error": None
        }

    except Exception as exc:
        logger.exception("recommend_psu_for_query_with_chips failed: %s", exc)
        return {"error": "internal", "text": "PSU estimation failed."}


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
    """Accurate wattage estimation with guaranteed values."""
    if not comp or not isinstance(comp, dict):
        return 0

    # Direct field checks first
    for key in ("power", "tdp", "wattage", "tdp_w", "power_draw", "w"):
        if key in comp and comp.get(key) is not None:
            v = _parse_watt_value(comp.get(key))
            if v:
                return v

    name = (comp.get("name") or "").lower()

    # GUARANTEED GPU WATTAGES
    if "4070" in name and "rtx" in name:
        return 200
    if "4080" in name and "rtx" in name:
        return 320
    if "3060" in name and "rtx" in name:
        return 170
    if "6700" in name and "rx" in name:
        return 220
    if "4060" in name and "rtx" in name:
        return 115
    if "3050" in name and "rtx" in name:
        return 130
    if any(x in name for x in ['rtx', 'gtx', 'radeon', 'rx']):
        return 200  # default GPU

    # GUARANTEED CPU WATTAGES
    if "5800x" in name and "ryzen" in name:
        return 105
    if "5700x" in name and "ryzen" in name:
        return 65
    if "3600" in name and "ryzen" in name:
        return 65
    if "13400" in name and "core" in name:
        return 65
    if "3200g" in name and "ryzen" in name:
        return 65
    if any(x in name for x in ['ryzen', 'core i', 'i3', 'i5', 'i7', 'i9']):
        return 65

    # Other components
    if any(x in name for x in ['motherboard', 'mobo', 'b760', 'b550']):
        return 50
    if any(x in name for x in ['ram', 'memory', 'ddr']):
        return 10
    if any(x in name for x in ['ssd', 'hdd', 'storage', 'nvme']):
        return 5

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

def improve_component_identification(found_a, found_b):
    """
    Robust CPU/GPU identification.
    - First uses strong keywords and 'type' field.
    - Falls back to a scored heuristic only when needed.
    - Returns a tuple (cpu_obj, gpu_obj) or (None, None) if low confidence.
    """
    def looks_like_cpu(comp):
        if not comp:
            return False
        name = (comp.get("name") or "").lower()
        ctype = (comp.get("type") or "").lower()
        # high confidence from explicit type field
        if "cpu" in ctype:
            return True
        # keyword matches
        return bool(re.search(r"\b(ryzen|intel|core\s*i|i[0-9]{1,2}\b|xeon|athlon|processor)\b", name))

    def looks_like_gpu(comp):
        if not comp:
            return False
        name = (comp.get("name") or "").lower()
        ctype = (comp.get("type") or "").lower()
        if "gpu" in ctype or "graphics" in ctype:
            return True
        return bool(re.search(r"\b(rtx|gtx|geforce|radeon|rx|vga|titan|nvidia|amd)\b", name))

    # 1) explicit keyword classification
    if looks_like_cpu(found_a) and looks_like_gpu(found_b):
        cpu_obj, gpu_obj = found_a, found_b
    elif looks_like_cpu(found_b) and looks_like_gpu(found_a):
        cpu_obj, gpu_obj = found_b, found_a
    else:
        # 2) fallback scoring heuristic (similar to original but used only here)
        def component_score(comp, desired_type):
            if not comp:
                return 0
            name = (comp.get("name") or "").lower()
            comp_type = (comp.get("type") or "").lower()
            score = 0
            if desired_type == "cpu":
                if "cpu" in comp_type:
                    score += 10
                if re.search(r"\b(ryzen|core\s*i|i[0-9]{1,2}\b|xeon|athlon)\b", name):
                    score += 5
                if "processor" in name:
                    score += 3
            else:  # gpu
                if "gpu" in comp_type or "graphics" in comp_type:
                    score += 10
                if re.search(r"\b(rtx|gtx|geforce|radeon|rx|nvidia|amd)\b", name):
                    score += 5
                if any(x in name for x in ("graphics", "video card")):
                    score += 3
            return score

        a_cpu_score = component_score(found_a, "cpu")
        a_gpu_score = component_score(found_a, "gpu")
        b_cpu_score = component_score(found_b, "cpu")
        b_gpu_score = component_score(found_b, "gpu")

        # prefer assignment that maximizes (cpu-likeness of cpu + gpu-likeness of gpu)
        if a_cpu_score + b_gpu_score >= a_gpu_score + b_cpu_score:
            cpu_obj, gpu_obj = found_a, found_b
        else:
            cpu_obj, gpu_obj = found_b, found_a

    # Confidence checks: require at least small positive signals
    cpu_conf = 0 if cpu_obj is None else (
        (2 if "cpu" in ((cpu_obj.get("type") or "").lower()) else 0)
        + (2 if re.search(r"\b(ryzen|core\s*i|i[0-9]{1,2}\b|xeon)\b",
           (cpu_obj.get("name") or "").lower()) else 0)
    )
    gpu_conf = 0 if gpu_obj is None else (
        (2 if "gpu" in ((gpu_obj.get("type") or "").lower()) else 0)
        + (2 if re.search(r"\b(rtx|gtx|geforce|radeon|rx|nvidia|amd)\b",
           (gpu_obj.get("name") or "").lower()) else 0)
    )

    logger.debug("🔍 [IDENT] cpu_conf=%s gpu_conf=%s cpu=%s gpu=%s", cpu_conf, gpu_conf,
                 (cpu_obj and cpu_obj.get("name")), (gpu_obj and gpu_obj.get("name")))

    # If both confidences are very low, return failure so caller can fallback
    if cpu_conf < 1 or gpu_conf < 1:
        logger.warning(
            "⚠️ [IDENT] Low confidence assignment: cpu_conf=%s gpu_conf=%s", cpu_conf, gpu_conf)
        return None, None

    return cpu_obj, gpu_obj


def estimate_gpu_capacity(gpu: dict) -> float:
    """
    Improved GPU capacity estimation with better model detection.
    """
    if not gpu or not isinstance(gpu, dict):
        return 0.0

    name = (gpu.get("name") or "").lower()

    # VRAM extraction
    vram_field = gpu.get("vram") or gpu.get("memory") or ""
    vram_gb = _safe_int_from_field(vram_field, default=0)

    # If no VRAM found, try to estimate from model name
    if vram_gb == 0:
        if '4090' in name:
            vram_gb = 24
        elif '4080' in name:
            vram_gb = 16
        elif '4070' in name:
            vram_gb = 12
        elif '4060' in name:
            vram_gb = 8
        elif '3090' in name:
            vram_gb = 24
        elif '3080' in name:
            vram_gb = 10
        elif '3070' in name:
            vram_gb = 8
        elif '3060' in name:
            vram_gb = 12
        elif '3050' in name:
            vram_gb = 8
        elif '1650' in name:
            vram_gb = 4
        elif '1660' in name:
            vram_gb = 6
        elif 'rx' in name and '7900' in name:
            vram_gb = 20
        elif 'rx' in name and '7800' in name:
            vram_gb = 16
        elif 'rx' in name and '7700' in name:
            vram_gb = 12
        elif 'rx' in name and '7600' in name:
            vram_gb = 8
        else:
            vram_gb = 6  # conservative default

    # GPU tier scoring with more precise model detection
    tier = 1.0
    if any(tok in name for tok in ["rtx 4090", "rx 7900 xtx"]):
        tier = 3.0
    elif any(tok in name for tok in ["rtx 4080", "rx 7900 xt", "rx 7800 xt"]):
        tier = 2.7
    elif any(tok in name for tok in ["rtx 4070", "rx 7700 xt"]):
        tier = 2.3
    elif any(tok in name for tok in ["rtx 4060", "rx 7600"]):
        tier = 1.8
    elif any(tok in name for tok in ["rtx 3060", "rtx 3070", "rx 6700"]):
        tier = 1.5
    elif any(tok in name for tok in ["rtx 3050", "gtx 1660", "rx 6600"]):
        tier = 1.2
    elif any(tok in name for tok in ["gtx 1650", "rx 6500"]):
        tier = 0.8
    elif any(tok in name for tok in ["gtx 1050", "rx 6400"]):
        tier = 0.6

    # Power hint
    power = _safe_int_from_field(
        gpu.get("power") or gpu.get("tdp") or gpu.get("wattage") or 0)

    # Compute capacity: combine tier, vram and power
    capacity = (tier * 120.0) + (vram_gb * 10.0) + (min(power, 450) * 0.3)

    # Ensure positive and reasonable
    return float(max(20.0, capacity))


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


def lookup_component_by_chip_id(chip_id: str) -> dict:
    """Lookup component by chip ID (type:slug format)."""
    try:
        if not chip_id or ':' not in chip_id:
            return None
        comp_type, slug = chip_id.split(':', 1)
        if comp_type in data:
            for comp_id, comp_data in data[comp_type].items():
                if slugify(comp_data.get('name', '')) == slug:
                    return comp_data
    except Exception:
        pass
    return None


def analyze_bottleneck_for_build(cpu: dict, gpu: dict, resolution: str = "1080p", settings: str = "high", target_fps: int = 60) -> str:
    """Bottleneck analysis between CPU and GPU."""
    return analyze_bottleneck_text(cpu, gpu, resolution, settings, target_fps)


def build_from_chip_ids(chip_ids: list) -> dict:
    """Create a build dict from chip IDs."""
    build = {}
    for chip_id in chip_ids:
        comp = lookup_component_by_chip_id(chip_id)
        if comp:
            comp_type = None
            for ct in ['cpu', 'gpu', 'motherboard', 'ram', 'storage', 'psu']:
                if comp in data.get(ct, {}).values():
                    comp_type = ct
                    break
            if comp_type:
                build[comp_type] = comp
    return build


def parse_build_from_text(text: str, data: dict, verbose: bool = False) -> dict:
    """Parse component names from free text to build dict."""
    build = {}
    text_lower = text.lower()

    # Try to match components using your existing best_match function
    for comp_type in ['cpu', 'gpu', 'motherboard', 'ram', 'storage']:
        # Use the query itself to find best match for each type
        best_obj, matched_type = _best_match_in_dataset(text, data)
        if best_obj and matched_type == comp_type:
            build[comp_type] = best_obj

    return build


def analyze_bottleneck_text(cpu: dict, gpu: dict, resolution: str = "1080p", settings: str = "high", target_fps: int = 60) -> str:
    """
    Clean, plain-text bottleneck analysis with readable formatting.
    """
    # Validate inputs
    if not cpu or not gpu:
        return "Error: Could not identify both CPU and GPU components."

    cpu_name = cpu.get("name", "Unknown CPU").strip()
    gpu_name = gpu.get("name", "Unknown GPU").strip()

    # Debug logging to see what components are being analyzed
    logger.info(f"Bottleneck analysis: CPU='{cpu_name}', GPU='{gpu_name}'")

    # --- Estimate performance values ---
    cpu_cap = estimate_cpu_capacity(cpu)
    gpu_cap = estimate_gpu_capacity(gpu)

    # If capacities are unrealistic, there might be identification issues
    if cpu_cap <= 0 or gpu_cap <= 0:
        return f"Error: Could not analyze performance characteristics. CPU: {cpu_name}, GPU: {gpu_name}"

    demand = estimate_workload_demand(
        resolution=resolution, settings=settings, target_fps=target_fps)

    cpu_load_pct = int(
        round((demand["cpu"] / cpu_cap) * 100)) if cpu_cap > 0 else 0
    gpu_load_pct = int(
        round((demand["gpu"] / gpu_cap) * 100)) if gpu_cap > 0 else 0

    # Cap load percentages at reasonable levels
    cpu_load_pct = min(cpu_load_pct, 200)
    gpu_load_pct = min(gpu_load_pct, 200)

    # --- Decide verdict ---
    diff = gpu_load_pct - cpu_load_pct
    abs_diff = abs(diff)

    if cpu_load_pct >= 110 and cpu_load_pct - gpu_load_pct >= 15:
        verdict = "⚠️ CPU Bottleneck — Processor is overloaded."
        summary = "The CPU limits overall performance. Consider upgrading your CPU or lowering CPU-heavy settings/FPS target."
    elif gpu_load_pct >= 110 and gpu_load_pct - cpu_load_pct >= 15:
        verdict = "⚠️ GPU Bottleneck — Graphics card is overloaded."
        summary = "The GPU limits performance. Consider lowering resolution or upgrading your GPU."
    elif abs_diff <= 15:
        verdict = "✅ Balanced — both components perform similarly."
        summary = "Your CPU and GPU are well-matched for this workload."
    elif diff > 15:
        verdict = "⚠️ GPU Bottleneck — GPU more utilized."
        summary = "The GPU is the limiting factor. Lower graphical settings or upgrade your GPU."
    else:
        verdict = "⚠️ CPU Bottleneck — CPU more utilized."
        summary = "The CPU is the limiting factor. Lower CPU-heavy settings or upgrade your CPU."

    # --- Build clean formatted output ---
    settings_label = settings.capitalize()

    result = f"""
Bottleneck Analysis

CPU: {cpu_name}
GPU: {gpu_name}
Resolution: {resolution} ({settings_label} Settings, {target_fps} FPS Target)

→ CPU Load: {cpu_load_pct}%
→ GPU Load: {gpu_load_pct}%

Verdict: {verdict}
Summary: {summary}
""".strip()

    return result


def improve_component_identification(found_a, found_b):
    """Improve CPU/GPU identification in bottleneck analysis"""
    def component_score(comp, desired_type):
        """Score how likely a component is to be the desired type"""
        if not comp:
            return 0

        name = (comp.get('name') or '').lower()
        comp_type = (comp.get('type') or '').lower()

        score = 0
        if desired_type == 'cpu':
            if 'cpu' in comp_type:
                score += 10
            if any(term in name for term in ['ryzen', 'core i', 'i3', 'i5', 'i7', 'i9', 'xeon', 'athlon']):
                score += 5
            if 'processor' in name:
                score += 3

        elif desired_type == 'gpu':
            if 'gpu' in comp_type:
                score += 10
            if any(term in name for term in ['rtx', 'gtx', 'radeon', 'rx', 'geforce', 'nvidia']):
                score += 5
            if any(term in name for term in ['graphics', 'video card']):
                score += 3

        return score

    # Score both components for CPU and GPU
    a_cpu_score = component_score(found_a, 'cpu')
    a_gpu_score = component_score(found_a, 'gpu')
    b_cpu_score = component_score(found_b, 'cpu')
    b_gpu_score = component_score(found_b, 'gpu')

    print(
        f"🔍 [DEBUG] Component A: {found_a.get('name')} - CPU score: {a_cpu_score}, GPU score: {a_gpu_score}")
    print(
        f"🔍 [DEBUG] Component B: {found_b.get('name')} - CPU score: {b_cpu_score}, GPU score: {b_gpu_score}")

    # Determine best assignment
    if a_cpu_score + b_gpu_score > a_gpu_score + b_cpu_score:
        cpu_obj, gpu_obj = found_a, found_b
    else:
        cpu_obj, gpu_obj = found_b, found_a

    print(f"🔍 [DEBUG] Identified CPU: {cpu_obj.get('name')}")
    print(f"🔍 [DEBUG] Identified GPU: {gpu_obj.get('name')}")

    # If scores are very low, might be wrong identification
    if component_score(cpu_obj, 'cpu') < 3 or component_score(gpu_obj, 'gpu') < 3:
        print("⚠️ [DEBUG] Low confidence in identification")
        return None, None

    return cpu_obj, gpu_obj


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
