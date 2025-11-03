from typing import Any, Dict, Tuple, List, Optional
from typing import List, Dict, Optional
from typing import Any, Dict, Optional
import time
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


# --------------------------
# Setup
# --------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Fallback")

# -------------------------- GEMINI SETUP --------------------------

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "YOUR_API_KEY_HERE")

# Register Gemini for global access
globals()['genai'] = genai
globals()['GEMINI_MODEL'] = globals().get(
    'GEMINI_MODEL') or "gemini-2.5-flash-lite"

logger.info("✅ Gemini client initialized | GEMINI_MODEL=%s",
            globals().get("GEMINI_MODEL"))


# --- BEGIN: stream_response wrapper (append to arsemble_ai.py) ---
# This wrapper yields newline-delimited JSON chunks quickly by slicing the
# blocking get_ai_response(prompt) text result. Keeps existing behavior intact.


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


def budget_builds(budget: Optional[int] = None, usage: str = "general", top_n: int = 3, user_request: Optional[str] = None) -> List[Dict]:
    """
    Generate budget-appropriate builds with intelligent component selection.

    This function accepts either an explicit numeric `budget` or a free-text
    `user_request` such as "build me a gaming pc". When `user_request` is
    provided and `budget` is None, the function will auto-select reasonable
    defaults for `usage` and `budget`.
    """

    # Interpret natural language requests (e.g., "build me a gaming pc")
    if user_request:
        req_lower = user_request.lower()

        if "gaming" in req_lower:
            usage = "gaming"
            budget = 60000
        elif "office" in req_lower:
            usage = "office"
            budget = 30000
        elif "balanced" in req_lower or "general" in req_lower:
            usage = "balanced"
            budget = 40000
        elif "editing" in req_lower or "creator" in req_lower:
            usage = "content_creation"
            budget = 80000
        else:
            usage = "general"
            budget = 25000

        try:
            logger.info(
                f"Auto-selected usage '{usage}' and budget ₱{budget} from request: {user_request}")
        except Exception:
            pass

    # Fallback if neither budget nor user_request provided
    if budget is None:
        budget = 35000

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
    try:
        logger.info(
            f"Budget {budget}: Selected {len(cpus)} CPUs, {len(gpus)} GPUs, {len(mobos)} Mobos")
    except Exception:
        pass

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
        try:
            logger.info(
                f"Generated {len(builds_sorted)} builds for budget {budget}")
            for i, build in enumerate(builds_sorted[:3]):
                logger.info(
                    f"Build {i+1}: ₱{build['total_price']} (score: {build['score']})")
        except Exception:
            pass
    else:
        try:
            logger.warning(f"No builds generated for budget {budget}")
        except Exception:
            pass

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


def build_chips(items, comp_type):
    chips = []
    for i in items:
        chips.append({
            "label": f"{i.get('name', 'Unknown')} — ₱{i.get('price', 'N/A')}",
            "type": comp_type,
            "id": i.get("id", None)
        })
    return chips


# ----------------- compatibility helper utilities -----------------


# --- small helpers used by get_compatible_components ---

# ==========================================
# =========== HELPER FUNCTIONS =============
# ==========================================


def normalize_key(s: str) -> str:
    """Canonicalize textual keys/names to snake_case-ish form."""
    if s is None:
        return s
    s = s.strip().lower()
    s = re.sub(r'[\s\-]+', '_', s)  # spaces/dashes -> underscore
    s = re.sub(r'[^a-z0-9_]', '', s)  # drop other punctuation
    return s


def _to_set(value):
    """Convert compatibility values to a set for robust comparison."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return set(v for v in value if v is not None)
    if isinstance(value, (int, float)):
        return {value}
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(',') if p.strip()]
        return set(parts) if parts else {value.strip().lower()}
    return {value}


def is_compatible(item_a: Dict[str, Any], item_b: Dict[str, Any], rules: Any = None) -> bool:
    """Generic compatibility check between two items."""
    if item_a is None or item_b is None:
        return False

    a = {k: v for k, v in item_a.items()}
    b = {k: v for k, v in item_b.items()}

    def get_val(d, key):
        if key in d:
            return d.get(key)
        nk = normalize_key(key)
        if nk in d:
            return d.get(nk)
        for k in d.keys():
            if normalize_key(k) == nk:
                return d.get(k)
        return None

    def check_pair(a_key, b_key):
        a_val = get_val(a, a_key)
        b_val = get_val(b, b_key)
        if a_val is None or b_val is None:
            return False
        a_set = _to_set(a_val)
        b_set = _to_set(b_val)

        try:
            if len(a_set) == 1 and len(b_set) == 1:
                a_token = next(iter(a_set))
                b_token = next(iter(b_set))
                a_num = float(a_token) if str(a_token).replace(
                    '.', '', 1).isdigit() else None
                b_num = float(b_token) if str(b_token).replace(
                    '.', '', 1).isdigit() else None
                if a_num is not None and b_num is not None:
                    return a_num >= b_num
        except Exception:
            pass

        return bool(a_set & b_set)

    # rules
    if rules:
        if isinstance(rules, dict):
            for a_key, b_key in rules.items():
                if check_pair(a_key, b_key):
                    return True
        elif isinstance(rules, (list, tuple)):
            for r in rules:
                if isinstance(r, dict):
                    ak = r.get('a_key') or r.get('a') or r.get('left')
                    bk = r.get('b_key') or r.get('b') or r.get('right')
                    if ak and bk and check_pair(ak, bk):
                        return True
        return False

    # fallback heuristics
    heuristics = [
        ('socket', 'socket'),
        ('ram_type', 'ram_type'),
        ('interface', 'interface'),
        ('tdp', 'wattage'),
        ('wattage', 'tdp'),
    ]
    for ak, bk in heuristics:
        if check_pair(ak, bk):
            return True
    return False


# ==========================================
# =========== MAIN COMPATIBILITY ===========
# ==========================================
# assume logger, _best_match_in_dataset, improve_component_identification,
# analyze_bottleneck_text, _parse_price_to_int, _parse_psu_wattage, slugify exist elsewhere


def _parse_wattage_to_int(val):
    """
    Extract integer wattage from a string: "200W", "200 W", "approx 250", or plain "200".
    Returns 0 if nothing found.
    """
    try:
        if val is None:
            return 0
        s = str(val)
        m = re.search(r"(\d{2,4})", s)
        if m:
            return int(m.group(1))
        try:
            return int(float(s))
        except Exception:
            digits = re.sub(r"[^\d]", "", s)
            return int(digits) if digits else 0
    except Exception:
        return 0


def _format_price(price):
    """Normalize price value (int/str) to '₱X,XXX' or return original if empty."""
    try:
        if price is None or price == "":
            return ""
        # if price string with dash range like "4000-5500" or "4000–5500"
        if isinstance(price, str) and ("-" in price or "–" in price):
            parts = re.split(r"[-–]", price)
            parts_int = []
            for p in parts:
                digits = re.sub(r"[^\d]", "", p)
                if digits:
                    parts_int.append(int(digits))
            if len(parts_int) == 2:
                return f"₱{parts_int[0]:,}–₱{parts_int[1]:,}"
        digits = re.sub(r"[^\d]", "", str(price) or "")
        if digits:
            val = int(digits)
            return f"₱{val:,}"
    except Exception:
        pass
    return str(price or "")


# ======================
#   Main Compatibility
# ======================
def get_compatible_components(query: str, data: dict) -> dict:
    """
    Clean, efficient compatibility checker with proper logic flow.
    """
    try:
        q_raw = (query or "").strip()
        q_lower = q_raw.lower()

        # Helper: build chip dict for frontend
        def _chip_obj(item, comp_type):
            return {
                "id": f"{comp_type}:{slugify(item.get('name', ''))}",
                "text": item.get("name") or "",
                "price": item.get("price") or "",
                "type": comp_type,
                "meta": item,
            }

        # Helper: prefer items available in CLIENT_SHOP_ID
        def _prefer_shop(items):
            try:
                preferred = [it for it in items if CLIENT_SHOP_ID in (
                    it.get("stores") or [])]
                return preferred if preferred else items
            except Exception:
                return items

        # --- PATTERN 1: BOTTLENECK DETECTION ---
        bottleneck_patterns = [
            r"(.+?)\s+vs\.?\s+(.+)",
            r"(.+?)\s+vs\s+(.+)",
            r"(.+?)\s+and\s+(.+?)\s+bottleneck",
        ]

        for pat in bottleneck_patterns:
            m = re.search(pat, q_lower, re.IGNORECASE)
            if m:
                a, b = m.group(1).strip(), m.group(2).strip()
                cpu_obj, gpu_obj = None, None

                # Try to identify CPU and GPU
                a_obj, a_type = _best_match_in_dataset(a, data)
                b_obj, b_type = _best_match_in_dataset(b, data)

                # Assign CPU/GPU
                if a_type == "cpu" and b_type == "gpu":
                    cpu_obj, gpu_obj = a_obj, b_obj
                elif b_type == "cpu" and a_type == "gpu":
                    cpu_obj, gpu_obj = b_obj, a_obj
                else:
                    # Heuristic fallback
                    a_name = (a_obj.get("name") or "").lower() if a_obj else ""
                    b_name = (b_obj.get("name") or "").lower() if b_obj else ""

                    if re.search(r"\b(ryzen|intel|i5|i7|i9)\b", a_name) and re.search(r"\b(rtx|gtx|rx)\b", b_name):
                        cpu_obj, gpu_obj = a_obj, b_obj
                    elif re.search(r"\b(ryzen|intel|i5|i7|i9)\b", b_name) and re.search(r"\b(rtx|gtx|rx)\b", a_name):
                        cpu_obj, gpu_obj = b_obj, a_obj

                if cpu_obj and gpu_obj:
                    cpu_cap = estimate_cpu_capacity(cpu_obj) or 1.0
                    gpu_cap = estimate_gpu_capacity(gpu_obj) or 1.0
                    ratio = gpu_cap / max(cpu_cap, 0.1)

                    severity = "low"
                    if ratio < 0.45:
                        severity = "severe"
                    elif ratio < 0.75:
                        severity = "high"
                    elif ratio < 0.95:
                        severity = "moderate"

                    chips = [_chip_obj(cpu_obj, "cpu"),
                             _chip_obj(gpu_obj, "gpu")]
                    reason = f"Bottleneck analysis: GPU ≈ {ratio:.2f}x CPU performance ({severity})"

                    text_lines = [
                        "Compatibility Check",
                        f"Component Pair: {cpu_obj.get('name')} + {gpu_obj.get('name')}",
                        f"Severity: {severity.upper()}",
                        reason,
                        "Matched components:"
                    ]
                    for c in chips:
                        text_lines.append(
                            f"• {c['text']}{' — ' + _format_price(c.get('price')) if c.get('price') else ''}")

                    return {
                        "source": "local-bottleneck",
                        "found": True,
                        "verdict": "compatible",
                        "reason": reason,
                        "text": "\n".join(text_lines),
                        "target": f"{cpu_obj.get('name')} + {gpu_obj.get('name')}",
                        "target_type": "cpu_gpu_pair",
                        "compatible_type": "bottleneck_analysis",
                        "results": chips,
                        "chips": chips,
                        "bottleneck_ratio": float(f"{ratio:.2f}"),
                    }

        # --- PATTERN 2: COMPATIBILITY QUERIES ---
        compatibility_patterns = [
            # "What CPU is compatible with [motherboard]"
            (r"what\s+(cpu|processor)s?\s+(?:are|is)\s+compatible\s+with\s+(.+)", "cpu_for_mobo"),
            # "What works with [component]"
            (r"what\s+(?:works with|is compatible with)\s+(.+)", "what_works"),
            # "[A] works with [B]"
            (r"(.+?)\s+works with\s+(.+)", "works_with"),
            # "[A] compatible with [B]"
            (r"(.+?)\s+compatible with\s+(.+)", "compatible_with"),
            # "What [component] for [base]"
            (r"what\s+(.+?)\s+(?:for|with)\s+(.+)", "what_for"),
        ]

        for pattern, pattern_type in compatibility_patterns:
            match = re.search(pattern, q_lower, re.IGNORECASE)
            if match:
                groups = match.groups()

                if pattern_type == "cpu_for_mobo":
                    # "What CPU is compatible with [motherboard]"
                    mobo_name = groups[1].strip()
                    return _handle_motherboard_compatibility(mobo_name, "cpu", data, _chip_obj, _prefer_shop)

                elif pattern_type == "what_works":
                    # "What works with [component]"
                    component_name = groups[0].strip()
                    return _handle_what_works_with(component_name, data, _chip_obj, _prefer_shop)

                elif pattern_type in ["works_with", "compatible_with"]:
                    # "[A] works/compatible with [B]"
                    item_a, item_b = groups[0].strip(), groups[1].strip()
                    return _handle_pair_compatibility(item_a, item_b, data, _chip_obj, _prefer_shop)

                elif pattern_type == "what_for":
                    # "What [component] for [base]"
                    target_type, base_component = groups[0].strip(
                    ), groups[1].strip()
                    return _handle_what_for(target_type, base_component, data, _chip_obj, _prefer_shop)

        # --- PATTERN 3: SINGLE COMPONENT LOOKUP ---
        # If no patterns matched, try single component detection
        return _handle_single_component(q_raw, data, _chip_obj, _prefer_shop)

    except Exception as e:
        logger.exception("get_compatible_components failed:")
        return {"found": False, "error": str(e), "message": "Compatibility check failed."}


# --- HELPER FUNCTIONS ---

def _handle_motherboard_compatibility(mobo_name: str, target_type: str, data: dict, chip_func, shop_func) -> dict:
    """Handle compatibility queries for motherboards"""
    mobo_obj, mobo_type = _best_match_in_dataset(mobo_name, data)

    if not mobo_obj or mobo_type != "motherboard":
        return {"found": False, "message": f"Motherboard '{mobo_name}' not found."}

    mobo_socket = (mobo_obj.get("socket") or "").lower()
    mobo_ram_type = (mobo_obj.get("ram_type") or "").lower()

    compatible_items = []
    chips = []

    if target_type == "cpu":
        # Find CPUs with matching socket
        for cpu_id, cpu in data.get("cpu", {}).items():
            cpu_socket = (cpu.get("socket") or "").lower()
            if cpu_socket and mobo_socket and (cpu_socket in mobo_socket or mobo_socket in cpu_socket):
                compatible_items.append(cpu)

        compatible_items = shop_func(compatible_items)
        compatible_items.sort(
            key=lambda x: _parse_price_to_int(x.get("price", "0")))
        chips = [chip_func(cpu, "cpu") for cpu in compatible_items[:12]]

        response_lines = [
            "Compatibility Check",
            "",
            f"Component: {mobo_obj.get('name', mobo_name)}",
            "Technical Reason: Found CPU options related to motherboard socket compatibility",
            "",
            "✅ Compatible CPU Options:"
        ]

    elif target_type == "ram":
        # Find RAM with matching type
        for ram_id, ram in data.get("ram", {}).items():
            ram_type = (ram.get("ram_type") or "").lower()
            if not ram_type:
                # Infer from name
                name = (ram.get("name") or "").lower()
                if "ddr5" in name:
                    ram_type = "ddr5"
                elif "ddr4" in name:
                    ram_type = "ddr4"

            if ram_type and mobo_ram_type and ram_type in mobo_ram_type:
                compatible_items.append(ram)

        compatible_items = shop_func(compatible_items)
        compatible_items.sort(
            key=lambda x: _parse_price_to_int(x.get("price", "0")))
        chips = [chip_func(ram, "ram") for ram in compatible_items[:12]]

        response_lines = [
            "Compatibility Check",
            "",
            f"Component: {mobo_obj.get('name', mobo_name)}",
            "Technical Reason: Found RAM options with compatible memory type",
            "",
            "✅ Compatible RAM Options:"
        ]

    else:
        return {"found": False, "message": f"Compatibility type '{target_type}' not supported for motherboards."}

    # Add items to response
    for item in compatible_items[:6]:
        name = item.get("name", "Unknown")
        price = item.get("price", "")
        response_lines.append(f"• {name} — {price}")

    response_lines.extend(["", "Tap any item to view details"])

    return {
        "source": "local-compatibility",
        "found": True,
        "verdict": "compatible",
        "reason": f"Found compatible {target_type} options",
        "text": "\n".join(response_lines),
        "target": mobo_obj.get("name", mobo_name),
        "target_type": "motherboard",
        "compatible_type": target_type,
        "results": chips,
        "chips": chips
    }


def _handle_what_works_with(component_name: str, data: dict, chip_func, shop_func) -> dict:
    """Handle 'What works with [component]' queries"""
    comp_obj, comp_type = _best_match_in_dataset(component_name, data)

    if not comp_obj:
        return {"found": False, "message": f"Component '{component_name}' not found."}

    # Route based on component type
    if comp_type == "motherboard":
        return _handle_motherboard_compatibility(component_name, "cpu", data, chip_func, shop_func)
    elif comp_type == "cpu":
        return _handle_cpu_compatibility(comp_obj, data, chip_func, shop_func)
    elif comp_type == "ram":
        return _handle_ram_compatibility(comp_obj, data, chip_func, shop_func)
    else:
        return {"found": False, "message": f"Compatibility check for {comp_type} not implemented."}


def _handle_cpu_compatibility(cpu_obj: dict, data: dict, chip_func, shop_func) -> dict:
    """Handle compatibility for CPUs"""
    cpu_socket = (cpu_obj.get("socket") or "").lower()
    compatible_mobos = []

    for mobo_id, mobo in data.get("motherboard", {}).items():
        mobo_socket = (mobo.get("socket") or "").lower()
        if mobo_socket and cpu_socket and (mobo_socket in cpu_socket or cpu_socket in mobo_socket):
            compatible_mobos.append(mobo)

    compatible_mobos = shop_func(compatible_mobos)
    compatible_mobos.sort(
        key=lambda x: _parse_price_to_int(x.get("price", "0")))
    chips = [chip_func(mobo, "motherboard") for mobo in compatible_mobos[:12]]

    response_lines = [
        "Compatibility Check",
        "",
        f"Component: {cpu_obj.get('name', 'Unknown CPU')}",
        "Technical Reason: Found motherboard options with matching socket",
        "",
        "✅ Compatible Motherboard Options:"
    ]

    for mobo in compatible_mobos[:6]:
        name = mobo.get("name", "Unknown Motherboard")
        price = mobo.get("price", "")
        response_lines.append(f"• {name} — {price}")

    response_lines.extend(["", "Tap any item to view details"])

    return {
        "source": "local-compatibility",
        "found": True,
        "reason": "Found motherboard options with matching socket",
        "text": "\n".join(response_lines),
        "target": cpu_obj.get('name'),
        "target_type": "cpu",
        "compatible_type": "motherboard",
        "results": chips,
        "chips": chips,
    }


def _handle_ram_compatibility(ram_obj: dict, data: dict, chip_func, shop_func) -> dict:
    """Handle compatibility for RAM"""
    ram_type = (ram_obj.get("ram_type") or "").lower()
    if not ram_type:
        # Infer from name
        name = (ram_obj.get("name") or "").lower()
        if "ddr5" in name:
            ram_type = "ddr5"
        elif "ddr4" in name:
            ram_type = "ddr4"

    compatible_mobos = []
    for mobo_id, mobo in data.get("motherboard", {}).items():
        mobo_ram = (mobo.get("ram_type") or "").lower()
        if ram_type and mobo_ram and ram_type in mobo_ram:
            compatible_mobos.append(mobo)

    compatible_mobos = shop_func(compatible_mobos)
    compatible_mobos.sort(
        key=lambda x: _parse_price_to_int(x.get("price", "0")))
    chips = [chip_func(mobo, "motherboard") for mobo in compatible_mobos[:12]]

    response_lines = [
        "Compatibility Check",
        "",
        f"Component: {ram_obj.get('name', 'Unknown RAM')}",
        f"Technical Reason: Found motherboard options supporting {ram_type.upper() if ram_type else 'the RAM type'}",
        "",
        "✅ Compatible Motherboard Options:"
    ]

    for mobo in compatible_mobos[:6]:
        name = mobo.get("name", "Unknown Motherboard")
        price = mobo.get("price", "")
        response_lines.append(f"• {name} — {price}")

    response_lines.extend(["", "Tap any item to view details"])

    return {
        "source": "local-compatibility",
        "found": True,
        "reason": f"Found motherboards supporting {ram_type or 'RAM type'}",
        "text": "\n".join(response_lines),
        "target": ram_obj.get('name'),
        "target_type": "ram",
        "compatible_type": "motherboard",
        "results": chips,
        "chips": chips,
    }


def _handle_pair_compatibility(item_a: str, item_b: str, data: dict, chip_func, shop_func) -> dict:
    """Handle '[A] works with [B]' style queries"""
    # For now, treat this as "what works with B" using the second item as base
    return _handle_what_works_with(item_b, data, chip_func, shop_func)


def _handle_what_for(target_type: str, base_component: str, data: dict, chip_func, shop_func) -> dict:
    """Handle 'What [component] for [base]' queries"""
    # Map common terms to component types
    type_map = {
        "cpu": "cpu", "processor": "cpu", "cpus": "cpu",
        "gpu": "gpu", "graphics": "gpu", "video": "gpu",
        "ram": "ram", "memory": "ram",
        "motherboard": "motherboard", "mobo": "motherboard", "board": "motherboard",
        "storage": "storage", "ssd": "storage", "hdd": "storage",
        "psu": "psu", "power": "psu",
    }

    target_comp_type = type_map.get(target_type.lower(), target_type.lower())

    # Find the base component
    base_obj, base_type = _best_match_in_dataset(base_component, data)

    if not base_obj:
        return {"found": False, "message": f"Base component '{base_component}' not found."}

    # Route based on base component type
    if base_type == "motherboard" and target_comp_type == "cpu":
        return _handle_motherboard_compatibility(base_component, "cpu", data, chip_func, shop_func)
    elif base_type == "cpu" and target_comp_type == "motherboard":
        return _handle_cpu_compatibility(base_obj, data, chip_func, shop_func)
    elif base_type == "motherboard" and target_comp_type == "ram":
        return _handle_motherboard_compatibility(base_component, "ram", data, chip_func, shop_func)
    elif base_type == "ram" and target_comp_type == "motherboard":
        return _handle_ram_compatibility(base_obj, data, chip_func, shop_func)
    else:
        return {"found": False, "message": f"Compatibility between {base_type} and {target_comp_type} not implemented."}


def _handle_single_component(query: str, data: dict, chip_func, shop_func) -> dict:
    """Handle single component queries with no specific compatibility request"""
    comp_obj, comp_type = _best_match_in_dataset(query, data)

    if not comp_obj:
        return {"found": False, "message": f"Component '{query}' not found."}

    # Return basic info about the component
    chips = [chip_func(comp_obj, comp_type)]

    response_lines = [
        "Component Found",
        "",
        f"Name: {comp_obj.get('name', 'Unknown')}",
        f"Type: {comp_type.upper()}",
        f"Price: {comp_obj.get('price', 'N/A')}",
        "",
        "Tap to view details"
    ]

    # Add some basic specs if available
    specs_to_show = ["socket", "ram_type", "cores", "clock", "vram", "wattage"]
    for spec in specs_to_show:
        if comp_obj.get(spec):
            response_lines.insert(-2, f"{spec.title()}: {comp_obj[spec]}")

    return {
        "source": "local-compatibility",
        "found": True,
        "reason": f"Found {comp_type} component",
        "text": "\n".join(response_lines),
        "target": comp_obj.get('name'),
        "target_type": comp_type,
        "compatible_type": "info",
        "results": chips,
        "chips": chips,
    }
# --------------------------
# Gemini Fallback with Data
# --------------------------
# --------------------------
# Replace this: gemini_fallback_with_data
# --------------------------


def gemini_fallback_with_data(user_input: str, context_data: dict, chat_history: list = None) -> str:
    """
    Fallback when Gemini / GenAI is unavailable or unhelpful.

    Priority order:
      1) Compatibility deterministic handlers (attempts to answer compatibility questions first)
         - Includes a dedicated motherboard M.2 / NVMe deterministic path (uses local ctx data)
         - Falls back to get_compatible_components() if available
      2) Bottleneck analysis (deferred; returned if nothing else matched)
      3) Educational / comparison handlers (Gemini preferred, local KB fallback)
      4) Specs / build requests (budget_builds if available)
      5) General fallback

    The function tries multiple GenAI entrypoints but never relies on them for deterministic compatibility.
    """
    try:
        import re
        q = (user_input or "").strip()
        if not q:
            return "Please provide a question (e.g., 'Ryzen 5 5600 and RTX 3070 bottleneck?')."

        ctx = context_data or {}
        chat_history = chat_history or []
        q_lower = q.lower()
        logger.info("Running gemini_fallback_with_data for query: %s", q)

        # ---------- small helpers ----------
        def _extract_text_from_candidate(candidate):
            if candidate is None:
                return None
            if isinstance(candidate, dict):
                for k in ("text", "content", "output", "output_text"):
                    v = candidate.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            text = getattr(candidate, "text", None) or getattr(
                candidate, "output", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            return str(candidate).strip() if candidate else None

        def try_genai_text(prompt: str, *, max_tokens: int = 800, temperature: float = 0.0):
            # Safe wrapper calling the configured genai/Gemini helper if available
            genai_mod = globals().get("genai", None)
            GEMINI = globals().get("GEMINI_MODEL", None)
            # prefer user-provided try_genai_text if present
            if callable(globals().get("try_genai_text")) and globals().get("try_genai_text") is not try_genai_text:
                try:
                    return globals()["try_genai_text"](prompt)
                except Exception:
                    pass
            if not genai_mod or not GEMINI:
                return None
            try:
                model = genai_mod.GenerativeModel(GEMINI)
                response = model.generate_content(prompt)
                if hasattr(response, "text"):
                    return response.text.strip()
                elif hasattr(response, "candidates") and response.candidates:
                    c = response.candidates[0]
                    if hasattr(c, "content") and hasattr(c.content, "parts"):
                        text_parts = [
                            p.text for p in c.content.parts if hasattr(p, "text")]
                        if text_parts:
                            return "\n".join(text_parts).strip()
                return str(response).strip()
            except Exception as e:
                logger.warning("Gemini call failed: %s", e)
            return None

        # ---------- Intent detectors (compatibility first) ----------
        is_compat_q = bool(re.search(
            r"\b(compatible|works with|fit|fits|is compatible with|pair with|support|will it work|can i use|which.*fit|what.*fit)\b",
            q_lower, flags=re.IGNORECASE))
        comp_keywords = [
            "cpu", "gpu", "graphics card", "motherboard", "mobo", "ram", "memory",
            "ddr4", "ddr5", "nvme", "ssd", "sata", "m.2", "m2", "cooler", "psu", "power supply",
            "connector", "pcie", "slot", "socket", "case", "chassis"
        ]
        comp_kw_present = any(kw in q_lower for kw in comp_keywords)

       # ---------- COMPATIBILITY ----------
        if is_compat_q:
            compat = get_compatible_components(user_input, context_data or {})
            if compat and compat.get("found"):
                chips = compat.get("chips", []) or []
                reason = compat.get("reason", "Socket/chipset compatibility")
                lines = [
                    "✅ Compatible",
                    f"Reason: {reason}",
                    "💡 Recommendation: Choose from these compatible options:"
                ]
                for c in chips[:4]:
                    name = c.get("text", "Unknown")
                    price = c.get("price", "")
                    lines.append(f"• {name}{(' — ' + price) if price else ''}")
                lines.append("")
                lines.append(
                    "Check local retailers for exact models & pricing.")
                return "\n".join(lines)
            return (
                "✅ Compatible\n"
                "Reason: Socket and chipset appear to match based on query.\n"
                "💡 Recommendation: Verify BIOS/firmware updates for older CPU support and check local retailers for exact model compatibility."
            )

        # ---------------- Bottleneck detection (cpu vs gpu) ----------------
        analysis_result = None

        def looks_like_cpu(s: str) -> bool:
            return bool(re.search(r"\b(ryzen|core\s*i|intel|xeon|pentium|athlon|i\d)\b", s, flags=re.IGNORECASE))

        def looks_like_gpu(s: str) -> bool:
            return bool(re.search(r"\b(rtx|gtx|geforce|radeon|rx|vga|nvidia|amd)\b", s, flags=re.IGNORECASE))

        parts = re.split(r"\b(?:and|vs|v|with|,|/|-)\b",
                         q, flags=re.IGNORECASE)
        candidates = [p.strip() for p in parts if p.strip()]

        cpu_name = gpu_name = None
        if len(candidates) >= 2:
            a, b = candidates[0], candidates[1]
            if looks_like_cpu(a) and looks_like_gpu(b):
                cpu_name, gpu_name = a, b
            elif looks_like_gpu(a) and looks_like_cpu(b):
                cpu_name, gpu_name = b, a
            else:
                for part in candidates:
                    if not cpu_name and looks_like_cpu(part):
                        cpu_name = part
                    if not gpu_name and looks_like_gpu(part):
                        gpu_name = part

        # token fallback for CPU / GPU names
        if not cpu_name:
            m = re.search(
                r"((?:[A-Za-z0-9\-\s]+?(?:ryzen|core\s*i|intel|xeon|pentium|athlon|i\d)[A-Za-z0-9\-\s]*))", q, flags=re.IGNORECASE)
            if m:
                cpu_name = m.group(1).strip()
        if not gpu_name:
            m = re.search(
                r"((?:[A-Za-z0-9\-\s]+?(?:rtx|gtx|radeon|rx|geforce|vga|nvidia|amd)[A-Za-z0-9\-\s]*))", q, flags=re.IGNORECASE)
            if m:
                gpu_name = m.group(1).strip()

        if cpu_name and gpu_name:
            # coarse tier inference (keeps behavior simple and deterministic)
            def _tier_from_name(n):
                n = (n or "").lower()
                if re.search(r"\b(7950x3d|7950x|ryzen\s*9|i9|14900k|13900k)\b", n):
                    return "high_end"
                if re.search(r"\b(7700|7700x|ryzen\s*7|i7|14700|13700k)\b", n):
                    return "mid_high"
                if re.search(r"\b(5600\b|5800\b|3600\b|ryzen\s*5|i5|12400)\b", n):
                    return "mid"
                if re.search(r"\b(ryzen\s*3|i3|pentium|celeron|g6400)\b", n):
                    return "low"
                return "unknown"
            cpu_tier = _tier_from_name(cpu_name)
            gpu_tier = _tier_from_name(gpu_name)
            tier_weight = {"low": 1, "mid_low": 2, "mid": 3,
                           "mid_high": 4, "high_end": 5, "unknown": 3}
            cpu_w = tier_weight.get(cpu_tier, 3)
            gpu_w = tier_weight.get(gpu_tier, 3)
            ratio = (gpu_w + 0.0) / (cpu_w if cpu_w else 1.0)

            if ratio >= 2.5:
                verdict = "🔴 MAJOR CPU BOTTLENECK"
                explanation = f"The {cpu_name} is much weaker vs the {gpu_name}. Expect CPU saturation; consider CPU upgrade."
            elif ratio >= 1.4:
                verdict = "🟡 MODERATE CPU BOTTLENECK"
                explanation = f"The {cpu_name} may limit the {gpu_name} in CPU-bound scenarios."
            elif ratio <= 0.4:
                verdict = "🔴 MAJOR GPU BOTTLENECK"
                explanation = f"The {gpu_name} is much weaker than the {cpu_name}; GPU will limit performance."
            elif ratio <= 0.7:
                verdict = "🟡 MODERATE GPU BOTTLENECK"
                explanation = f"The {gpu_name} may somewhat limit the {cpu_name} in graphics-heavy scenarios."
            else:
                verdict = "✅ RELATIVELY BALANCED"
                explanation = f"The {cpu_name} and {gpu_name} are well matched."

            analysis_result = f"Bottleneck Analysis: {cpu_name} + {gpu_name}\n\n{verdict}\n\n{explanation}\n"
            # if no other answer, return this later

        # --------------- Educational / comparison / latest (Gemini preferred) ---------------
        # detect learning intent (moved after compatibility/bottleneck)
        learn_patterns = [
            r"\b(what is|what's|what does|explain|define|how does|why is|purpose of|tell me about|meaning of|difference between|difference|compare)\b",
            r"\b(latest|newest|recent|released|release|list|show me|what are|which are)\b",
        ]
        learn_tokens = re.compile(
            r"\b(pcie|pci[-\s]?express|vram|nvme|ssd|gpu|graphics card|cpu|processor|ram|memory|motherboard|mobo|psu|power supply|bottleneck|fps|tdp|socket|bios|thermal|ddr4|ddr5|sata|bandwidth|latency)\b",
            flags=re.IGNORECASE,
        )
        is_learn_q = any(re.search(p, q_lower, re.IGNORECASE)
                         for p in learn_patterns) or bool(learn_tokens.search(q_lower))

        if is_learn_q:
            # select prompt type
            if re.search(r"difference|compare", q_lower):
                gemini_prompt = (
                    "SYSTEM: You are a PC hardware engineer and educator. Answer concisely in an academic-casual tone.\n\n"
                    "FORMAT:\n1. TL;DR — one-line.\n2. Key specs — 2–4 bullets.\n3. Compatibility — 1 short note.\n4. Real-world difference — 1-2 short paragraphs.\n5. Recommendation — one concise tip.\n\n"
                    f"USER QUESTION: {q}\n\nREPLY:"
                )
            elif re.search(r"latest|newest|list|show me", q_lower):
                gemini_prompt = (
                    "SYSTEM: You are a concise PC hardware curator. Provide 4–8 bullet points: Model — Vendor — Year/Gen — Price (₱, approx) — one factual remark.\n\n"
                    f"USER QUESTION: {q}\n\nREPLY:"
                )
            else:
                gemini_prompt = (
                    "SYSTEM: You are a concise PC hardware expert. Provide a short, accurate explanation.\n\n"
                    "FORMAT:\n1. Definition — 1 sentence.\n2. Function — 1 short paragraph.\n3. Importance — 1 short paragraph.\n4. Example — 1 line tip.\n\n"
                    f"USER QUESTION: {q}\n\nREPLY:"
                )
            response_text = try_genai_text(gemini_prompt)
            if response_text:
                clean = re.sub(
                    r'^(hi|hello|hey)[\s,\!\.]+', '', response_text.strip(), flags=re.IGNORECASE)
                clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
                clean = re.sub(r'\*(.*?)\*', r'\1', clean)
                clean = re.sub(r'(?m)^[\*\-\+]\s+', '• ', clean)
                clean = re.sub(r'\n{3,}', '\n\n', clean)
                if len(clean.split()) > 5:
                    return clean
            # local KB fallback
            local_edu_kb = {
                "pcie": "PCIe (Peripheral Component Interconnect Express) is the high-speed interface for GPUs and SSDs. Each new version increases per-lane bandwidth.",
                "nvme": "NVMe is a storage protocol over PCIe; NVMe SSDs are significantly faster than SATA SSDs.",
                "ssd": "SSD stores data on NAND flash. NVMe SSDs are faster than SATA SSDs.",
                "ram": "RAM temporarily stores active working data. DDR5 offers higher bandwidth and efficiency vs DDR4.",
                "motherboard": "Motherboards define CPU socket, chipset, and supported memory types. Check the manual for exact M.2 wiring and RAM compatibility."
            }
            for k, v in local_edu_kb.items():
                if k in q_lower:
                    return v
            return "Sorry — I couldn’t fetch a Gemini explanation right now. Try again later."

        # ---------------- Specs / build requests ----------------
        try:
            is_specs_build_q = bool(re.search(
                r"\b(build(?: me| a)?|pc build|build pc|build me a pc|recommend(?: me)? (?:a )?pc|suggest(?: me)? (?:a )?pc|gaming pc|office pc|workstation|buy a pc|specs for)\b",
                q_lower, flags=re.IGNORECASE))
            if is_specs_build_q:
                if "budget_builds" in globals() and callable(globals().get("budget_builds")):
                    try:
                        builds = globals()["budget_builds"](user_request=q)
                    except TypeError:
                        try:
                            builds = globals()["budget_builds"](
                                user_request=q, top_n=3)
                        except Exception:
                            builds = []
                    except Exception:
                        try:
                            logger.exception("budget_builds call failed")
                        except Exception:
                            pass
                        builds = []
                    if builds:
                        def _comp_name(comp):
                            if not isinstance(comp, dict):
                                return str(comp)
                            return comp.get("name") or comp.get("title") or comp.get("text") or comp.get("model") or comp.get("id") or "Unknown"
                        out_lines = [
                            f"Here are the top {min(len(builds), 3)} suggested builds for: \"{q}\""]
                        for i, b in enumerate(builds[:3], start=1):
                            total = int(b.get("total_price", 0)) if isinstance(
                                b.get("total_price", 0), (int, float)) else b.get("total_price", 0)
                            cpu_n = _comp_name(b.get("cpu", {}))
                            gpu_n = _comp_name(b.get("gpu", {}))
                            mobo_n = _comp_name(b.get("motherboard", {}))
                            out_lines.append(
                                f"{i}. ₱{total} — CPU: {cpu_n} | GPU: {gpu_n} | Mobo: {mobo_n}")
                        out_lines.append("")
                        out_lines.append(
                            "Tip: ask 'show me details for build 1' or 'show components for build 2' to view more.")
                        return "\n".join(out_lines)
                else:
                    return ("I detected a request to build a PC. The local build generator is not available right now, "
                            "but you can ask: 'Build me a gaming PC' or 'Build me an office PC' and I'll try to help.")
        except Exception:
            try:
                logger.exception("Specs/build handler failed; continuing.")
            except Exception:
                pass

        # --------------- Return deferred bottleneck if present --------------
        if 'analysis_result' in locals() and analysis_result:
            return analysis_result

        # ----------------- Final general fallback -------------------------
        return ("I wasn’t able to detect a specific PC-building topic. Try asking about components or compatibility with exact model names. "
                "Example: 'Is Ryzen 9 7900X compatible with ASUS ROG STRIX B650E-F?'")

    except Exception:
        try:
            logger.exception("Gemini fallback failed.")
        except Exception:
            pass
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
        # Normalize query early for all downstream handlers
        q_raw = (user_input or "").strip()
        lower_q = q_raw.lower()

        # For backward compatibility, keep alias "q" used in other parts
        q = q_raw

        # Safe debug logging
        try:
            logger.info("[GET_AI_RESPONSE] incoming_query=%s", q_raw)
        except Exception:
            pass

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
# ---------- 0) Bottleneck natural-language detection (robust replacement) ----------
            try:
                # quick detect explicit bottleneck intent
                if re.search(r"\b(bottleneck|bottlenecks|bottlenecking)\b", lower_q, flags=re.IGNORECASE):
                    logger.info(
                        "Routing to bottleneck analyzer for query: %s", q)
                    try:
                        # call the robust fallback which will attempt Gemini and then local deterministic analysis
                        gem_text = gemini_fallback_with_data(
                            user_input, make_public_data(data), chat_history or [])
                        # gemini_fallback_with_data returns a compact string (or analysis) — keep response shape consistent
                        return {"source": "bottleneck", "text": gem_text, "used_fallback": True}
                    except Exception as e:
                        logger.exception(
                            "Bottleneck handler failed; falling back to local simple reply: %s", e)
                        return {"source": "bottleneck", "text": "Sorry — I couldn't analyze the bottleneck automatically. Please provide clearer CPU and GPU names (e.g., 'Ryzen 5 5600 and RTX 3070').", "used_fallback": True}
            except Exception:
                logger.exception(
                    "Error while checking for bottleneck intent; continuing to other handlers.")
# ---------- end bottleneck detection ----------

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
                        "You are ARIA — a concise PC-building assistant.\n\n"
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

        # ========== 2) COMPATIBILITY CHECK (more permissive) ==========
        # ========== 2) COMPATIBILITY CHECK ==========
        compatibility_patterns = [
            r"\bcompatible\b",
            r"\bworks with\b",
            r"\bfit\b",
            r"\bsupport\b",
            r"\bwhat.*fit\b",
            r"\bwhich.*fit\b",
            r"\bfor.*socket\b",
            r"\bwhat.*work with\b",
            r"\bmotherboard.*cpu\b",
            r"\bcpu.*motherboard\b",
            r"\bcooler.*socket\b",
        ]

        is_compatibility_q = any(
            re.search(pattern, lower_q, flags=re.IGNORECASE)
            for pattern in compatibility_patterns
        )

        if is_compatibility_q:
            try:
                logger.info(f"🔄 Routing to compatibility checker: {q}")
                compat = get_compatible_components(user_input, data)

                if compat and compat.get("found"):
                    target_name = compat.get("target", "Unknown component")
                    comp_type_label = (compat.get(
                        "compatible_type") or "component").upper()
                    # robustly read chips: prefer 'chips', fall back to 'results'
                    raw_chips = compat.get(
                        "chips") or compat.get("results") or []
                    reason = compat.get(
                        "reason", "Compatibility check completed")

                    # Build clean response
                    response_lines = [
                        "Compatibility Check",
                        "",
                        f"Component: {target_name}",
                        f"Technical Reason: {reason}",
                        "",
                    ]

                    if raw_chips:
                        response_lines.append(
                            f"✅ Compatible {comp_type_label} Options:")
                        for chip in raw_chips[:6]:  # Limit to 6 items
                            name = chip.get("text", "Unknown")
                            price = chip.get("price", "")
                            price_str = f" — {price}" if price else ""
                            response_lines.append(f"• {name}{price_str}")
                        response_lines.append("")
                        response_lines.append("Tap any item to view details")
                    else:
                        response_lines.append(
                            "No specific compatible items found in local database."
                        )

                    friendly_text = "\n".join(response_lines)

                    return {
                        "source": "local-compatibility",
                        "type": "compatibility",
                        "target": target_name,
                        "compatible_type": compat.get("compatible_type"),
                        "reason": reason,
                        "results": raw_chips,
                        "chips": raw_chips,
                        "text": friendly_text,
                    }
                else:
                    # Fallback to Gemini for compatibility
                    logger.info(
                        "No local compatibility found, using Gemini fallback")
                    gem_text = gemini_fallback_with_data(
                        user_input, make_public_data(data))
                    return {"source": "gemini-fallback", "text": gem_text, "used_fallback": True}

            except Exception as e:
                logger.exception(f"Compatibility check failed: {e}")
                # Fall through to other handlers (YUNG CHIPS HINDI NABABASA)

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
            r"\bshop(s)?\b",
            r"\bstore(s)?\b",
            r"\bretailer(s)?\b",
            r"\bwhere to buy\b",
            r"\bwhere can i buy\b",
            r"\bpurchase(s)?\b",
            r"\bbuy(s)?\b",
            r"\bcomputer shop(s)?\b",
            r"\bpc shop(s)?\b",
            r"\bcomputer store(s)?\b",
            r"\bsuggest shop(s)?\b",
            r"\bsuggest any computer shop(s)?\b",

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
                            f"• {client_shop.get('name', 'SMFP Computer')} ")
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
        list_keywords = ["list", "show me", "available", "all the",
                         "give me the list", "give me a list", "options", "what are the"]
        is_list_query = any(keyword in lower_q for keyword in list_keywords)

        component_types = {
            "cpu cooler": "cpu_cooler", "cpu_cooler": "cpu_cooler", "cooler": "cpu_cooler",
            "cpu": "cpu", "cpus": "cpu", "processor": "cpu", "processors": "cpu",
            "gpu": "gpu", "gpus": "gpu", "graphics card": "gpu", "graphics cards": "gpu",
            "video card": "gpu", "video cards": "gpu",
            "motherboard": "motherboard", "motherboards": "motherboard", "mobo": "motherboard",
            "ram": "ram", "memory": "ram", "ddr": "ram", "ddr4": "ram", "ddr5": "ram",
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
            ram_type_filter = None

            # Check for specific RAM type requests
            if "ddr5" in lower_q:
                ram_type_filter = "ddr5"
                matched_type = "ram"
            elif "ddr4" in lower_q:
                ram_type_filter = "ddr4"
                matched_type = "ram"

            # If no specific RAM type requested, try general component matching
            if not matched_type:
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

            # Brand filtering
            for brand, patterns in brand_patterns.items():
                if any(pattern in lower_q for pattern in patterns):
                    requested_brands.append(brand)

            if matched_type:
                items = list(data.get(matched_type, {}).values()) or []

                # Apply RAM type filtering if requested
                if matched_type == "ram" and ram_type_filter:
                    filtered_items = []
                    for item in items:
                        name = (item.get("name") or "").lower()
                        ram_type = (item.get("ram_type") or "").lower()
                        if ram_type_filter in name or ram_type_filter in ram_type:
                            filtered_items.append(item)
                    items = filtered_items

                # Apply brand filtering
                if requested_brands and matched_type in ("cpu", "gpu", "ram"):
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

                    items = [it for it in items if matches_brand(
                        it.get("name", ""), brand_keys)]

                if items:
                    lines = []
                    limit = min(20, len(items))

                    for comp in items[:limit]:
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
                            watt = _parse_psu_wattage(comp)
                            watt_str = f" — {watt} W" if watt > 0 else " — Unknown wattage"
                            brand = name.split()[0] if name else "Unknown"
                            lines.append(f"{brand}{watt_str}{price_str}")

                        elif matched_type == "ram":
                            # For RAM, show capacity and type if available
                            capacity = comp.get("capacity", "")
                            ram_type = comp.get("ram_type", "")
                            specs = []
                            if capacity:
                                specs.append(capacity)
                            if ram_type:
                                specs.append(ram_type.upper())
                            spec_str = f" — {', '.join(specs)}" if specs else ""
                            lines.append(f"{name}{spec_str}{price_str}")

                        else:
                            lines.append(f"{name}{price_str}")

                    joined_lines = "\n".join(lines)
                    type_label = matched_type.replace('_', ' ').upper()

                    if ram_type_filter:
                        friendly_text = f"Available {ram_type_filter.upper()} {type_label}:\n\n{joined_lines}"
                    elif requested_brands:
                        brand_label = " ".join([b.upper()
                                               for b in requested_brands])
                        friendly_text = f"Available {brand_label} {type_label}s:\n\n{joined_lines}"
                    else:
                        friendly_text = f"Available {type_label}s:\n\n{joined_lines}"

                    if len(items) > limit:
                        friendly_text += f"\n\n(Showing {limit} of {len(items)} items. Ask for specific models for more details.)"

                    return {"source": "local-list", "type": "component_list_textonly", "target": matched_type, "results": [], "text": friendly_text}

                type_label = matched_type.replace('_', ' ').upper()
                return {"source": "local-list", "type": "component_list_missing", "target": matched_type, "results": [], "text": f"No {type_label} entries found in the current dataset."}

            else:
                categories = [k for k, v in data.items() if v]
                friendly_text = f"I have data available for: {', '.join(categories)}. Ask 'list CPUs', 'list GPUs', 'list DDR5 memory', etc."
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


def _parse_psu_wattage(psu_obj_or_str):
    try:
        if isinstance(psu_obj_or_str, dict):
            val = psu_obj_or_str.get("wattage") or psu_obj_or_str.get(
                "watt") or psu_obj_or_str.get("power") or ""
            return _parse_wattage_to_int(val)
        return _parse_wattage_to_int(psu_obj_or_str)
    except Exception:
        return 0

    # Try direct wattage fields first
    for k in ("wattage", "w", "power", "rating", "max_power"):
        if k in psu_obj and psu_obj.get(k) is not None:
            w = _parse_watt_value(psu_obj.get(k))
            if w:
                return w

    # Try to parse from name with better pattern matching
    name = (psu_obj.get("name") or "").lower()

    # Common PSU naming patterns
    patterns = [
        r"(\d{3,4})\s*w",  # "650W", "750 W"
        r"(\d{3,4})w",     # "650w", "750w"
        r"\s(\d{3,4})\s",  # "PSU 650 Gold"
        r"^(\d{3,4})\s",   # "650 Power Supply"
        r"\((\d{3,4})w\)",  # "PSU (650w)"
    ]

    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue

    # Final fallback: look for any 3-4 digit number that's likely wattage
    digits = re.findall(r'\b(\d{3,4})\b', name)
    for digit in digits:
        watt = int(digit)
        if 400 <= watt <= 2000:  # Reasonable PSU wattage range
            return watt

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

def improve_component_identification(comp_a, comp_b):
    """
    Identify which component is CPU and which is GPU
    """
    def is_likely_cpu(comp):
        if not comp:
            return False
        name = (comp.get('name') or '').lower()
        comp_type = (comp.get('type') or '').lower()
        if 'cpu' in comp_type:
            return True
        return any(term in name for term in ['ryzen', 'core i', 'i3', 'i5', 'i7', 'i9', 'xeon', 'athlon', 'processor'])

    def is_likely_gpu(comp):
        if not comp:
            return False
        name = (comp.get('name') or '').lower()
        comp_type = (comp.get('type') or '').lower()
        if 'gpu' in comp_type or 'graphics' in comp_type:
            return True
        return any(term in name for term in ['rtx', 'gtx', 'radeon', 'rx', 'geforce', 'nvidia', 'video card', 'graphics card'])

    if is_likely_cpu(comp_a) and is_likely_gpu(comp_b):
        return comp_a, comp_b
    elif is_likely_cpu(comp_b) and is_likely_gpu(comp_a):
        return comp_b, comp_a
    else:
        # Fallback: assume first is CPU, second is GPU if uncertain
        return comp_a, comp_b


def _safe_int_from_field(val, default=0):
    """
    Safely extract an integer from a field that might be int, numeric string,
    or contain units/extra text (e.g. "8GB", "8 GB", "approx 8", "8.0").
    Returns `default` if nothing parseable is found. Never raises.
    """
    try:
        if val is None:
            return int(default)
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        s = str(val).strip().lower()
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            try:
                return int(float(m.group(1)))
            except Exception:
                return int(m.group(1))
        digits = re.sub(r"[^\d]", "", s)
        if digits:
            return int(digits)
        return int(default)
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0


def estimate_gpu_capacity(gpu: dict) -> float:
    """
    Heuristic GPU capacity estimator. Returns a numeric capacity score (higher = stronger GPU).
    Uses 'gpu_score' or 'score' if present; otherwise estimate from memory, clocks, tdp, or name tokens.
    """
    try:
        if not gpu:
            # conservative small positive default to avoid division by zero downstream
            return 1000.0

        # prefer an explicit numeric score if present
        score = gpu.get("gpu_score") or gpu.get(
            "score") or gpu.get("g3d_score")
        if score is not None:
            try:
                return float(score)
            except Exception:
                pass

        # helper to safely parse ints from strings like "8 GB", "8192", etc.
        def _parse_int_safe(val):
            try:
                if val is None:
                    return 0
                s = str(val)
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else int(float(s))
            except Exception:
                return 0

        # memory based heuristics (interpret as GB unless obviously MB)
        mem_gb = 0
        try:
            mem = gpu.get("memory") or gpu.get(
                "vram") or gpu.get("memory_gb") or gpu.get("mem")
            if mem:
                # mem may be "8 GB", "8192", "8192 MB", "8GB"
                s = str(mem).lower()
                if "mb" in s and re.search(r"(\d+)", s):
                    # e.g. "8192 MB" => convert to GB
                    mem_mb = int(re.search(r"(\d+)", s).group(1))
                    mem_gb = max(1, int(round(mem_mb / 1024.0)))
                else:
                    # try parse first number as GB
                    m = re.search(r"(\d+)", s)
                    if m:
                        mem_gb = int(m.group(1))
                        # if value looks like MB (very large) convert
                        if mem_gb > 64 and mem_gb <= 16384:
                            # likely given as MB (e.g., 8192) -> convert to GB
                            mem_gb = int(round(mem_gb / 1024.0))
        except Exception:
            mem_gb = 0

        tdp = _parse_int_safe(gpu.get("tdp") or gpu.get(
            "power") or gpu.get("tdp_w") or 0)

        # clock parse (GHz or MHz) -> normalize as GHz if needed
        clock = 0.0
        try:
            s = str(gpu.get("clock") or gpu.get(
                "boost_clock") or gpu.get("boost") or "")
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if m:
                clock = float(m.group(1))
                # if value looks like MHz (>1000), convert to GHz
                if clock > 1000:
                    clock = clock / 1000.0
        except Exception:
            clock = 0.0

        # If any numeric hints exist, build a combined heuristic score
        if mem_gb or tdp or clock:
            # weights chosen so typical modern GPUs are in thousands..tens of thousands
            score_est = (mem_gb * 200.0) + (tdp * 2.0) + (clock * 50.0)
            rank = gpu.get("gpu_rank") or gpu.get("rank")
            if rank:
                try:
                    score_est *= (1.0 + (float(rank) / 100.0))
                except Exception:
                    pass
            return max(100.0, float(score_est))

        # name-based fallbacks (coarse)
        name = (gpu.get("name") or gpu.get("model") or "").lower()
        name_map = {
            r"\b4090\b": 30000.0,
            r"\b4080\b": 20000.0,
            r"\b4070\b": 12000.0,
            r"\b3090\b": 22000.0,
            r"\brtx\b.*4090": 30000.0,
            r"\bgtx\b": 6000.0,
            r"\bradeon\b.*7900\b": 18000.0,
            r"\brx\s?7[0-9]{2}\b": 15000.0,
        }
        for pat, val in name_map.items():
            try:
                if re.search(pat, name):
                    return float(val)
            except Exception:
                continue

        # final conservative fallback (non-zero)
        return 1000.0
    except Exception:
        # last-resort fallback (non-zero)
        return 1000.0


def estimate_cpu_capacity(cpu: dict) -> float:
    """
    Heuristic CPU capacity estimator. Returns a numeric capacity score (higher = stronger CPU).
    Uses 'score' if present, otherwise estimates from cores/threads/base/boost clocks and tdp.
    If those fields are missing, falls back to a name-based heuristic.
    """
    try:
        if not cpu:
            return 100.0

        # direct numeric score preferred
        score = cpu.get("score") or cpu.get("cpu_score") or cpu.get("passmark")
        if score is not None:
            try:
                return float(score)
            except Exception:
                pass

        # safe int parser
        def _parse_int_safe(val):
            try:
                if val is None:
                    return 0
                s = str(val)
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else int(float(s))
            except Exception:
                return 0

        # safe ghz parser
        def _parse_ghz(val):
            try:
                if val is None:
                    return 0.0
                s = str(val)
                m = re.search(r"(\d+(?:\.\d+)?)", s)
                if m:
                    ghz = float(m.group(1))
                    # if number is > 1000 it's probably MHz -> convert
                    if ghz > 1000:
                        ghz = ghz / 1000.0
                    return ghz
                return 0.0
            except Exception:
                return 0.0

        cores = _parse_int_safe(cpu.get("cores") or cpu.get(
            "physical_cores") or cpu.get("core_count") or 0)
        threads = _parse_int_safe(
            cpu.get("threads") or cpu.get("logical_cores") or 0)
        base_clock = _parse_ghz(cpu.get("base_clock") or cpu.get(
            "base") or cpu.get("clock") or 0)
        boost_clock = _parse_ghz(
            cpu.get("boost_clock") or cpu.get("boost") or 0)
        tdp = _parse_int_safe(cpu.get("tdp") or cpu.get(
            "power") or cpu.get("tdp_w") or 0)

        # Use numeric fields if present
        if cores or threads or base_clock or boost_clock or tdp:
            core_factor = float(cores) if cores else (
                float(threads) * 0.6 if threads else 0.0)
            clock_factor = base_clock if base_clock else (
                boost_clock * 0.9 if boost_clock else 0.0)
            # Build a heuristic that produces values in a similar order to GPU scores (thousands)
            score_est = (core_factor * max(clock_factor, 1.0) * 100.0)
            score_est += (boost_clock * 10.0) + (threads * 5.0)
            score_est += (tdp * 0.5)
            return max(50.0, float(score_est))

        # fallback by name token matching when numeric/specs are missing
        name = (cpu.get("name") or cpu.get("model") or "").lower()
        name_map = {
            r"\b5600\b": 3500.0,   # Ryzen 5 5600
            r"\b3600\b": 3000.0,
            r"\b5800\b": 4200.0,
            r"\b7700\b": 6500.0,
            r"\b14700\b": 12000.0,
            r"\bi9\b": 14000.0,
            r"\bi7\b": 9000.0,
            r"\bi5\b": 5000.0,
            r"\bryzen\b.*\b9\b": 14000.0,
            r"\bxeon\b": 8000.0,
        }
        for pat, val in name_map.items():
            try:
                if re.search(pat, name):
                    return float(val)
            except Exception:
                continue

        # conservative fallback
        return 100.0
    except Exception:
        return 100.0


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
    print("💡 ARIA Assistant with Gemini 2.5 Flash-Lite fallback\n")
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
