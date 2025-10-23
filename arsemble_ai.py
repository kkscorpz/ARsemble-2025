from typing import Any, Dict, Tuple, List
import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from components_data import data
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
    logger.info("Dataset summary: cpu=%d gpu=%d motherboard=%d ram=%d storage=%d psu=%d",
                len(data.get("cpu", {})),
                len(data.get("gpu", {})),
                len(data.get("motherboard", {})),
                len(data.get("ram", {})),
                len(data.get("storage", {})),
                len(data.get("psu", {})))
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
    Robust fuzzy-ish matcher:
      - tries slug substring
      - normalized substring
      - token subset
      - scoring fallback (token overlap + numeric bonus)
      - raw substring last-resort
    Returns (obj, comp_type) or (None, None)
    """
    if not query:
        return None, None

    q_slug = slugify(query)
    q_norm = _normalize_text_for_match(query)
    q_tokens = set([t for t in re.split(r"\s+", q_norm) if t])

    # Quick slug substring match (fast path)
    for comp_type, comps in data.items():
        for obj in comps.values():
            name = obj.get("name", "") or ""
            name_slug = slugify(name)
            if name_slug and name_slug in q_slug:
                return obj, comp_type

    # Normalized substring match (handles punctuation/hyphen differences)
    for comp_type, comps in data.items():
        for obj in comps.values():
            name = obj.get("name", "") or ""
            name_norm = _normalize_text_for_match(name)
            if name_norm and name_norm in q_norm:
                return obj, comp_type

    # Token subset match (all name tokens appear in query tokens)
    for comp_type, comps in data.items():
        for obj in comps.values():
            name = obj.get("name", "") or ""
            name_norm = _normalize_text_for_match(name)
            name_tokens = set([t for t in re.split(r"\s+", name_norm) if t])
            if name_tokens and name_tokens.issubset(q_tokens):
                return obj, comp_type

    # Scoring fallback: token overlap + numeric bonus
    best_obj = None
    best_type = None
    best_score = 0
    for comp_type, comps in data.items():
        for obj in comps.values():
            name = obj.get("name", "") or ""
            name_norm = _normalize_text_for_match(name)
            name_tokens = set([t for t in re.split(r"\s+", name_norm) if t])
            if not name_tokens:
                continue
            common = len(q_tokens & name_tokens)
            numeric_bonus = sum(2 for t in name_tokens if re.search(
                r"\d", t) and t in q_tokens)
            score = common + numeric_bonus
            if score > best_score:
                best_score = score
                best_obj = obj
                best_type = comp_type

    if best_obj and best_score >= 1:
        return best_obj, best_type

    # Last resort: case-insensitive substring on raw human name
    lower_q = query.lower()
    for comp_type, comps in data.items():
        for obj in comps.values():
            if lower_q in (obj.get("name", "") or "").lower():
                return obj, comp_type

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
    for comp in ["cpu", "gpu", "motherboard", "ram", "storage", "psu"]:
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
            "id": f"{ctype}:{ctype}-{slugify(safe_name)}",
            "text": safe_name,
            "price": v.get("price", ""),
            "type": ctype,
            "meta": v
        })
    return chips


def get_compatible_components(query: str, data: dict) -> dict:
    """
    Robust compatibility resolver that returns a dict with keys:
      - found (bool)
      - target (str)
      - compatible_type (str)
      - reason (optional)
      - chips (list)
      - message (optional)

    Features:
      - DDRx-aware motherboard filtering (e.g. "Which motherboards support DDR4?")
      - GPU <- motherboard rule (asked_type == "gpu" and target_type == "motherboard")
      - Socket-aware CPU queries (detects AM4, AM5, LGA1700, etc.)
      - Defensive: always returns a dict, logs exceptions
      - Debug print to show which dataset entry matched (for CLI)
    """
    result = {"found": False}
    try:
        q = (query or "").lower()
        q_norm = _normalize_text_for_match(query)
        component_types = ["cpu", "gpu",
                           "motherboard", "ram", "storage", "psu"]

        # detect what the user is asking FOR (asked_type), e.g. "motherboard" or "cpu"
        asked_type = None
        for comp_type in component_types:
            if re.search(rf"\b{comp_type}s?\b", q):
                asked_type = comp_type
                break

        # robustly detect a named component mentioned in query
        target_obj, target_type = _best_match_in_dataset(query, data)
        print(
            "DEBUG: _best_match_in_dataset ->",
            getattr(target_obj, "get", None) and target_obj.get(
                "name") or None,
            target_type,
        )

        # ---------- CASE A: user asked for a TYPE and mentioned a specific component ----------
        if asked_type and target_obj:
            # USER: "What CPU is compatible with RAMSTA RS-B450MP?"
            if asked_type == "cpu" and target_type == "motherboard":
                mobo_socket = (target_obj.get("socket") or "").lower()
                if not mobo_socket:
                    result[
                        "message"] = f"No socket information available for {target_obj.get('name', 'that motherboard')}."
                    return result
                matches = [
                    c
                    for c in data.get("cpu", {}).values()
                    if mobo_socket and mobo_socket in (c.get("socket") or "").lower()
                ]
                if not matches:
                    result[
                        "message"] = f"I couldn’t find any CPUs compatible with {target_obj.get('name')}."
                    return result
                result.update({
                    "found": True,
                    "target": target_obj.get("name"),
                    "compatible_type": "cpu",
                    "reason": f"CPUs that fit socket {target_obj.get('socket', '')}",
                    "chips": build_chips(matches, "cpu"),
                })
                return result

            # USER: "What motherboard is compatible with Ryzen 5 5600X?"
            if asked_type == "motherboard" and target_type == "cpu":
                cpu_socket = (target_obj.get("socket") or "").lower()
                if not cpu_socket:
                    result[
                        "message"] = f"No socket information available for {target_obj.get('name', 'that CPU')}."
                    return result
                matches = [
                    m
                    for m in data.get("motherboard", {}).values()
                    if cpu_socket and cpu_socket in (m.get("socket") or "").lower()
                ]
                if not matches:
                    result[
                        "message"] = f"I couldn’t find any motherboards compatible with {target_obj.get('name')}."
                    return result
                result.update({
                    "found": True,
                    "target": target_obj.get("name"),
                    "compatible_type": "motherboard",
                    "reason": f"Motherboards with socket {target_obj.get('socket', '')}",
                    "chips": build_chips(matches, "motherboard"),
                })
                return result

            # USER: "What motherboard is compatible with <RAM module>?"
            if asked_type == "motherboard" and target_type == "ram":
                ram_type = (target_obj.get("ram_type") or "").lower()
                if not ram_type:
                    result[
                        "message"] = f"No RAM type information available for {target_obj.get('name', 'that RAM')}."
                    return result
                matches = [
                    m
                    for m in data.get("motherboard", {}).values()
                    if ram_type and ram_type in (m.get("ram_type") or "").lower()
                ]
                if not matches:
                    result[
                        "message"] = f"I couldn’t find any motherboards compatible with {target_obj.get('name')}."
                    return result
                result.update({
                    "found": True,
                    "target": target_obj.get("name"),
                    "compatible_type": "motherboard",
                    "reason": f"Motherboards that support {target_obj.get('ram_type', '')}",
                    "chips": build_chips(matches, "motherboard"),
                })
                return result

            # USER: "What PSU is compatible with RTX 3060?" (heuristic)
            if asked_type == "psu" and target_type == "gpu":
                matches = list(data.get("psu", {}).values())
                if not matches:
                    result[
                        "message"] = f"I couldn’t find PSUs compatible with {target_obj.get('name')}."
                    return result
                result.update({
                    "found": True,
                    "target": target_obj.get("name"),
                    "compatible_type": "psu",
                    "reason": "Suggested PSUs (check wattage vs GPU power draw)",
                    "chips": build_chips(matches, "psu"),
                })
                return result

            # GPU requested for a motherboard (most motherboards have PCIe x16)
            if asked_type == "gpu" and target_type == "motherboard":
                matches = list(data.get("gpu", {}).values())
                if not matches:
                    result[
                        "message"] = f"I couldn’t find any GPUs compatible with {target_obj.get('name')}."
                    return result
                result.update({
                    "found": True,
                    "target": target_obj.get("name"),
                    "compatible_type": "gpu",
                    "reason": "GPUs that fit PCIe x16 slots (verify physical fit & power requirements)",
                    "chips": build_chips(matches, "gpu"),
                })
                return result

            # fallback for unhandled combos
            result[
                "message"] = f"I don’t have a rule for matching {asked_type} with {target_obj.get('name')} in the dataset."
            return result

        # ---------- CASE B: asked_type provided but no specific named target (type-to-type) ----------
        if asked_type and not target_obj:
            # DDRx-specific motherboard query: "Which motherboards support DDR4?"
            if asked_type == "motherboard" and (("ddr4" in q) or ("ddr5" in q) or re.search(r"\bddr\d\b", q)):
                desired = "ddr5" if "ddr5" in q else (
                    "ddr4" if "ddr4" in q else None)
                matches = [
                    m
                    for m in data.get("motherboard", {}).values()
                    if desired and desired in (m.get("ram_type") or "").lower()
                ]
                if matches:
                    result.update({
                        "found": True,
                        "target": f"Motherboards supporting {desired.upper()}",
                        "compatible_type": "motherboard",
                        "reason": f"Motherboards that support {desired.upper()} RAM",
                        "chips": build_chips(matches, "motherboard"),
                    })
                    return result

            # Socket-specific CPU query: "Which CPUs work with AM4 socket motherboards?"
            # Detect common socket tokens in the normalized query (am4, am5, lga1700, etc.)
            socket_match = re.search(r"\b(am\d+|lga\s*\d+)\b", q_norm or q)
            if asked_type == "cpu" and socket_match:
                # normalize e.g. "lga 1700" -> "lga1700"
                sock_token = socket_match.group(1).replace(" ", "").lower()
                matches = [
                    c
                    for c in data.get("cpu", {}).values()
                    if sock_token and sock_token in (c.get("socket") or "").lower().replace(" ", "")
                ]
                if matches:
                    result.update({
                        "found": True,
                        "target": f"CPUs for socket {sock_token.upper()}",
                        "compatible_type": "cpu",
                        "reason": f"CPUs that fit socket {sock_token.upper()}",
                        "chips": build_chips(matches, "cpu"),
                    })
                    return result

            if asked_type == "gpu":
                # user asked "which GPUs" or similar - show GPUs
                matches = list(data.get("gpu", {}).values())
                result.update({
                    "found": True,
                    "target": "GPUs",
                    "compatible_type": "gpu",
                    "reason": "Available GPUs (verify physical fit & power requirements)",
                    "chips": build_chips(matches, "gpu"),
                })
                return result

            if asked_type == "cpu":
                matches = list(data.get("cpu", {}).values())
                result.update({
                    "found": True,
                    "target": "CPUs",
                    "compatible_type": "cpu",
                    "reason": "CPUs grouped by socket",
                    "chips": build_chips(matches, "cpu"),
                })
                return result

            if asked_type == "ram":
                # list motherboards (since user asked about RAM -> show motherboards supporting RAM types)
                matches = list(data.get("motherboard", {}).values())
                result.update({
                    "found": True,
                    "target": "RAM modules",
                    "compatible_type": "motherboard",
                    "reason": "Motherboards that support DDR4/DDR5",
                    "chips": build_chips(matches, "motherboard"),
                })
                return result

            result["message"] = f"I can show typical compatible components for {asked_type}, but for precise matching I need a specific model."
            return result

        # ---------- CASE C: only a specific component mentioned but user didn't say what they want ----------
        if target_obj and not asked_type:
            # default to suggesting motherboards for CPUs
            if target_type == "cpu":
                cpu_socket = (target_obj.get("socket") or "").lower()
                matches = [
                    m
                    for m in data.get("motherboard", {}).values()
                    if cpu_socket and cpu_socket in (m.get("socket") or "").lower()
                ]
                if matches:
                    result.update({
                        "found": True,
                        "target": target_obj.get("name"),
                        "compatible_type": "motherboard",
                        "reason": f"Motherboards with socket {target_obj.get('socket', '')}",
                        "chips": build_chips(matches, "motherboard"),
                    })
                    return result
            result["message"] = "I found that component but not sure what compatibility you want—ask e.g. 'What motherboard is compatible with <name>?'"
            return result

        # ---------- nothing matched ----------
        result["message"] = "I couldn’t find any component in the dataset that matches what you mentioned."
        return result

    except Exception as e:
        logger.exception("get_compatible_components failed:")
        return {"found": False, "error": str(e), "message": "Compatibility resolver error."}


# --------------------------
# Gemini Fallback with Data
# --------------------------


def gemini_fallback_with_data(user_input: str, context_data: dict) -> str:
    """
    Smarter Gemini fallback:
    - Returns nicely formatted build summaries when the user asks about builds or lists components.
    - Returns clean spec sheets, single-attribute answers, or short educational answers for other queries.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        context_snippet = json.dumps(context_data, indent=2)[:6000]
        user_lower = (user_input or "").lower()

        # Heuristics to detect query intent
        is_specs_query = any(k in user_lower for k in [
                             "details", "specs", "specifications", "show me"])
        is_attribute_query = any(k in user_lower for k in [
                                 "power", "vram", "price", "socket", "clock", "wattage", "compatibility"])
        is_educational_query = any(k in user_lower for k in [
                                   "what is", "what does", "why", "how", "difference", "explain"])
        is_build_query = any(k in user_lower for k in [
                             "build", "recommend", "recommendation", "estimated total", "estimate", "pc build", "₱", "budget", "+", "with"])
        # also treat queries that list multiple components (e.g. "RTX 3060 + Ryzen 5 5600X") as build queries
        if len(re.findall(r"\b(ryzen|intel|rtx|gtx|rx|core|i3|i5|i7|i9)\b", user_lower)) >= 2:
            is_build_query = True

        # Detect if a component name is present
        possible_names = ["ryzen", "intel", "core", "i3", "i5", "i7", "i9",
                          "gtx", "rtx", "msi", "asus", "gigabyte", "b550", "am4", "am5",
                          "inplay", "corsair", "cooler master", "thermaltake", "evga", "psu"]
        mentions_component = any(name in user_lower for name in possible_names)

        # ---------- Build summary formatting instruction ----------
        build_template = f"""
You are ARsemble AI — a professional PC-building assistant.
When the user asks about a PC build or lists multiple components, return a single clean build summary USING THIS EXACT FORMAT (use emojis and dividers, keep line breaks and spacing):

💻 PC Build Summary
────────────────────────────
Estimated Total: ₱<total>
Score: <score>

🧠 CPU: <name> — <price>
🎮 GPU: <name> — <price>
🧩 Motherboard: <name> — <price>
⚡ RAM: <name> — <price>
💾 Storage: <name> — <price>
🔌 PSU: <name> — <price>

⚙️ Estimated Power Draw: <number> W
🔋 Recommended PSU: <number> W

Suggested PSUs:
   • <psu_name> — <price>
   • <psu_name> — <price>
────────────────────────────

- If some component is missing, omit that line.
- Don't print duplicate PSUs.
- Use commas for thousands in prices (e.g., ₱28,200).
- Do not include extra narration or analysis — only the formatted summary.
"""

        # ---------- Spec sheet instruction ----------
        specs_template = f"""
You're ARsemble AI — a professional PC-building assistant.
The user wants full details for a specific component.

Dataset (trimmed):
{context_snippet}

User query: "{user_input}"

Return the clean specs only. Use this format:
Name: <Component Name>
Type: <Type>
Then list any fields (VRAM, Clock, Power, Slot, Price, Compatibility).
No emojis, no markdown, just aligned text.
"""

        # ---------- Attribute instruction ----------
        attribute_template = f"""
You are ARsemble AI.
The user wants one attribute (like power, VRAM, or clock) for a specific component.

Dataset (trimmed):
{context_snippet}

User query: "{user_input}"

Find the matching component and return ONLY that attribute value, e.g.:
"~60 Watts" or "₱8,500" or "3.6 GHz / 4.0 GHz Boost"
No extra words or sentences.
"""

        # ---------- Educational instruction ----------
        educational_template = f"""
You're "ARsemble AI" — a friendly PC-building buddy who explains hardware concepts casually.
Keep your answer simple, conversational, and easy to understand (2–3 sentences max).
No technical dump, no symbols — just a clear explanation.

Dataset (trimmed):
{context_snippet}

User query: "{user_input}"
"""

        # ---------- Default concise instruction ----------
        default_template = f"""
You are ARsemble AI, a helpful PC-building assistant.
Use the dataset below for factual accuracy and keep your answer short and clear (1–3 sentences).

Dataset:
{context_snippet}

User query: "{user_input}"
"""

        # Choose tone/template
        if is_build_query and (mentions_component or re.search(r"\b(build|recommend|estimate|recommended)\b", user_lower)):
            tone = build_template + "\nDataset:\n" + \
                context_snippet + f"\nUser query: \"{user_input}\"\n"
        elif is_specs_query:
            tone = specs_template
        elif is_attribute_query and mentions_component:
            tone = attribute_template
        elif is_educational_query:
            tone = educational_template
        else:
            tone = default_template

        # Run Gemini
        response = model.generate_content(tone)
        if response and response.text:
            clean_text = response.text.strip()
            # Clean up stray asterisks or markdown markers if any
            clean_text = clean_text.replace("*", "").replace("**", "")
            return clean_text

        return "Sorry — I couldn’t find that information in the data."
    except Exception as e:
        logger.exception("Gemini fallback failed:")
        return "Sorry! I ran into a small issue while generating that info."


# --------------------------
# Unified Handler
# --------------------------


def primary_model_response(user_input: str) -> str:
    """
    Placeholder primary model response. Replace with your primary model call.
    Currently raises NotImplementedError so the fallback path is exercised cleanly.
    """
    raise NotImplementedError("primary_model_response not implemented")


def get_ai_response(user_input: str) -> dict:
    """
    Unified AI handler with added plain-text 'bottleneck' query handling.

    - Detects PSU questions, compatibility, budget builds, component details, AND
      bottleneck questions like:
        "What is the bottleneck of Ryzen 5 5600X and RTX 3060?"
        "bottleneck between i5-10400 + GTX 1650"
        "which is the bottleneck: cpu:amd-ryzen-5-5600x and gpu:msi-rtx-3060"
    - For bottleneck queries it resolves components (using lookup_component_by_chip_id
      and _best_match_in_dataset), runs analyze_bottleneck_for_build, and returns
      plain text in `text`.
    - Falls back to existing handlers (PSU/compat/lookup/budget/Gemini).
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

        # ---------- 0) Bottleneck natural-language detection ----------
        # Common phrasings: "bottleneck", "what is the bottleneck", "bottleneck between X and Y"
        if "bottleneck" in lower_q:
            # Try to extract two components from common patterns
            # Patterns like: "bottleneck of <A> and <B>", "bottleneck between <A> and <B>", "<A> + <B>"
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
                    # attempt resolution via chip-id lookup first
                    a_obj = lookup_component_by_chip_id(a_raw) or None
                    b_obj = lookup_component_by_chip_id(b_raw) or None
                    # fallback to best-match in dataset
                    if not a_obj:
                        a_obj, _ = _best_match_in_dataset(a_raw, data)
                    if not b_obj:
                        b_obj, _ = _best_match_in_dataset(b_raw, data)
                    found_a = a_obj
                    found_b = b_obj
                    break

            # If not matched by patterns, try a simpler split on '+' or ' and '
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

            # If still not resolved, try best-match across the whole query and pick top two matches
            if not (found_a and found_b):
                # gather top matches by scanning dataset tokens
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

            # If we have at least one resolved component, proceed (if only one resolved, we still attempt)
            if found_a or found_b:
                # If one side missing, provide helpful message
                if not found_a or not found_b:
                    missing = "CPU" if not found_a else "GPU"
                    msg = f"Could not confidently identify both components from your query. Missing: {missing}. Try 'bottleneck of <cpu> and <gpu>'."
                    return {"source": "local-bottleneck", "text": msg}

                # Try to ensure we pass CPU as cpu and GPU as gpu (swap if types reversed)
                # If component types available use them; otherwise assume first is CPU if name contains 'ryzen'/'i' etc.
                a_type = None
                b_type = None
                # detect types from data entries if present
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

                # fallback heuristics by name tokens if types not present
                if not cpu_obj or not gpu_obj:
                    # assign by token cues
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

                # Final fallback: assume found_a is CPU and found_b is GPU
                if not cpu_obj:
                    cpu_obj = found_a
                if not gpu_obj:
                    gpu_obj = found_b

                # Use the wrapper analyzer (resolve dicts)
                try:
                    # prefer the wrapper if available
                    analyzer = globals().get("analyze_bottleneck_for_build") or analyze_bottleneck_text
                    if analyzer == analyze_bottleneck_text:
                        # we must pass dicts; ensure these are dicts
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
                    # fall through to other handlers

        # ---------- 1) PSU / power-supply natural language detection (HIGH PRIORITY) ----------
        if re.search(r"\b(psu|power supply|power-supply|power supply i need|psu do i need|psu needed|what power supply|how much power)\b", lower_q):
            try:
                psu_resp = recommend_psu_for_query_with_chips(
                    user_input, data, headroom_percent=30)
                if psu_resp and not psu_resp.get("error"):
                    return {
                        "source": "local-psu",
                        "type": "psu_recommendation",
                        **psu_resp
                    }
            except Exception:
                logger.exception(
                    "PSU recommendation failed; falling through to other handlers.")

        # ---------- 2) Compatibility question detection (next priority) ----------
        if "compatible" in lower_q or "works with" in lower_q:
            try:
                compat = get_compatible_components(user_input, data)
                if compat and compat.get("found"):
                    return {
                        "source": "local-compatibility",
                        "type": "compatibility",
                        "target": compat["target"],
                        "compatible_type": compat["compatible_type"],
                        "results": compat["chips"]
                    }
                else:
                    return {
                        "source": "local-compatibility",
                        "type": "compatibility",
                        "message": compat.get("message", "No compatible parts found.") if compat else "No compatible parts found."
                    }
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

        # ---------- 3.5) Local component DETAILS/specs lookup ----------
        if re.search(r"\b(details|detail|specs|specification|specifications|show me|what are the specs|tell me about)\b", lower_q):
            try:
                logger.info(
                    "Attempting local dataset lookup for specs/details query: %s", user_input)
                details = get_component_details(user_input)
                if details.get("found"):
                    return {
                        "source": "local-lookup",
                        "type": "component_details",
                        "found": True,
                        "message": f"Found {details.get('name')}",
                        "component": {
                            "name": details.get("name"),
                            "type": details.get("type"),
                            "price": details.get("price"),
                            "specs": details.get("specs")
                        },
                        "recommendations": []
                    }
                else:
                    return {
                        "source": "local-lookup",
                        "type": "component_not_found",
                        "message": details.get("error", "No match found in dataset."),
                        "suggestions": details.get("suggestions", [])
                    }
            except Exception:
                logger.exception(
                    "Local specs lookup failed; falling through to Gemini.")

        # ---------- 4) Fallback to Gemini ----------
        gemini_text = gemini_fallback_with_data(user_input, data)
        return {"source": "gemini-fallback", "text": gemini_text, "used_fallback": True}

    except Exception as e:
        logger.exception("get_ai_response failed unexpectedly:")
        return {"source": "error", "text": "Internal error processing request."}


def lookup_component_by_chip_id(chip_id_or_slug: str):
    """
    Accepts:
      - 'cpu:amd-ryzen-5-3600'
      - 'amd-ryzen-5-3600'
      - 'AMD Ryzen 5 3600'
    Returns the component dict or None.
    """
    if not chip_id_or_slug:
        return None

    # normalize
    text = chip_id_or_slug.strip()
    # If format includes type prefix 'cpu:slug', split it
    if ":" in text:
        comp_type, slug = text.split(":", 1)
        slug = slugify(slug)
        comp_type = comp_type.lower()
        # try direct lookup in that type
        for k, v in data.get(comp_type, {}).items():
            if slugify(v.get("name", "")) == slug:
                return v
        # not found in that type — fall back to full search
    # otherwise treat the whole thing as slug or name
    slug = slugify(text)
    # search all categories for slug match
    for comp_type, bucket in data.items():
        for k, v in bucket.items():
            if slugify(v.get("name", "")) == slug:
                return v
    # fuzzy fallback: substring match on name (case-insensitive)
    lower = text.lower()
    for comp_type, bucket in data.items():
        for k, v in bucket.items():
            if lower in (v.get("name", "") or "").lower():
                return v
    return None


def get_component_details(raw: str):
    """
    Improved flexible lookup for any component (cpu, gpu, motherboard, ram, storage, psu).
    - Tries direct chip-id / slug lookup first (lookup_component_by_chip_id).
    - Uses _best_match_in_dataset for robust fuzzy matching.
    - If no confident match, returns 'suggestions' (top close matches) to help the user.
    - Always returns a dict; useful fields:
        - found: bool
        - name, type, price, specs  (if found)
        - suggestions: list of {"name","type","score"} (if not found or low confidence)
        - debug: optional diagnostic string for logs
    """
    try:
        if not raw or not isinstance(raw, str):
            return {"found": False, "error": "No component name provided."}

        query_raw = raw.strip()
        query_norm = _normalize_text_for_match(query_raw)
        q_slug = slugify(query_raw)

        logger.debug("LOOKUP START: query_raw=%r, q_slug=%r, q_norm=%r",
                     query_raw, q_slug, query_norm)

        # 1) direct chip-id or slug-aware lookup (supports "cpu:amd-ryzen-5-3600" and plain names)
        direct = lookup_component_by_chip_id(query_raw)
        if direct:
            logger.debug("LOOKUP: direct lookup_component_by_chip_id -> %r (%s)",
                         direct.get("name"), direct.get("type"))
            return {
                "found": True,
                "name": direct.get("name"),
                "type": direct.get("type"),
                "price": direct.get("price", ""),
                "specs": {k: v for k, v in direct.items() if k not in ("name", "price", "type")},
                "debug": "direct lookup_component_by_chip_id"
            }

        # 2) use your robust fuzzy matcher which checks many heuristics
        best_obj, best_type = _best_match_in_dataset(query_raw, data)
        if best_obj and best_type:
            logger.debug("LOOKUP: _best_match_in_dataset -> %s (%s)",
                         best_obj.get("name"), best_type)
            return {
                "found": True,
                "name": best_obj.get("name"),
                "type": best_type,
                "price": best_obj.get("price", ""),
                "specs": {k: v for k, v in best_obj.items() if k not in ("name", "price")},
                "debug": "best_match_in_dataset"
            }

        # 3) fallback fuzzy ranking: compute simple token-overlap scores across dataset to produce suggestions
        q_tokens = set([t for t in re.split(r"\s+", query_norm) if t])
        scores = []
        for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu"):
            for comp in (data.get(ctype, {}) or {}).values():
                name = (comp.get("name") or "")
                if not name:
                    continue
                name_norm = _normalize_text_for_match(name)
                name_tokens = set(
                    [t for t in re.split(r"\s+", name_norm) if t])
                # token overlap
                overlap = len(q_tokens & name_tokens)
                # numeric match bonus (e.g., "500gb", "970")
                numeric_bonus = sum(2 for t in name_tokens if re.search(
                    r"\d", t) and t in q_tokens)
                score = overlap + numeric_bonus
                # small boost if query is substring of component name
                if query_norm and query_norm in name_norm:
                    score += 1
                if score > 0:
                    scores.append((score, ctype, name))
        # sort suggestions by score desc
        suggestions = []
        if scores:
            scores_sorted = sorted(scores, key=lambda x: -x[0])[:8]
            seen = set()
            for sc, ctype, name in scores_sorted:
                key = (ctype, name.lower())
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(
                    {"name": name, "type": ctype, "score": int(sc)})
        # 4) final response
        if suggestions:
            logger.debug("LOOKUP: suggestions -> %s", suggestions[:5])
            return {
                "found": False,
                "error": f"I couldn't find an exact match for '{raw}', but here are close matches.",
                "suggestions": suggestions,
                "debug": "suggestions returned"
            }

        logger.debug(
            "LOOKUP: no match found for %r in any category.", query_raw)
        return {"found": False, "error": f"I cannot find '{raw}' in my database.", "debug": "no matches at all"}
    except Exception as e:
        logger.exception("get_component_details failed:")
        return {"found": False, "error": "Lookup failed due to an internal error.", "exception": str(e)}


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
        found = {"cpu": [], "gpu": [],
                 "motherboard": [], "ram": [], "storage": []}

        # 1) direct name/slug matches across categories
        for ctype in ("cpu", "gpu", "motherboard", "ram", "storage", "psu"):
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
    Uses direct fields first (power/tdp/wattage), else simple name/type heuristics.
    """
    if not comp or not isinstance(comp, dict):
        return 0

    # direct keys to check
    for key in ("power", "tdp", "wattage", "tdp_w", "power_draw", "w"):
        if key in comp and comp.get(key) is not None:
            v = _parse_watt_value(comp.get(key))
            if v:
                return v

    name = (comp.get("name") or "").lower()
    ctype = (comp.get("type") or "").lower()

    # GPU heuristics (conservative)
    if ctype == "gpu" or "gpu" in name or name.startswith("rtx") or name.startswith("gtx") or "radeon" in name:
        if "3060" in name or "rx 6600" in name:
            return 170
        if "3070" in name or "6700" in name:
            return 220
        if "3080" in name or "6800" in name:
            return 320
        if "3090" in name or "6900" in name:
            return 350
        return 180

    # CPU heuristics
    if ctype == "cpu" or "ryzen" in name or re.search(r"\bi\d\b|\bi3\b|\bi5\b|\bi7\b|\bi9\b", name):
        cores_field = comp.get("cores") or ""
        if isinstance(cores_field, str):
            m = re.search(r"(\d+)", cores_field)
            if m:
                cores = int(m.group(1))
                est = cores * 8 + 35
                return max(35, min(140, est))
        return 65

    # Motherboard
    if ctype == "motherboard" or any(k in name for k in ("mobo", "prime", "b450", "b550", "x570")):
        return 50

    # RAM
    if ctype == "ram" or "ddr" in name:
        return 8

    # Storage
    if ctype == "storage" or "ssd" in name or "nvme" in name:
        return 5
    if "hdd" in name:
        return 8

    # fans/peripherals
    if "fan" in name or "rgb" in name:
        return 4

    # fallback
    return 10


def compute_total_wattage_from_build(build: dict, include_headroom: bool = True, headroom_percent: int = 30) -> dict:
    """
    Compute components' watt estimates, total draw, and recommended PSU (rounded to 50W).
    Returns dict with keys:
      - component_watts
      - total_draw
      - recommended_psu
      - headroom_percent
    """
    comp_watts = {}
    total = 0

    # storage may be list
    storages = []
    if isinstance(build.get("storage"), list):
        storages = build.get("storage")
    elif build.get("storage"):
        storages = [build.get("storage")]

    # list of components to check
    keys = ["cpu", "gpu", "motherboard", "ram", "storage", "psu"]
    for k in keys:
        if k == "storage":
            s_total = 0
            for s in storages:
                s_total += estimate_component_wattage(s)
            comp_watts["storage"] = s_total
            total += s_total
            continue
        comp = build.get(k)
        # if component is PSU, do not include its watt as load
        if k == "psu":
            comp_watts["psu"] = 0
            continue
        w = estimate_component_wattage(comp)
        comp_watts[k] = w
        total += w

    # extras
    extras = 15
    comp_watts["extras"] = extras
    total += extras

    recommended = None
    if include_headroom:
        target = total * (1 + headroom_percent / 100.0)
        # round up to nearest 50
        recommended = int((target // 50) * 50)
        if recommended < target:
            recommended += 50

    return {
        "component_watts": comp_watts,
        "total_draw": int(total),
        "recommended_psu": recommended,
        "headroom_percent": headroom_percent,
    }


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
