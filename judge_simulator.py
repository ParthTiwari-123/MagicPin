#!/usr/bin/env python3
"""
magicpin AI Challenge — LLM-Powered Judge Simulator (Gemini Edition)
====================================================================

A strict but fair judge that scores your bot and explains WHY.
"""

# =============================================================================
# ██████  CONFIGURATION - EDIT THIS SECTION ██████
# =============================================================================

# Your bot's URL (where your bot is running)
BOT_URL = "http://localhost:8080"

# Choose your LLM provider: "gemini" (configured below)
LLM_PROVIDER = "gemini"

# Your Gemini API key (or set environment variable GEMINI_API_KEY)
LLM_API_KEY = ""  # <-- Or leave empty if using $env:GEMINI_API_KEY

# Model to use
LLM_MODEL = "gemini-3.5-flash-lite"

# Which test to run by default
TEST_SCENARIO = "all"

# =============================================================================
# ██████  END OF CONFIGURATION - DON'T EDIT BELOW THIS LINE ██████
# =============================================================================

import os
import sys
import json
import time
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from urllib import request as urlrequest, error as urlerror
from abc import ABC, abstractmethod

# Import official Google GenAI SDK
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    print("Error: 'google-genai' package is not installed.")
    print("Run: pip install google-genai")
    sys.exit(1)

# Constants
TIMEOUT_LLM = 45
DATASET_DIR = Path(__file__).parent / "dataset"

# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- {text} ---{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {text}")

def print_fail(text: str):
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {text}")

def print_warn(text: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")

def print_info(text: str):
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")

def print_llm(text: str):
    print(f"{Colors.MAGENTA}[LLM]{Colors.RESET} {text}")

def print_score_bar(dimension: str, score: int, max_score: int = 10):
    bar_filled = int((score / max_score) * 20)
    bar_empty = 20 - bar_filled
    color = Colors.GREEN if score >= 7 else Colors.YELLOW if score >= 4 else Colors.RED
    print(f"  {dimension:22} [{color}{'█' * bar_filled}{Colors.DIM}{'░' * bar_empty}{Colors.RESET}] {color}{score:2}/{max_score}{Colors.RESET}")

def print_reason(text: str):
    wrapped = text[:200] + "..." if len(text) > 200 else text
    print(f"    {Colors.DIM}{wrapped}{Colors.RESET}")

def print_hint(hint: str):
    print(f"\n  {Colors.YELLOW}Hint:{Colors.RESET} {hint}")

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScoreResult:
    specificity: int = 0
    specificity_reason: str = ""
    category_fit: int = 0
    category_fit_reason: str = ""
    merchant_fit: int = 0
    merchant_fit_reason: str = ""
    decision_quality: int = 0
    decision_quality_reason: str = ""
    engagement_compulsion: int = 0
    engagement_reason: str = ""
    penalties: int = 0
    penalty_reasons: List[str] = field(default_factory=list)
    hint: str = ""

    @property
    def total(self) -> int:
        return max(0, self.specificity + self.category_fit + self.merchant_fit +
                   self.decision_quality + self.engagement_compulsion - self.penalties)

# =============================================================================
# LLM PROVIDERS (GEMINI)
# =============================================================================

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = None) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str = "", model: str = ""):
        self.model = model or "gemini-1.5-flash"
        # Initialize the official google-genai client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Will automatically look for GEMINI_API_KEY environment variable
            self.client = genai.Client()

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def complete(self, prompt: str, system: str = None) -> str:
        # Prepend system prompt if provided, as generate_content accepts configuration/prompts
        full_prompt = f"System Instructions:\n{system}\n\nTask:\n{prompt}" if system else prompt
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        return response.text


def create_provider() -> LLMProvider:
    """Create LLM provider (Gemini)."""
    # If key is provided in script config, set it in environment for the SDK
    if LLM_API_KEY:
        os.environ["GEMINI_API_KEY"] = LLM_API_KEY
        
    return GeminiProvider(api_key=LLM_API_KEY, model=LLM_MODEL)

# =============================================================================
# DATASET & BOT CLIENT
# =============================================================================

class DatasetLoader:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.categories = {}
        self.merchants = {}
        self.customers = {}
        self.triggers = {}

    def load(self) -> bool:
        try:
            cat_dir = self.dataset_dir / "categories"
            if cat_dir.exists():
                for f in cat_dir.glob("*.json"):
                    data = json.load(open(f, encoding="utf-8"))
                    self.categories[data.get("slug", f.stem)] = data

            for name, container, key in [
                ("merchants_seed.json", "merchants", "merchant_id"),
                ("customers_seed.json", "customers", "customer_id"),
                ("triggers_seed.json", "triggers", "id")
            ]:
                path = self.dataset_dir / name
                if path.exists():
                    data = json.load(open(path, encoding="utf-8"))
                    items = data.get(container, data.get(container.rstrip("s"), []))
                    storage = getattr(self, container)
                    for item in items:
                        if key in item:
                            storage[item[key]] = item
            return True
        except Exception as e:
            print_fail(f"Dataset load error: {e}")
            return False


class BotClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, timeout: int = 30,
                 body_dict: Dict = None) -> Tuple[Optional[Dict], Optional[str], float]:
        url = f"{self.base_url}{path}"
        start = time.time()
        body = json.dumps(body_dict).encode("utf-8") if body_dict else None
        headers = {"Content-Type": "application/json"}
        req = urlrequest.Request(url, data=body, method=method, headers=headers)

        try:
            resp = urlrequest.urlopen(req, timeout=timeout)
            return json.loads(resp.read().decode("utf-8")), None, (time.time() - start) * 1000
        except urlerror.HTTPError as e:
            latency = (time.time() - start) * 1000
            if e.code == 401:
                return None, "Unauthorized", latency
            try:
                return json.loads(e.read().decode("utf-8")), None, latency
            except:
                return None, f"HTTP {e.code}", latency
        except Exception as e:
            return None, str(e), (time.time() - start) * 1000

    def healthz(self):
        return self._request("GET", "/v1/healthz", 5)

    def metadata(self):
        return self._request("GET", "/v1/metadata", 5)

    def push_context(self, scope, cid, version, payload):
        return self._request("POST", "/v1/context", 10, {
            "scope": scope, "context_id": cid, "version": version,
            "payload": payload, "delivered_at": datetime.utcnow().isoformat() + "Z"
        })

    def tick(self, triggers):
        return self._request("POST", "/v1/tick", 15, {
            "now": datetime.utcnow().isoformat() + "Z", "available_triggers": triggers
        })

    def reply(self, conv_id, merchant_id, message, turn):
        return self._request("POST", "/v1/reply", 15, {
            "conversation_id": conv_id, "merchant_id": merchant_id, "customer_id": None,
            "from_role": "merchant", "message": message,
            "received_at": datetime.utcnow().isoformat() + "Z", "turn_number": turn
        })

# =============================================================================
# LLM SCORING ENGINE
# =============================================================================

class LLMScorer:
    SYSTEM = """You are a STRICT judge for the magicpin AI Challenge. You score merchant engagement messages.

SCORING DIMENSIONS (0-10 each, be strict - 5 is average, 7+ is good, 9+ is excellent):

1. SPECIFICITY: Does the message have VERIFIABLE facts? (Numbers, dates, metrics)
2. CATEGORY FIT: Does the voice match the business type?
3. MERCHANT FIT: Is it personalized to THIS merchant?
4. TRIGGER RELEVANCE: Does it connect to WHY NOW?
5. ENGAGEMENT COMPULSION: Would they reply?

RESPOND ONLY WITH VALID JSON FORMAT IN THIS EXACT STRUCTURE:
{
  "specificity": <0-10>,
  "specificity_reason": "<why this score>",
  "category_fit": <0-10>,
  "category_fit_reason": "<why this score>",
  "merchant_fit": <0-10>,
  "merchant_fit_reason": "<why this score>",
  "decision_quality": <0-10>,
  "decision_quality_reason": "<why this score>",
  "engagement_compulsion": <0-10>,
  "engagement_reason": "<why this score>",
  "hint": "<one sentence guidance for improvement>"
}"""

    def __init__(self, llm: LLMProvider, dataset: DatasetLoader):
        self.llm = llm
        self.dataset = dataset

    def score(self, action: Dict, category: Dict, merchant: Dict,
              trigger: Dict, customer: Dict = None) -> ScoreResult:
        body = action.get("body", "")
        prompt = f"""SCORE THIS MESSAGE:
Category: {category.get('slug', 'unknown')}
Merchant: {merchant.get('identity', {}).get('name', 'unknown')}
Trigger Kind: {trigger.get('kind', 'unknown')}
Message Body: "{body}"
Score each dimension 0-10 with clear reasoning. Be STRICT."""

        try:
            print_llm("Analyzing message with Gemini...")
            response = self.llm.complete(prompt, self.SYSTEM)
            return self._parse_response(response, action)
        except Exception as e:
            print_warn(f"LLM error: {e}")
            return self._fallback_score(action)

    def _parse_response(self, response: str, action: Dict) -> ScoreResult:
        match = re.search(r'\{[\s\S]*\}', response)
        if not match:
            return self._fallback_score(action)
        try:
            data = json.loads(match.group())
            return ScoreResult(
                specificity=min(10, max(0, int(data.get("specificity", 5)))),
                specificity_reason=data.get("specificity_reason", ""),
                category_fit=min(10, max(0, int(data.get("category_fit", 5)))),
                category_fit_reason=data.get("category_fit_reason", ""),
                merchant_fit=min(10, max(0, int(data.get("merchant_fit", 5)))),
                merchant_fit_reason=data.get("merchant_fit_reason", ""),
                decision_quality=min(10, max(0, int(data.get("decision_quality", 5)))),
                decision_quality_reason=data.get("decision_quality_reason", ""),
                engagement_compulsion=min(10, max(0, int(data.get("engagement_compulsion", 5)))),
                engagement_reason=data.get("engagement_reason", ""),
                hint=data.get("hint", "")
            )
        except:
            return self._fallback_score(action)

    def _fallback_score(self, action: Dict) -> ScoreResult:
        return ScoreResult(specificity=5, category_fit=5, merchant_fit=5,
                           decision_quality=5, engagement_compulsion=5)

# =============================================================================
# MAIN JUDGE
# =============================================================================

class JudgeSimulator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.client = BotClient(BOT_URL)
        self.dataset = DatasetLoader(DATASET_DIR)
        self.scorer: Optional[LLMScorer] = None
        self.all_scores: List[ScoreResult] = []

    def run(self, scenario: str) -> bool:
        print_header(f"LLM JUDGE — {scenario.upper()}")
        if not self.dataset.load():
            print_fail("Dataset load failed")
            return False

        self.scorer = LLMScorer(self.llm, self.dataset)
        scenarios = {
            "warmup": self._warmup,
            "phase2_short": self._phase2_short,
            "auto_reply_hell": self._auto_reply,
            "intent_transition": self._intent,
            "hostile": self._hostile,
            "all": self._all,
            "full_evaluation": self._full,
        }
        success = scenarios.get(scenario, lambda: False)()
        self._final_summary()
        return success

    def _warmup(self) -> bool:
        print_section("WARMUP")
        data, err, lat = self.client.healthz()
        if err:
            print_fail(f"healthz: {err}")
            return False
        print_success(f"healthz ({lat:.0f}ms)")
        for slug, cat in self.dataset.categories.items():
            self.client.push_context("category", slug, 1, cat)
        for mid, m in list(self.dataset.merchants.items())[:5]:
            self.client.push_context("merchant", mid, 1, m)
        return True

    def _phase2_short(self) -> bool:
        if not self._warmup(): return False
        print_section("TICK TEST")
        trigs = list(self.dataset.triggers.keys())[:3]
        data, err, lat = self.client.tick(trigs)
        for action in data.get("actions", []):
            self._score_and_display(action)
        return True

    def _auto_reply(self) -> bool:
        print_section("AUTO-REPLY")
        mid = list(self.dataset.merchants.keys())[0] if self.dataset.merchants else "m_test"
        self.client.reply("conv_auto_1", mid, "Thank you for contacting us!", 2)
        print_success("Auto-reply test complete")
        return True

    def _intent(self) -> bool:
        print_section("INTENT TRANSITION")
        mid = list(self.dataset.merchants.keys())[0] if self.dataset.merchants else "m_test"
        self.client.reply("conv_intent_1", mid, "Ok lets do it. Whats next?", 2)
        print_success("Intent test complete")
        return True

    def _hostile(self) -> bool:
        print_section("HOSTILE HANDLING")
        mid = list(self.dataset.merchants.keys())[0] if self.dataset.merchants else "m_test"
        self.client.reply("conv_hostile", mid, "Stop messaging me. This is spam.", 2)
        print_success("Hostile test complete")
        return True

    def _all(self) -> bool:
        for name, fn in [("warmup", self._warmup), ("auto_reply", self._auto_reply), 
                         ("intent", self._intent), ("hostile", self._hostile)]:
            try: 
                fn()
            except Exception as e: 
                print_fail(f"{name} failed: {e}")
        return True

    def _full(self) -> bool:
        return self._warmup()

    def _score_and_display(self, action: Dict, verbose: bool = True):
        tid, mid = action.get("trigger_id", ""), action.get("merchant_id", "")
        score = self.scorer.score(action, {}, self.dataset.merchants.get(mid, {}), self.dataset.triggers.get(tid, {}))
        self.all_scores.append(score)
        print_score_bar("Total Score", score.total, 50)

    def _final_summary(self):
        if not self.all_scores: return
        print_section("FINAL SUMMARY")
        print_success(f"Evaluated {len(self.all_scores)} actions successfully.")

def main():
    print_header("magicpin AI Challenge — Gemini LLM Judge")
    try:
        llm = create_provider()
        print_info(f"LLM Provider: {llm.name()}")
    except Exception as e:
        print_fail(f"Failed to create Gemini provider: {e}")
        sys.exit(1)

    print_info("Testing LLM connection...")
    try:
        if llm.complete("Say 'ready' if you can hear me.", "Test assistant."):
            print_success("LLM connected successfully")
        else:
            sys.exit(1)
    except Exception as e:
        print_fail(f"LLM connection failed: {e}")
        sys.exit(1)

    judge = JudgeSimulator(llm)
    sys.exit(0 if judge.run(TEST_SCENARIO) else 1)

if __name__ == "__main__":
    main()