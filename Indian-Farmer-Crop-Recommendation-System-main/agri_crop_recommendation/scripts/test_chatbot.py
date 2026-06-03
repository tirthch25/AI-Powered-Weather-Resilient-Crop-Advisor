"""
Chatbot Random-Question Tester
==============================
Tests the LLM chat service with 12 diverse farming questions and evaluates
whether each answer is meaningful and relevant.

Run from the agri_crop_recommendation directory:
    python scripts/test_chatbot.py
"""

# ---- Force UTF-8 stdout so Windows cp1252 never crashes ----
import io
import sys as _sys
_sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys
import os
import time
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.services.llm_chat import answer_farmer_question

# ------------------------------------------------------------------ #
# 12 random, diverse farming questions                                 #
# ------------------------------------------------------------------ #
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Which crops are best to sow in June in Maharashtra?",
        "region_id": "MH_PUNE",
        "season": "Kharif",
        "expected_keywords": ["kharif", "soybean", "cotton", "rice", "jowar", "bajra", "june"],
        "category": "Crop Selection",
    },
    {
        "id": 2,
        "question": "How much water does rice need per acre?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["water", "irrigation", "mm", "rice"],
        "category": "Irrigation",
    },
    {
        "id": 3,
        "question": "What is the best fertilizer for wheat crop?",
        "region_id": "UP_LUCKNOW",
        "season": "Rabi",
        "expected_keywords": ["urea", "nitrogen", "fertilizer", "wheat"],
        "category": "Fertilizer",
    },
    {
        "id": 4,
        "question": "How do I control aphids on mustard plants?",
        "region_id": "",
        "season": "Rabi",
        "expected_keywords": ["aphid", "insecticide", "spray", "neem", "pest"],
        "category": "Pest Control",
    },
    {
        "id": 5,
        "question": "What government schemes are available for small farmers in India?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["pm-kisan", "kisan", "scheme", "subsidy", "government"],
        "category": "Govt Schemes",
    },
    {
        "id": 6,
        "question": "What is the ideal soil pH for tomato cultivation?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["ph", "6", "soil", "tomato"],
        "category": "Soil",
    },
    {
        "id": 7,
        "question": "How can I improve the organic matter in my sandy soil?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["compost", "manure", "organic", "sandy"],
        "category": "Soil Health",
    },
    {
        "id": 8,
        "question": "When should I harvest sugarcane?",
        "region_id": "MH_KOLHAPUR",
        "season": "",
        "expected_keywords": ["harvest", "month", "sugarcane", "maturity"],
        "category": "Harvesting",
    },
    {
        "id": 9,
        "question": "What are the signs of nitrogen deficiency in plants?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["yellow", "nitrogen", "leaf", "deficiency"],
        "category": "Crop Health",
    },
    {
        "id": 10,
        "question": "How do I store onions after harvest to prevent rotting?",
        "region_id": "",
        "season": "",
        "expected_keywords": ["store", "dry", "onion", "humidity"],
        "category": "Post-Harvest",
    },
    {
        "id": 11,
        "question": "What is drip irrigation and is it suitable for cotton?",
        "region_id": "GJ_SURAT",
        "season": "Kharif",
        "expected_keywords": ["drip", "irrigation", "cotton", "water"],
        "category": "Irrigation Tech",
    },
    {
        "id": 12,
        "question": "What is the current temperature in Pune today?",
        "region_id": "MH_PUNE",
        "season": "",
        "expected_keywords": ["temperature", "high", "low", "today", "pune"],
        "category": "Weather",
    },
]


# ------------------------------------------------------------------ #
# Evaluation helpers                                                   #
# ------------------------------------------------------------------ #

def evaluate_answer(answer, expected_keywords):
    lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in lower]
    is_error = any(phrase in lower for phrase in [
        "api key", "not available", "trouble connecting", "unavailable"
    ])
    passed = (len(found) / len(expected_keywords) >= 0.4) and not is_error
    return passed, found, missing


def divider(char="-", width=70):
    print(char * width)


def colored(text, code):
    print(f"\033[{code}m{text}\033[0m")


def wrap_answer(text, width=66, indent="    "):
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line.strip(), width=width,
                                         initial_indent=indent,
                                         subsequent_indent=indent))
        else:
            wrapped.append(indent + line.strip())
    return "\n".join(wrapped[:18])


# ------------------------------------------------------------------ #
# Main test runner                                                     #
# ------------------------------------------------------------------ #

def run_tests():
    print()
    colored("=" * 70, "96")
    colored("  [CROP-AI] Indian Farmer Chatbot -- Random Question Tester", "96")
    colored("  Testing 12 diverse farming questions via Gemini LLM", "96")
    colored("=" * 70, "96")
    print()

    results = []

    for test in TEST_QUESTIONS:
        divider()
        qid = test["id"]
        cat = test["category"]
        q   = test["question"]
        rid = test["region_id"]
        sea = test["season"]

        colored(f"  Q{qid:02d} [{cat}]", "93")
        print(f"  >> {q}")
        if rid:
            print(f"     Region: {rid}  |  Season: {sea or 'auto'}")
        print()

        t0 = time.time()
        try:
            answer, _ = answer_farmer_question(
                question=q,
                region_id=rid,
                region_name="",
                season=sea,
                history=[],
                crop_context="",
                state_name="",
                climate_zone="",
                soil_info="",
                weather_summary="",
            )
            elapsed = round(time.time() - t0, 2)
            passed, found_kws, missing_kws = evaluate_answer(answer, test["expected_keywords"])

            colored("  Answer:", "97")
            print(wrap_answer(answer))
            print()

            kw_ratio = f"{len(found_kws)}/{len(test['expected_keywords'])}"
            if passed:
                colored(f"  [PASS]  Keywords: {kw_ratio}  |  Time: {elapsed}s", "92")
            else:
                colored(f"  [FAIL]  Keywords: {kw_ratio}  |  Time: {elapsed}s", "91")
                if missing_kws:
                    colored(f"     Missing: {', '.join(missing_kws[:5])}", "91")

            results.append({
                "id": qid, "category": cat, "question": q, "answer": answer,
                "passed": passed, "found_kws": found_kws,
                "missing_kws": missing_kws, "elapsed": elapsed,
            })

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            colored(f"  [ERROR] {exc}", "91")
            results.append({
                "id": qid, "category": cat, "question": q,
                "answer": f"ERROR: {exc}", "passed": False,
                "found_kws": [], "missing_kws": test["expected_keywords"],
                "elapsed": elapsed,
            })

        print()
        time.sleep(1.5)   # avoid rate-limit

    # -------- Summary --------
    divider("=")
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    avg_t   = round(sum(r["elapsed"] for r in results) / total, 2)

    colored("\n  FINAL REPORT", "96")
    divider()
    print(f"  Total Questions : {total}")
    colored(f"  Passed         : {passed}", "92")
    colored(f"  Failed         : {failed}", "91" if failed > 0 else "92")
    print(f"  Avg Response   : {avg_t}s\n")

    print(f"  {'ID':<4} {'Category':<16} {'Result':<8} {'Keywords':<12} Time")
    divider("-", 60)
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        kw = f"{len(r['found_kws'])}/{len(r['found_kws'])+len(r['missing_kws'])}"
        col = "92" if r["passed"] else "91"
        colored(f"  Q{r['id']:<3} {r['category']:<16} {status:<8} {kw:<12} {r['elapsed']}s", col)

    divider("=")
    if passed == total:
        colored("\n  All questions passed! Chatbot is working great.\n", "92")
    elif passed >= total * 0.75:
        colored(f"\n  {passed}/{total} passed -- Chatbot is mostly accurate.\n", "93")
    else:
        colored(f"\n  Only {passed}/{total} passed -- Review failed questions.\n", "91")

    failed_list = [r for r in results if not r["passed"]]
    if failed_list:
        colored("  Failed Questions:", "91")
        for r in failed_list:
            colored(f"     Q{r['id']:02d} [{r['category']}] -- {r['question'][:60]}", "91")
            if r["missing_kws"]:
                colored(f"          Missing keywords: {', '.join(r['missing_kws'])}", "90")
    print()


if __name__ == "__main__":
    run_tests()
