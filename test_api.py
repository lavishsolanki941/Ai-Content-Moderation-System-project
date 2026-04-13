"""
test_api.py
-----------
Tests for the Content Moderation API.

Two modes:
  1. Direct function tests — tests predict_toxicity() without running the server
  2. HTTP tests — tests the FastAPI endpoints using TestClient (no server needed)

Run:
    python test_api.py
"""

import sys
import json

# ── Colour helpers for terminal output ─────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg): print(f"  {BLUE}INFO{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "-"*50)
# ───────────────────────────────────────────────────────────────────────────


TEST_CASES = [
    {
        "name": "Clean comment",
        "text": "I really enjoyed reading your article. Very insightful!",
        "expect_toxic": False,
        "expect_severity": ["low", "moderate"],
    },
    {
        "name": "Clearly toxic",
        "text": "You are a worthless idiot and I hope you die",
        "expect_toxic": True,
        "expect_severity": ["high", "critical"],
    },
    {
        "name": "Threat",
        "text": "I will find you and make you pay for this",
        "expect_toxic": True,
        "expect_severity": ["moderate", "high", "critical"],
    },
    {
        "name": "Compliment",
        "text": "Great work on this project! You should be proud.",
        "expect_toxic": False,
        "expect_severity": ["low"],
    },
    {
        "name": "Identity hate",
        "text": "People like you don't deserve to exist in this country",
        "expect_toxic": True,
        "expect_severity": ["moderate", "high", "critical"],
    },
    {
        "name": "Sarcastic but clean",
        "text": "Oh sure, that's totally the best idea I've ever heard",
        "expect_toxic": False,
        "expect_severity": ["low", "moderate"],
    },
    {
        "name": "Technical discussion",
        "text": "The algorithm complexity is O(n log n) using merge sort",
        "expect_toxic": False,
        "expect_severity": ["low"],
    },
    {
        "name": "Mixed — frustrated but not abusive",
        "text": "This is so frustrating, why can't anyone get this right?",
        "expect_toxic": False,
        "expect_severity": ["low", "moderate"],
    },
]


def run_direct_tests():
    """Test predict_toxicity() function directly."""
    header("DIRECT FUNCTION TESTS (predict_toxicity)")

    try:
        from predict import predict_toxicity
    except Exception as e:
        print(f"{RED}Could not import predict module: {e}{RESET}")
        print("Make sure you have trained the model first (python train.py)")
        return

    passed = 0
    failed = 0

    for case in TEST_CASES:
        print(f"\n  Testing: \"{case['name']}\"")
        info(f"Input: {case['text'][:60]}...")

        try:
            result = predict_toxicity(case["text"])

            # Check structure
            assert "is_toxic" in result
            assert "severity_level" in result
            assert "severity_score" in result
            assert "breakdown" in result
            assert "flagged_labels" in result

            # Check score range
            assert 0.0 <= result["severity_score"] <= 1.0

            # Check toxicity expectation
            if result["is_toxic"] == case["expect_toxic"]:
                ok(f"is_toxic={result['is_toxic']} (expected {case['expect_toxic']})")
                passed += 1
            else:
                fail(f"is_toxic={result['is_toxic']} (expected {case['expect_toxic']})")
                failed += 1

            # Show breakdown
            info(f"Severity: {result['severity_level']} ({result['severity_score']:.3f})")
            info(f"Flagged:  {result['flagged_labels'] or 'none'}")

        except RuntimeError as e:
            fail(f"RuntimeError: {e}")
            print(f"  {YELLOW}Note: Train the model first with python train.py{RESET}")
            failed += 1
        except Exception as e:
            fail(f"Unexpected error: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")


def run_api_tests():
    """Test FastAPI endpoints using TestClient."""
    header("API ENDPOINT TESTS (FastAPI TestClient)")

    try:
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
    except ImportError:
        print(f"{YELLOW}httpx not installed. Run: pip install httpx{RESET}")
        return
    except Exception as e:
        print(f"{RED}Could not start test client: {e}{RESET}")
        return

    passed = 0
    failed = 0

    # Test 1: Root endpoint
    print("\n  Testing: GET /")
    r = client.get("/")
    if r.status_code == 200 and r.json()["status"] == "running":
        ok("GET / returns 200 with status=running")
        passed += 1
    else:
        fail(f"GET / returned {r.status_code}")
        failed += 1

    # Test 2: Health endpoint
    print("\n  Testing: GET /health")
    r = client.get("/health")
    if r.status_code == 200:
        ok(f"GET /health returns 200 — {r.json()['message']}")
        passed += 1
    else:
        fail(f"GET /health returned {r.status_code}")
        failed += 1

    # Test 3: Single moderation — clean text
    print("\n  Testing: POST /moderate (clean text)")
    r = client.post("/moderate", json={"text": "Have a great day!"})
    if r.status_code == 200:
        data = r.json()
        if not data["is_toxic"] and "breakdown" in data:
            ok(f"Clean text correctly classified — severity={data['severity_level']}")
            passed += 1
        else:
            fail(f"Clean text marked as toxic: {data}")
            failed += 1
    else:
        fail(f"POST /moderate returned {r.status_code}: {r.text}")
        failed += 1

    # Test 4: Single moderation — toxic text
    print("\n  Testing: POST /moderate (toxic text)")
    r = client.post("/moderate", json={"text": "I hate you, you are worthless garbage"})
    if r.status_code == 200:
        data = r.json()
        info(f"Severity={data['severity_level']}, score={data['severity_score']}")
        info(f"Flagged: {data['flagged_labels']}")
        ok("Toxic text processed without error")
        passed += 1
    else:
        fail(f"POST /moderate returned {r.status_code}: {r.text}")
        failed += 1

    # Test 5: Empty text validation
    print("\n  Testing: POST /moderate (empty text — should 422)")
    r = client.post("/moderate", json={"text": "   "})
    if r.status_code == 422:
        ok("Empty text correctly rejected with 422")
        passed += 1
    else:
        fail(f"Empty text returned {r.status_code} (expected 422)")
        failed += 1

    # Test 6: Batch endpoint
    print("\n  Testing: POST /moderate/batch")
    r = client.post("/moderate/batch", json={
        "texts": [
            "You are wonderful!",
            "I will destroy you",
            "The weather is nice today",
        ]
    })
    if r.status_code == 200:
        data = r.json()
        if data["total"] == 3 and "toxic_count" in data:
            ok(f"Batch of 3: total={data['total']}, toxic={data['toxic_count']}")
            passed += 1
        else:
            fail(f"Unexpected batch response: {data}")
            failed += 1
    else:
        fail(f"POST /moderate/batch returned {r.status_code}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")


def show_sample_output():
    """Show a nicely formatted sample prediction."""
    header("SAMPLE API OUTPUT")
    try:
        from predict import predict_toxicity
        result = predict_toxicity("You are so stupid, I can't believe anyone listens to you")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"{YELLOW}Could not generate sample (model not loaded): {e}{RESET}")


if __name__ == "__main__":
    print(f"\n{BOLD}{'='*50}")
    print("AI CONTENT MODERATION — TEST SUITE")
    print(f"{'='*50}{RESET}")

    run_direct_tests()
    run_api_tests()
    show_sample_output()

    print(f"\n{BOLD}All tests complete.{RESET}")
    print("To start the API server, run:")
    print(f"  {BLUE}cd backend && uvicorn main:app --reload --port 8000{RESET}\n")
