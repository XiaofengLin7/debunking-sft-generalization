"""
Diversify the EleutherAI/hendrycks_math dataset with three OpenAI-backed agents:

1. DiversifierAgent: rewrites each original problem into a fresh surface form that preserves
   the underlying reasoning structure (no solutions produced here).
2. SolverAgent: uses the original problem/solution as a reference, then solves the rewritten problem.
3. VerifierAgent: judges whether the solver's answer is correct for the new problem.

By default the script iterates over every official subset (algebra, geometry, etc.), grabs the
first two training problems per subset, and writes the augmented records to JSONL.

Usage:
  python -m debunk_sft.utils.dataset.math.diversify_math \
    --model-diversifier gpt-4.1 \
    --model-solver o3 \
    --model-verifier o3-mini \
    --output /path/to/diversified_math.jsonl

Environment:
  - Requires OPENAI_API_KEY set in the environment or passed through OpenAI's default config.
  - Optionally supports OPENAI_BASE_URL for compatible endpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

try:
    # OpenAI SDK v1.x
    from openai import OpenAI
except Exception as _exc:  # pragma: no cover - import fallback for environments without openai
    OpenAI = None  # type: ignore


DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY_S = 2.0

MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _ensure_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "openai package not found. Install `openai>=1.0.0` and set OPENAI_API_KEY."
        )
    # Let the SDK pick up OPENAI_API_KEY / OPENAI_BASE_URL from env if present.
    return OpenAI()


def _exponential_backoff(
    attempt: int,
    base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
) -> float:
    # Jittered exponential backoff
    expo = base_delay_s * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0, base_delay_s)
    return min(60.0, expo + jitter)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try parsing JSON from an LLM response. If the response contains extra text,
    attempt to extract the first JSON object.
    """
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # Heuristic extraction: find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from model response.")


@dataclasses.dataclass
class Diversification:
    diversified_problem: str
    transformation_type: str = "rewrite"
    diversified_solution: Optional[str] = None
    change_description: Optional[str] = None
    changed_quantity_before: Optional[str] = None
    changed_quantity_after: Optional[str] = None
    notes: Optional[str] = None


@dataclasses.dataclass
class Verification:
    verdict: str  # "pass" | "fail"
    reason: str
    consistency_checks: Dict[str, Any]


class DiversifierAgent:
    """LLM-backed problem rewriter that produces a new surface form only."""

    def __init__(self, model: str):
        self.model = model
        self.client = _ensure_openai_client()

    def diversify(
        self,
        problem: str,
        solution: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Tuple[str, Diversification]:
        system_msg = (
            "You are a math problem rewriter for dataset augmentation.\n\n"
            "Goal:\n"
            "Given an original problem and its official solution, create a NEW problem that:\n"
            "- Uses the SAME underlying reasoning steps and mathematical structure.\n"
            "- Has a DIFFERENT surface form (phrasing, variable names, ordering, numbers).\n"
            "- Is well-posed and solvable.\n\n"
            "Requirements:\n"
            "- Preserve the method, qualitative structure, and answer type.\n"
            "- You MAY change values, rename symbols, reorder presentation, or tweak style.\n"
            "- You MUST NOT change the topic/method, make it trivial/harder, introduce contradictions, or output any solution.\n\n"
            "Output only the new problem as a single self-contained statement."
        )
        user_msg = (
            "Original problem:\n---\n"
            f"{problem}\n---\n\n"
            "Official solution (for understanding the reasoning steps):\n---\n"
            f"{solution}\n---\n\n"
            "Task: Rewrite the problem as a NEW PROBLEM that satisfies the system instructions.\n"
            "Do NOT output any solution or explanation.\n\n"
            "NEW PROBLEM:"
        )

        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                )
                content = response.choices[0].message.content or ""
                new_problem = self._extract_problem_statement(content)
                if not new_problem:
                    raise ValueError("Diversifier returned empty problem text.")
                diversification = Diversification(
                    diversified_problem=new_problem,
                    transformation_type="rewrite",
                    diversified_solution=None,
                )
                return content, diversification
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == max_retries:
                    raise
                time.sleep(_exponential_backoff(attempt))
        raise RuntimeError(f"Diversifier failed after retries: {last_error}")

    @staticmethod
    def _extract_problem_statement(content: str) -> str:
        text = content.strip()
        marker = "NEW PROBLEM:"
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].strip() if len(parts) >= 2 else text.strip("`")
        return text.strip()

class VerifierAgent:
    """LLM-backed judge that decides CORRECT/INCORRECT based on the solver's answer."""

    def __init__(self, model: str):
        self.model = model
        self.client = _ensure_openai_client()

    def verify(
        self,
        original_problem: str,
        original_solution: str,
        diversification: Diversification,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Tuple[str, Verification]:
        system_msg = (
            "You are a math solution verifier.\n\n"
            "Your job:\n"
            "- Check whether a PROPOSED SOLUTION for a NEW PROBLEM is mathematically correct.\n"
            "- The ORIGINAL PROBLEM and OFFICIAL SOLUTION are provided only for context.\n\n"
            "Requirements:\n"
            "1. Focus on whether the proposed solution correctly solves the NEW PROBLEM.\n"
            "2. Different reasoning is acceptable if it is sound.\n"
            "3. Do not rewrite or fix the solution; simply judge it.\n"
            "4. Output format:\n"
            "   REASON: brief justification (1–5 sentences).\n"
            "   VERDICT: CORRECT or INCORRECT"
        )
        user_msg = (
            "ORIGINAL PROBLEM:\n---\n"
            f"{original_problem}\n---\n\n"
            "OFFICIAL SOLUTION:\n---\n"
            f"{original_solution}\n---\n\n"
            "NEW PROBLEM:\n---\n"
            f"{diversification.diversified_problem}\n---\n\n"
            "PROPOSED SOLUTION:\n---\n"
            f"{diversification.diversified_solution}\n---\n\n"
            "Decide if the proposed solution correctly solves the new problem."
        )

        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0 if self.model != "o3-mini" else None,
                )
                content = response.choices[0].message.content or ""
                verdict_word, reason = self._parse_verifier_response(content)
                if verdict_word not in {"correct", "incorrect"}:
                    print("content: ", content)
                    print("verdict_word: ", verdict_word)
                    print("reason: ", reason)
                    raise ValueError("Verifier must output VERDICT: CORRECT/INCORRECT.")
                verdict = "pass" if verdict_word == "correct" else "fail"
                if not reason:
                    reason = "No reason provided by verifier."
                verification = Verification(
                    verdict=verdict,
                    reason=reason,
                    consistency_checks={},
                )
                return content, verification
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == max_retries:
                    raise
                time.sleep(_exponential_backoff(attempt))
        raise RuntimeError(f"Verifier failed after retries: {last_error}")

    @staticmethod
    def _parse_verifier_response(content: str) -> Tuple[str, str]:
        verdict_word: Optional[str] = None
        reason_text: Optional[str] = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("VERDICT:"):
                verdict_word = stripped.split(":", 1)[1].strip().lower().rstrip(".! ")
            elif stripped.upper().startswith("REASON:"):
                reason_text = stripped.split(":", 1)[1].strip()
        return verdict_word or "", reason_text or ""

class SolverAgent:
    """LLM-backed solver that mirrors the original reasoning pattern for the new problem."""

    def __init__(self, model: str):
        self.model = model
        self.client = _ensure_openai_client()

    def solve(
        self,
        original_problem: str,
        original_solution: str,
        diversified_problem: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Tuple[str, str]:
        system_msg = (
            "You are a math solution writer.\n\n"
            "Given an ORIGINAL PROBLEM and its OFFICIAL SOLUTION, plus a NEW PROBLEM, produce a "
            "complete, correct solution for the NEW PROBLEM, adapting all steps and calculations accordingly.\n"
            "Show step-by-step reasoning and end with 'Final answer: ...'. Do not mention the original problem."
        )
        user_msg = (
            "Original problem:\n---\n"
            f"{original_problem}\n---\n\n"
            "Original official solution (for reference only):\n---\n"
            f"{original_solution}\n---\n\n"
            "New problem:\n---\n"
            f"{diversified_problem}\n---\n\n"
            "Task: Using the system instructions, solve the NEW PROBLEM. Provide a clear step-by-step solution and end with 'Final answer: ...'"
        )

        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=1.0,
                )
                content = response.choices[0].message.content or ""
                final_solution = content.strip()
                if "Final answer:" not in final_solution:
                    final_solution += "\n\nFinal answer: [unspecified]"
                return content, final_solution
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == max_retries:
                    raise
                time.sleep(_exponential_backoff(attempt))
        raise RuntimeError(f"Solver failed after retries: {last_error}")

def _load_hendrycks_math(
    subset: Optional[str],
    split: str,
) -> Any:
    """
    Load the EleutherAI/hendrycks_math dataset (optionally a specific subset).
    Valid subsets include: algebra, counting_and_probability, geometry, intermediate_algebra,
    number_theory, prealgebra, precalculus (see dataset card).
    """
    if subset:
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
    else:
        # When subset isn't provided, load the combined split
        ds = load_dataset("EleutherAI/hendrycks_math", split=split)
    return ds


def _iter_samples(dataset: Any, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    count = 0
    for row in dataset:
        yield row
        count += 1
        if limit is not None and count >= limit:
            break


def diversify_math_dataset(
    model_diversifier: str,
    model_verifier: str,
    model_solver: str,
    subset: Optional[str],
    split: str,
    per_subset_limit: int,
    seed: int,
    output_path: str,
    include_failed: bool = False,
) -> Tuple[int, int]:
    """
    Run the diversification pipeline and write results as JSONL.

    Returns:
      (num_attempted, num_accepted)
    """
    random.seed(seed)

    diversifier = DiversifierAgent(model=model_diversifier)
    verifier = VerifierAgent(model=model_verifier)
    solver = SolverAgent(model=model_solver)

    subsets = (
        MATH_SUBSETS
        if subset is None or subset.lower() == "all"
        else [subset]
    )

    attempted = 0
    accepted = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for subset_name in subsets:
            dataset = _load_hendrycks_math(subset=subset_name, split=split)
            for row in _iter_samples(dataset, limit=per_subset_limit):
                attempted += 1
                original_problem: str = str(row.get("problem", "")).strip()
                original_solution: str = str(row.get("solution", "")).strip()
                level = row.get("level")
                problem_type = row.get("type")

                if not original_problem or not original_solution:
                    continue

                raw_diversifier_response: Optional[str] = None
                raw_solver_response: Optional[str] = None
                raw_verifier_response: Optional[str] = None

                try:
                    raw_diversifier_response, diversification = diversifier.diversify(
                        problem=original_problem,
                        solution=original_solution,
                    )
                    raw_solver_response, solver_solution = solver.solve(
                        original_problem=original_problem,
                        original_solution=original_solution,
                        diversified_problem=diversification.diversified_problem,
                    )
                    if not solver_solution:
                        raise ValueError("Solver failed to produce a solution.")
                    diversification = dataclasses.replace(
                        diversification, diversified_solution=solver_solution
                    )
                    raw_verifier_response, verification = verifier.verify(
                        original_problem=original_problem,
                        original_solution=original_solution,
                        diversification=diversification,
                    )
                    passed = verification.verdict == "pass"
                except Exception as exc:
                    diversification = None
                    verification = Verification(
                        verdict="fail",
                        reason=f"Pipeline error: {type(exc).__name__}: {exc}",
                        consistency_checks={},
                    )
                    passed = False

                if passed or include_failed:
                    record = {
                        "subset": subset_name,
                        "split": split,
                        "level": level,
                        "type": problem_type,
                        "original_problem": original_problem,
                        "original_solution": original_solution,
                        "diversified_problem": getattr(diversification, "diversified_problem", None)
                        if diversification
                        else None,
                        "diversified_solution": getattr(diversification, "diversified_solution", None)
                        if diversification
                        else None,
                        "transformation_type": getattr(diversification, "transformation_type", None)
                        if diversification
                        else None,
                        "change_description": getattr(diversification, "change_description", None)
                        if diversification
                        else None,
                        "changed_quantity_before": getattr(diversification, "changed_quantity_before", None)
                        if diversification
                        else None,
                        "changed_quantity_after": getattr(diversification, "changed_quantity_after", None)
                        if diversification
                        else None,
                        "notes": getattr(diversification, "notes", None) if diversification else None,
                        "diversifier_raw_response": raw_diversifier_response,
                        "solver_raw_response": raw_solver_response,
                        "verifier_raw_response": raw_verifier_response,
                        "verification": dataclasses.asdict(verification),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print("record: ", record)
                    if passed:
                        accepted += 1
    return attempted, accepted


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diversify EleutherAI/hendrycks_math with three agents")
    parser.add_argument(
        "--model-diversifier",
        default="gpt-4.1",
        help="OpenAI chat model for diversification (default: gpt-4.1)",
    )
    parser.add_argument(
        "--model-verifier",
        default="o3-mini",
        help="OpenAI chat model for verification (default: o3-mini)",
    )
    parser.add_argument(
        "--model-solver",
        default="o3",
        help="OpenAI chat model for solving rewritten problems (default: o3)",
    )
    parser.add_argument(
        "--subset",
        default="all",
        help="Dataset subset to process (default: all official subsets).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (train|test). Default: train",
    )
    parser.add_argument(
        "--per-subset-limit",
        type=int,
        default=2,
        help="Number of problems to process per subset (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to JSONL output file",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed/invalid samples in the output JSONL",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    attempted, accepted = diversify_math_dataset(
        model_diversifier=args.model_diversifier,
        model_verifier=args.model_verifier,
        model_solver=args.model_solver,
        subset=args.subset,
        split=args.split,
        per_subset_limit=args.per_subset_limit,
        seed=args.seed,
        output_path=args.output,
        include_failed=args.include_failed,
    )
    print(f"attempted\t{attempted}")
    print(f"accepted\t{accepted}")


if __name__ == "__main__":
    main(sys.argv[1:])


