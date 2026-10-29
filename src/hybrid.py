"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer
"""
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle yourself.
Translate EVERY numbered clue literally one by one into `constraints`.

{ir_spec}

CRITICAL RULES:
1. VARIABLES & ALL_DIFFERENT:
   - Create an integer variable for EVERY name/value listed in the "Answer format" at the bottom.
   - Use the EXACT spelling and capitalization from the Answer format (e.g. if it says "Lesia", write `Lesia`, NOT `lesia`; if it says "vision", write `vision`).
   - Your "variables" dictionary MUST contain ONLY the exact names, colors, pets, or items explicitly listed in the "Answer format".
   - Grid puzzles (houses, spots, offices): domain is [1, N]. Group variables into separate `all_different` lists by category (Names together, Colors together, Pets together, etc.).
   - Line / Shelf / Photo ordering puzzles: domain is [1, N]. Put all items in ONE `all_different` list.
   - Knights and Knaves puzzles: Create variables ONLY for the names of the speakers (domain [0, 1]). NEVER create variables for the words "knight", "knave", "true", or "false".
   - NEVER include `"objective"` unless the puzzle asks to maximize or minimize a number.
   - COUNT THE CLUES! If the text has exactly 19 numbered clues, your `constraints` list MUST have EXACTLY 19 strings. Clue 1 must be string 1, Clue 19 must be string 19. Do not skip or combine clues!

2. HOW TO TRANSLATE CLUES FOR KNIGHTS AND KNAVES PUZZLES ONLY:
   - Represent each person as a variable with the domain [0, 1] (0 for knave, 1 for knight). 
   - Use the EXACT full names of the people as variable names (e.g., "Alice", "Bob"). Do NOT abbreviate to single letters.
   - Do NOT use `all_different`. Leave it empty.
   - A statement S made by person X translates to the constraint "X == (S)".
   - Use `and` and `or` for logical connections. 
   - "A and B are knights" translates to "(A == 1) and (B == 1)". 
   - "A or B is a knight" translates to "(A == 1) or (B == 1)".
   - Use addition ONLY for counting. "Exactly two of us are knights" translates to adding all variables: "(A == 1) + (B == 1) + (C == 1) == 2".
   - Do NOT use string values inside constraints, only integer checks.

3. HOW TO TRANSLATE CLUES (Translate EACH clue entirely in ISOLATION. Do NOT mix names from one clue into another!):
   - "X stands at the right end" (out of N positions): `X == N`
   - "X stands at the left end": `X == 1`
   - "X stands at one of the two ends" (out of N positions): `(X == 1) or (X == N)`
   - "X is next to Y or Z": `(abs(X - Y) == 1) or (abs(X - Z) == 1)` (NEVER put `or` inside `abs()`)
   - "Exactly K people/items stand between X and Y": `abs(X - Y) - 1 == K` (NOT FOR KNIGHTS AND KNAVES PUZZLES)

4. HOW TO TRANSLATE FOR GRID PUZZLES (houses, spots, parking lot):
   - "X has Y" / "The person with X has/owns/drives/plays/works on Y": `X == Y`
   - "X does NOT have/own/drive Y": `X != Y`
   - "X is in house/spot K" / "X stands K-th from the left": `X == K`
   - "X is next to Y" / "X lives/parks next to Y": `abs(X - Y) == 1`
   - "X is NOT next to Y": `abs(X - Y) != 1`
   - "X is somewhere to the left of Y": `X < Y`   (Notice: `<`, NOT `- 1`!)
   - "X is immediately to the left of Y": `X == Y - 1`
   - "X is immediately to the right of Y": `X == Y + 1`

Example 1 (Knights and Knaves):
Puzzle: You meet Alice, Bob and Charlie.
Alice says: "Bob and Charlie are both knights."
Bob says: "Alice is a knight or Charlie is a knight."
Charlie says: "Bob and I are of different kinds."
Output:
{{
  "variables": {{"Alice": [0, 1], "Bob": [0, 1], "Charlie": [0, 1]}},
  "all_different": [],
  "constraints": [
    "Alice == ((Bob == 1) and (Charlie == 1))",
    "Bob == ((Alice == 1) or (Charlie == 1))",
    "Charlie == (Bob != Charlie)"
  ]
}}

Example 2 (Ordering / Line):
Puzzle: Five items (Pen, Book, Cup, Lamp, Box) stand in positions 1 to 5 from left to right.
1. The Lamp stands at one of the two ends.
2. The Book stands immediately to the right of the Pen.
3. The Box is not next to the Lamp.
4. Exactly one item stands between the Pen and the Cup.
5. Exactly two items stand between the Book and the Box.
6. The Lamp is somewhere to the left of the Book.
Output:
{{
  "variables": {{"Pen": [1, 5], "Book": [1, 5], "Cup": [1, 5], "Lamp": [1, 5], "Box": [1, 5]}},
  "all_different": [["Pen", "Book", "Cup", "Lamp", "Box"]],
  "constraints": [
    "(Lamp == 1) or (Lamp == 5)",
    "Book == Pen + 1",
    "abs(Box - Lamp) != 1",
    "abs(Pen - Cup) - 1 == 1",
    "abs(Book - Box) - 1 == 2",
    "Lamp < Book"
  ]
}}

Example 3 (Multi-attribute Grid):
Puzzle: Three people live in houses 1 to 3 from left to right. Each has a different name, drink and fruit.
1. Sam lives in house 2.
2. Sam lives next to Taylor.
3. The person who eats plum drinks coffee.
4. Alex lives somewhere to the left of the person who eats apple.
5. The person who eats pear lives next to the person who drinks tea.
6. Taylor does not drink milk.
7. The person who drinks coffee lives in house 3.
8. Alex lives immediately to the left of Sam.
Answer format: Put into "solution" an object whose keys are "1", "2", "3" and whose values are objects with the keys "Name", "Drink", "Fruit". Use exactly these values — Name: "Alex", "Sam", "Taylor"; Drink: "tea", "coffee", "milk"; Fruit: "apple", "pear", "plum".
Output:
{{
  "variables": {{
    "Alex": [1, 3], "Sam": [1, 3], "Taylor": [1, 3],
    "tea": [1, 3], "coffee": [1, 3], "milk": [1, 3],
    "apple": [1, 3], "pear": [1, 3], "plum": [1, 3]
  }},
  "all_different": [
    ["Alex", "Sam", "Taylor"],
    ["tea", "coffee", "milk"],
    ["apple", "pear", "plum"]
  ],
  "constraints": [
    "Sam == 2",
    "abs(Sam - Taylor) == 1",
    "plum == coffee",
    "Alex < apple",
    "abs(pear - tea) == 1",
    "Taylor != milk",
    "coffee == 3",
    "Alex == Sam - 1"
  ]
}}

Puzzle:
{text}
{feedback}"""


def sanitize_ir(ir: dict, problem: dict) -> dict:
    """Fix common small-LLM (3B) formatting quirks before validation and solving."""
    vars_dict = ir.get("variables", {})
    if not vars_dict:
        return ir

    # Fix hallucinated full lists for domains
    for name, dom in vars_dict.items():
        if isinstance(dom, list) and len(dom) > 2:
            vars_dict[name] = [min(dom), max(dom)]
    
    # Fix case-sensitivity mismatches in all_different and constraints (e.g. "lesia" -> "Lesia")
    case_map = {k.lower(): k for k in vars_dict.keys()}

    def fix_expr_identifiers(expr: str) -> str:
        def repl(match):
            word = match.group(0)
            if word in ("and", "or", "not", "abs"):
                return word
            return case_map.get(word.lower(), word)
        return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, expr)

    ir["all_different"] = [
        [case_map.get(name.lower(), name) for name in group]
        for group in ir.get("all_different", [])
    ]
    ir["constraints"] = [fix_expr_identifiers(c) for c in ir.get("constraints", [])]

    # If it's a Knights & Knaves puzzle (all domains are [0, 1] and >2 variables),
    # remove hallucinated all_different and hallucinated objective
    all_binary = all(dom == [0, 1] for dom in vars_dict.values())
    if all_binary and len(vars_dict) > 2:
        ir["all_different"] = []

    # Remove objective unless problem is actually an optimization problem
    text_lower = (problem.get("text", "") + " " + problem.get("answer_format", "")).lower()
    if "optim" not in text_lower and "maxim" not in text_lower and "minim" not in text_lower:
        ir.pop("objective", None)

    return ir


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    feedback_text = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    full_text = f"{problem['text']}\n\nAnswer format: {problem.get('answer_format', '')}"
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=full_text, feedback=feedback_text)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    ir = parsed.model_dump(exclude_none=True)
    return sanitize_ir(ir, problem)


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid."""
    errors = []
    try:
        ir_solver.build(ir)
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Invalid expression syntax: {e}")

    for name, dom in ir.get("variables", {}).items():
        if isinstance(dom, list):
            if len(dom) != 2:
                errors.append(f"Domain for {name} must be [lo, hi], got {dom}")
            else:
                lo, hi = dom
                if not (isinstance(lo, int) and isinstance(hi, int)):
                    errors.append(f"Domain for {name} must contain integers, got {lo} and {hi}")
                elif lo > hi:
                    errors.append(f"Domain for {name} has lo > hi: {lo} > {hi}")
        elif isinstance(dom, dict):
            if "values" not in dom:
                errors.append(f"Domain dict for {name} must have 'values' key")
            else:
                vals = dom["values"]
                if not isinstance(vals, list) or not all(isinstance(v, int) for v in vals):
                    errors.append(f"Domain 'values' for {name} must be a list of integers")
        else:
            errors.append(f"Invalid domain format for {name}: {dom}")

    return errors


def run_solver(ir: dict) -> dict:
    return ir_solver.solve(ir, limit=2)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def _try_present_grid(answer_fmt_orig: str, assignment: dict) -> Optional[dict]:
    """Deterministic handler for multi-attribute grid puzzles (e.g., D2 Four houses, D3 Parking lot).
    Parses category definitions like:
    'Use exactly these values — Name: "Yurii", "Zlata"...; Pet: "cat", "dog"...'
    """
    if "use exactly these values" not in answer_fmt_orig.lower():
        return None

    try:
        keys_match = re.search(r"keys are ([^\n]+?) and whose values", answer_fmt_orig)
        if not keys_match:
            return None
        outer_keys = re.findall(r'"([^"]+)"', keys_match.group(1))
        if not outer_keys:
            return None

        after_values = re.split(r"use exactly these values\s*[—:-]\s*", answer_fmt_orig, flags=re.IGNORECASE)[-1]
        category_segments = after_values.split(";")
        
        categories = {}
        for segment in category_segments:
            if ":" not in segment:
                continue
            raw_category_name, raw_values_string = segment.split(":", 1)

            category_name = raw_category_name.strip().strip('".')
            category_values = re.findall(r'"([^"]+)"', raw_values_string)
            
            if category_name and category_values:
                categories[category_name] = category_values

        if not categories:
            return None

        assign_lower = {k.lower(): v for k, v in assignment.items()}

        solution = {}
        for idx, o_key in enumerate(outer_keys, start=1):
            pos = int(o_key) if o_key.isdigit() else idx
            row_obj = {}
            for category_name, category_values in categories.items():
                for val in category_values:
                    var_key = val.lower().replace(" ", "_").replace("-", "_")
                    if assign_lower.get(var_key) == pos or assign_lower.get(val.lower()) == pos:
                        row_obj[category_name] = val
                        break
            solution[o_key] = row_obj

        return solution
    
    except Exception:
        # If any parsing error occurs, return None to pass the task back to LLM
        return None


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format.
    Uses deterministic code for known problem types, and falls back to the LLM for unknown problems.
    """
    status = result["status"]

    if status == "none":
        return Answer(status="none").model_dump()

    assignment = result.get("solutions", [{}])[0] if result.get("solutions") else {}
    answer_fmt_orig = problem.get("answer_format", "")
    answer_fmt = answer_fmt_orig.lower()

    # Deterministic handler: Problem Type "Knights and Knaves"
    if "knight" in answer_fmt and "knave" in answer_fmt:
        solution = {k: ("knight" if v == 1 else "knave") for k, v in assignment.items()}
        return Answer(status=status, solution=solution).model_dump()

    # Deterministic handler: Problem Type "Ordering / Positions"
    if "position" in answer_fmt and "integer" in answer_fmt:
        return Answer(status=status, solution=assignment).model_dump()

    # Deterministic handler: Problem Type "Multi-attribute Grid"
    grid_sol = _try_present_grid(answer_fmt_orig, assignment)
    if grid_sol is not None:
        return Answer(status=status, solution=grid_sol).model_dump()

    # Fallback: Using LLM for unknown/complex tasks
    prompt = PRESENT_PROMPT.format(
        text=problem.get("text", ""),
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(assignment, ensure_ascii=False),
        answer_format=answer_fmt_orig
    )
    parsed, raw = llm.ask_json(prompt, Answer)

    return Answer(
        status=status,
        solution=parsed.solution,
        objective=result.get("objective")
    ).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        # One more try
        ir = formalize(problem, feedback="\n".join(errors))
        errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        raise HybridError("invalid IR after retry: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {
        "status": result["status"],
        "n_solutions": len(result["solutions"]),
        "objective": result.get("objective"),
        "first_solution": result["solutions"][0] if result["solutions"] else None
    }

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}