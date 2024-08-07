"""
Pretty prints JSONL files with optional selective output.

This is code that I've written in the past for dealing with JSONL files, including it here because I used it to examine the files.

Run it like this:

```
python pprint_json.py dev.jsonl -n 1 -p label text -r
```
"""

import argparse
import difflib
import json
import random
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import TextIO

try:
    from llms.agent_evals.primitives.code_generation_problem_types import CodeGenerationProblem
    from llms.agent_evals.primitives.code_generation_problem_types import SolutionAttempt

    CHECK_TYPES = True
except ImportError:
    CHECK_TYPES = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax

    USE_RICH = True
except ImportError:
    USE_RICH = False


WIDTH = 100
if USE_RICH:
    console = Console(force_terminal=True)

MAX_PRINT_LEN = None


def print_header_1(text: str) -> None:
    if USE_RICH:
        console.print(Markdown(f"# {text}"))
    else:
        space = (WIDTH - len(text)) // 2
        print("*" * WIDTH)
        print(" " * space + text)
        print("*" * WIDTH)


def print_header_2(text: str) -> None:
    if USE_RICH:
        console.print(Markdown(f"## {text}"))
    else:
        space = (WIDTH - len(text)) // 2
        underline_start = "\033[4m"
        underline_end = "\033[0m"
        print("\n" + " " * space + underline_start + text + underline_end)


def print_header_3(text: str) -> None:
    if USE_RICH:
        console.print(Markdown(f"### {text}"))
    else:
        print(f"\n**** {text} ****\n")


def print_text(text: str) -> None:
    if MAX_PRINT_LEN is not None and len(text) > MAX_PRINT_LEN:
        text = text[:MAX_PRINT_LEN] + f"... ({len(text) - MAX_PRINT_LEN} characters truncated)"
    if USE_RICH:
        console.print(Markdown(text))
    else:
        print(text)


def print_code(code: str, print_line_numbers: bool = False, lexer: str = "python") -> None:
    if MAX_PRINT_LEN is not None and len(code) > MAX_PRINT_LEN:
        code = code[:MAX_PRINT_LEN] + f"... ({len(code) - MAX_PRINT_LEN} characters truncated)"
    if USE_RICH:
        console.print(Syntax(code, lexer, theme="monokai", line_numbers=print_line_numbers, word_wrap=True))
    else:
        for i, l in enumerate(code.split("\n")):
            if print_line_numbers:
                print(f"{i + 1:3}:\t{l}")
            else:
                print(l)


def process_file(file: TextIO) -> str:
    s = file.read()
    return s


# Common locations in JSONL files for these parts to be found. This attempts to be robust to different naming conventions. "/" indicates a nested key, like problem["foo"]["bar"] as "foo/bar".
COMMON_LOCATIONS: dict[str, list[str]] = {
    "code": ["code", "attempt_module"],  # Note that this is "solution code", and you may want "prompt" or "attempts"
    "broken_code": ["broken_code"],
    "prompt": ["prompt", "question", "problem description", "problem/code_module", "code_module"],
    "tests": ["tests", "problem/test_module", "test_module"],
    "constraints": ["constraints"],
    "background_code": ["background_code", "background"],
    "broken_suggestions": ["broken_suggestions"],
    "tests_pass": ["tests_pass"],
    "tests_error": ["tests_error"],
    "attempts": ["attempts"],
}


def get_nested_value(d: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    return d


def build_parts(problem: dict, parts: list[str]) -> dict[str, Any]:
    """
    Tries to robustly find the listed parts, looking in likely places.
    """
    results = {}
    for part in COMMON_LOCATIONS.keys():
        if part in problem:
            results[part] = problem[part]
        else:
            for key in COMMON_LOCATIONS[part]:
                if "/" in key:
                    keys = key.split("/")
                    value = get_nested_value(problem, keys)
                    if value is not None:
                        results[part] = value
                        break
                elif key in problem:
                    results[part] = problem[key]
                    break

    for part in parts:
        if "/" in part:
            keys = part.split("/")
            value = get_nested_value(problem, keys)
            if value is not None:
                results[part] = value
                break
        if part not in results:
            results[part] = problem.get(part, None)

    return results


def print_problem(orig_problem, parts: Optional[list[str]] = None, print_line_numbers: bool = False) -> None:
    """
    Pretty print a problem, with an option to specify which parts are printed. Uses the `rich` library if installed. It attempts to print code blocks with syntax highlighting, and tests with pass/fail status.
    """
    using_default_parts = False
    if parts is None:
        parts = ["prompt", "code", "tests", "attempts"]
        using_default_parts = True
    problem = build_parts(orig_problem, parts=parts)  # This is where we look for alternative locations
    for part in parts:
        if (
            not using_default_parts
            or part in problem
            or (using_default_parts and part == "attempts" and "executed_attempts" in orig_problem)  # Edge case
        ):
            print_header_2(part.capitalize().replace("_", " "))
        # Sometimes the prompt is code-like. This is a heuristic to determine if it is.
        is_code_like = (
            part in problem
            and isinstance(problem[part], str)
            and (
                problem[part].strip().startswith("import")
                or problem[part].strip().startswith("from")
                or problem[part].strip().startswith("def")
            )
        )
        # These are special cases, before we check `part not in problem`
        if part == "broken_diff":
            code = problem["code"]
            broken_code = problem["broken_code"]
            diff = difflib.unified_diff(
                broken_code.splitlines(keepends=True),
                code.splitlines(keepends=True),
                fromfile="broken_code",
                tofile="code",
                lineterm="\n",
                n=3,  # Number of lines of context
            )
            diff_str = "\n".join([line.strip() for line in diff])
            print_code(diff_str, print_line_numbers=print_line_numbers, lexer="diff")
        elif part == "attempts":
            try:
                attempts = orig_problem["executed_attempts"]
                print_text(f"Found {len(attempts)} attempts")
                for i, attempt in enumerate(attempts):
                    print_header_3(f"Attempt {i + 1} Code")
                    attempt_code = attempt["attempt"]["attempt_module"]
                    print_code(attempt_code, print_line_numbers=print_line_numbers)
                    print_header_3(f"Attempt {i + 1} Results")
                    print_code(attempt["execution_result"]["stdout"], print_line_numbers=print_line_numbers)
                    print_header_3(f"Attempt {i + 1} Return Code")
                    rc = attempt["execution_result"]["return_code"]
                    print_text(f'Return code: {rc} ({"success" if rc == 0 else "failure" if rc == 1 else "unknown"})')
            except KeyError as e:
                print_text(f"Error parsing attempts. KeyError: {e}")
                print_text(f"Problem details: {json.dumps(orig_problem)}")
        elif part not in problem:
            print(problem)
            if not using_default_parts:
                print_text(f'Part "{part}" not found in problem')
            continue
        # Specially handle each type of thing
        elif part == "code":
            print_code(problem[part], print_line_numbers=print_line_numbers)
        elif part == "broken_code":
            print_code(problem[part], print_line_numbers=print_line_numbers)
        elif part == "prompt":
            if is_code_like:
                print_code(problem[part], print_line_numbers=print_line_numbers)
            else:
                print_text(problem[part])
        elif part == "tests":
            tests = problem[part]
            if isinstance(tests, str):
                tests = [tests]
            passes = orig_problem["tests_pass"] if "tests_pass" in orig_problem else ["Unk."] * len(tests)
            errors = orig_problem["tests_error"] if "tests_error" in orig_problem else ["Unk."] * len(tests)
            error_texts = (
                orig_problem["tests_error_texts"] if "tests_error_texts" in orig_problem else [""] * len(tests)
            )
            for j in range(len(tests)):
                quote = '"'
                if "tests_pass" in orig_problem or "tests_error" in orig_problem:
                    print_text(
                        f"Test {j + 1}. Passes: [{'blue' if passes[j] else 'red'}]{passes[j]}[/]. Errors: [{'red' if errors[j] else 'blue'}]{errors[j]}[/], {quote + error_texts[j] + quote if errors[j] else ''}"
                    )
                else:
                    print_text(f"Test {j + 1}.")
                print_code(tests[j], lexer="python", print_line_numbers=print_line_numbers)
        elif part == "constraints":
            print_text("\n".join([f" - {c}" for c in problem[part]]))
        elif part == "background":
            print_code(problem[part], print_line_numbers=print_line_numbers)
        elif part == "suggestions":
            print_text("\n".join([f" - {c}" for c in problem[part]]))
        elif part == "broken_suggestions":
            print_text("\n".join([f" - {c}" for c in problem[part]]))
        else:
            # Unknown type fallback
            if not isinstance(problem[part], str):
                print_text(json.dumps(problem[part], indent=4))
            else:
                if is_code_like:
                    print_code(problem[part], print_line_numbers=print_line_numbers)
                else:
                    print_text(problem[part])


def get_type(value: Any) -> str:
    """Get the type of the value as a string."""
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return f"list[{get_type(value[0])}]" if value else "list"
    return type(value).__name__


def print_json_structure(data: Dict[str, Any], indent: int = 1) -> str:
    """Recursively print the structure of a JSON object."""
    s = "    " * (indent - 1) + "{\n"
    if "__type" in data:
        s += "    " * indent + f'"__type": {data["__type"]}\n'
    for key, value in data.items():
        the_type = get_type(value)
        if key == "__type":
            pass
        elif isinstance(value, str):
            s += "    " * indent + f'"{key}": {the_type} ({len(value)} characters)\n'
        elif isinstance(value, list):
            s += "    " * indent + f'"{key}": {the_type} ({len(value)} items)\n'
        elif isinstance(value, dict):
            s += "    " * indent + f'"{key}": {the_type} ({len(value)} items)\n'
        else:
            s += "    " * indent + f'"{key}": {the_type}\n'
        # Recursion
        if isinstance(value, dict):
            s += print_json_structure(value, indent + 1) + "\n"
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            s += "    " * indent + f"[\n"
            s += print_json_structure(value[0], indent + 2) + "\n"
            s += "    " * (indent + 1) + f"... ({len(value) - 1} more items)\n"
            s += "    " * (indent) + f"]\n"
    return s + "    " * (indent - 1) + "}"


def remove_type_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: remove_type_keys(v) for k, v in data.items() if k != "__type"}
    elif isinstance(data, list):
        return [remove_type_keys(item) for item in data]
    else:
        return data


def truncate_strings(data, max_length):
    if isinstance(data, dict):
        return {k: truncate_strings(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_strings(item, max_length) for item in data]
    elif isinstance(data, str):
        if len(data) > max_length:
            return data[:max_length] + f"... ({len(data) - max_length} characters truncated)"
        else:
            return data
    else:
        return data


def main() -> None:
    global WIDTH
    description, epilog = __doc__.split("\n\n", 1)
    parser = argparse.ArgumentParser(
        description=description.strip(),
        epilog=epilog.strip(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_argument_group("Main Arguments")
    group.add_argument(
        "file",
        nargs="?",
        type=str,
        default=sys.stdin,
        help="The file to process. This may be an S3 location. Defaults to stdin.",
    )
    group.add_argument(
        "-p",
        "--parts",
        nargs="*",
        type=str,
        help=f'Optional list of parts to print. Will attempt to find useful defaults if not specified. Possible values: {list(COMMON_LOCATIONS.keys())}, or any other top level JSONL key. You can use a slash, like "foo/bar" to indicate a nested key like problem["foo"]["bar"].',
    )
    group = parser.add_argument_group("Line Selection")
    group.add_argument("-n", "--number", type=int, help="Number of problems to print (defaults to all)")
    group.add_argument("--start", "-s", type=int, default=0, help="Start at this index (inclusive, 0-indexed).")
    group.add_argument("--search", type=str, help="Only include problems that contain this string in the JSON")
    group.add_argument("-r", "--randomize", action="store_true", help="Randomize the order of the problems")
    group.add_argument(
        "--renumber",
        action="store_true",
        help="This only makes a difference with --randomize. If set, it renumbers problems from 1 to N. Otherwise, it keeps the original indexes.",
    )

    group = parser.add_argument_group("Printing Options")
    group.add_argument("-l", "--line-numbers", action="store_true", help="Print line numbers in the code blocks")
    group.add_argument("-w", "--width", type=int, help=f"Set the console width (defaults to {WIDTH})", default=WIDTH)
    group.add_argument(
        "--structure",
        action="store_true",
        help="Print the structure of the loaded data instead of printing the contents",
    )
    group.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw JSONL data instead of pretty printing it. This ignores the --parts flag.",
    )
    group.add_argument("--max-str-len", type=int, help="Maximum length of strings in raw mode.")

    group = parser.add_argument_group("Filtering")
    group.add_argument(
        "--manual-filter", action="store_true", help="Enter a manual filtering mode to select problems with y/n."
    )
    group.add_argument(
        "--filter-output", type=str, default="output.jsonl", help="Output file for filtered problems. Append-mode."
    )
    group.add_argument(
        "--file-output",
        type=str,
        help='Output file for filtered problems. Defaults to text but will write html if the file name ends with ".html". Overwrites.',
    )

    group = parser.add_argument_group("Miscellaneous Options")
    group.add_argument(
        "-b",
        "--bust-cache",
        action="store_true",
        help="Delete the cached temp file if it exists. Use this if you believe the file has changed on S3. Normally cached files expire after 1 day.",
    )

    args = parser.parse_args()

    if USE_RICH:
        global console
        if args.width and args.width != WIDTH:
            console = Console(force_terminal=True, width=args.width, record=True)
        else:
            console = Console(force_terminal=True, record=True)
    WIDTH = args.width

    if args.max_str_len:
        global MAX_PRINT_LEN
        MAX_PRINT_LEN = args.max_str_len

    if args.file == sys.stdin:
        file_contents = process_file(sys.stdin)
    elif args.file.lower().startswith("s3://"):
        raise NotImplementedError("S3 support is not yet implemented.")
    else:
        # Else read local file
        with open(args.file, "r") as file:
            file_contents = process_file(file)

    problems = file_contents
    lines = list(enumerate(problems.rstrip().split("\n")))
    total_num_problems = len(lines)
    print_text(f"Found {total_num_problems} problems")
    if args.randomize:
        random.shuffle(lines)
    if args.search:
        lines = [(num, line) for num, line in lines if args.search in line]
        print_text(f"After searching, found {total_num_problems} problems")
    if args.start:
        lines = lines[args.start :]
    if args.number:
        lines = lines[: args.number]

    if args.structure:
        print_header_1(f"JSON Structure (problem {lines[0][0]})")
        problem = json.loads(lines[0][1])
        if CHECK_TYPES:
            for typ in [CodeGenerationProblem, SolutionAttempt]:
                try:
                    # The JSON sometimes has "__type" keys that prevent parsing.
                    _ = typ(**remove_type_keys(problem))
                    print_text(f"Parsed as type: {typ.__name__}")
                except TypeError:
                    pass

        structure = print_json_structure(problem)
        print_code(structure, print_line_numbers=args.line_numbers, lexer="python")

    # The main case, iterate over problems
    else:
        problem_number = args.start if args.start else 0  # For the post-loop filtering summary
        filter_included = 0
        try:
            for selection_index, (original_index, line) in enumerate(lines):
                problem_number += 1
                if args.renumber:
                    print_header_1(f"Problem {selection_index + 1}")
                else:
                    print_header_1(f"Problem {original_index}")
                try:
                    problem = json.loads(line)
                except json.JSONDecodeError:
                    print_text(f"Problem on line {original_index} is not valid JSON")
                    continue
                if args.raw:
                    p = problem
                    if args.max_str_len:
                        p = truncate_strings(p, args.max_str_len)
                    print_code(json.dumps(p, indent=4), print_line_numbers=args.line_numbers, lexer="json")
                else:
                    print_problem(problem, parts=args.parts, print_line_numbers=args.line_numbers)
                if args.manual_filter:
                    include = input(f"Include this problem in {args.filter_output}? (y/N/q) ")
                    if include.lower() == "q":
                        raise KeyboardInterrupt()
                    if include.lower() == "y":
                        filter_included += 1
                        with open(args.filter_output, "a") as f:
                            f.write(json.dumps(problem) + "\n")
        except KeyboardInterrupt:
            if args.manual_filter:
                print_text("")
                print_header_1("Manual Filtering Statistics")
                print_text(f"Manually selected {filter_included} problems from {args.file}.")
                if not args.randomize:
                    print_code(
                        f"From lines: {args.start} to {problem_number - 1}. Next time use `--start {problem_number - 1}` to resume.",
                        lexer="markdown",
                    )
                print_code(
                    f"Output to file: {args.filter_output}, use `python scripts/pprint_problems.py {args.filter_output}` to view.",
                    lexer="markdown",
                )

    if args.file_output:
        if USE_RICH:
            if args.file_output.lower().endswith(".html"):
                console.save_html(args.file_output)
            else:
                console.save_text(args.file_output)
        else:
            raise NotImplementedError(
                "File output is only supported with the rich library. Use `> output.txt` instead."
            )


if __name__ == "__main__":
    main()
