"""
main.py
-------
The entry point for the Ronix interpreter.

Usage:
    python main.py <file.rx>

This is what will eventually be wired up to the `ronix` command
(e.g. via a console_scripts entry point in packaging, added later).
For now it can be run directly.

Pipeline:
    source text -> Lexer -> tokens -> Parser -> AST -> Interpreter -> output

Any RonixError raised at any stage is caught here and printed as a
clean, single-line diagnostic (no raw Python traceback) including
the source filename and line/column. Unexpected internal errors are
still surfaced, but clearly labeled as internal so they're not
confused with user code mistakes.
"""

import sys

from lexer import Lexer
from parser import Parser
from runtime import Interpreter
from errors import RonixError


def run_source(source: str) -> None:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    Interpreter().interpret(program)


def main(argv):
    if len(argv) != 2:
        print("Usage: ronix <file.rx>", file=sys.stderr)
        return 1

    path = argv[1]
    if not path.endswith(".rx"):
        print(f"Error: Ronix source files must end with '.rx' (got '{path}')", file=sys.stderr)
        return 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: '{path}'", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error: could not read '{path}': {e}", file=sys.stderr)
        return 1

    try:
        run_source(source)
    except RonixError as e:
        print(f"{path}: {e.format()}", file=sys.stderr)
        return 1
    except Exception as e:  # pragma: no cover - safety net for genuinely unexpected bugs
        print(f"{path}: Internal Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
