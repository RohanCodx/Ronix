"""
errors.py
---------
Defines Ronix's exception hierarchy.

Every stage of the pipeline (Lexer, Parser, Interpreter) raises a
subclass of RonixError instead of a bare Python exception. This
gives us:

  - A single place to catch "any Ronix error" (in main.py) and print
    a clean, user-facing message instead of a raw Python traceback.
  - Consistent line/column reporting across all stages.
  - Room to grow: future versions can add more specific subclasses
    (e.g. RonixImportError, RonixIndexError) without changing how
    they're caught and displayed.
"""


class RonixError(Exception):
    """Base class for every error Ronix can raise."""

    #: short label shown in the formatted error message, overridden by subclasses
    label = "Error"

    def __init__(self, message: str, line: int = None, column: int = None):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self.format())

    def format(self) -> str:
        if self.line is not None and self.column is not None:
            return f"{self.label} at line {self.line}, column {self.column}: {self.message}"
        if self.line is not None:
            return f"{self.label} at line {self.line}: {self.message}"
        return f"{self.label}: {self.message}"


class RonixSyntaxError(RonixError):
    """Raised by the Lexer or Parser when source code doesn't follow Ronix grammar."""

    label = "Syntax Error"


class RonixRuntimeError(RonixError):
    """Raised by the Interpreter when a program is syntactically valid but fails at runtime
    (e.g. undefined variable, division by zero, type mismatch)."""

    label = "Runtime Error"


class RonixNameError(RonixRuntimeError):
    """Raised when referencing a variable that hasn't been defined."""

    label = "Name Error"


class RonixTypeError(RonixRuntimeError):
    """Raised when an operation is applied to incompatible types."""

    label = "Type Error"


class RonixZeroDivisionError(RonixRuntimeError):
    """Raised on division by zero."""

    label = "Division Error"
