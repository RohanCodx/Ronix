"""
tokens.py
----------
Defines the vocabulary of the Ronix language: every kind of token the
Lexer is allowed to produce, and the Token data structure itself.

Every later stage of the interpreter (Lexer -> Parser -> Runtime)
depends on this file, so it is kept small, explicit, and easy to
extend as new language features are added in future versions.
"""

from enum import Enum, auto


class TokenType(Enum):
    """
    Enumerates every category of token in Ronix v0.1.

    Grouped by purpose:
      - Literals
      - Identifiers
      - Keywords
      - Operators
      - Punctuation
      - Structural / control tokens
    """

    # ----- Literals -----
    NUMBER = auto()      # e.g. 10, 3.14
    STRING = auto()       # e.g. "Hello"

    # ----- Identifiers -----
    IDENTIFIER = auto()   # e.g. x, name, total

    # ----- Keywords -----
    LET = auto()          # 'let'
    SHOW = auto()          # 'show'
    WHEN = auto()          # 'when'
    OTHERWISE = auto()      # 'otherwise'
    END = auto()             # 'end'
    YES = auto()              # 'yes'  (boolean literal, true)
    NO = auto()                # 'no'   (boolean literal, false)

    # ----- Operators -----
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()           # *
    SLASH = auto()          # /
    ASSIGN = auto()         # =

    # ----- Comparison operators -----
    GT = auto()              # >
    LT = auto()              # <
    GTE = auto()              # >=
    LTE = auto()              # <=
    EQ = auto()                # ==
    NEQ = auto()                # !=

    # ----- Punctuation -----
    LPAREN = auto()          # (   (grouped expressions AND show(...))
    RPAREN = auto()          # )

    # ----- Structural / control tokens -----
    NEWLINE = auto()          # marks the end of a statement
    EOF = auto()               # marks the end of the token stream


# Reserved words map directly to keyword token types.
# The Lexer checks every identifier-like word against this table
# to decide whether it's a keyword or a plain identifier.
KEYWORDS = {
    "let": TokenType.LET,
    "show": TokenType.SHOW,
    "when": TokenType.WHEN,
    "otherwise": TokenType.OTHERWISE,
    "end": TokenType.END,
    "yes": TokenType.YES,
    "no": TokenType.NO,
}


class Token:
    """
    A single token produced by the Lexer.

    Attributes:
        type   (TokenType): the category of this token.
        value  (Any):        the literal value/text this token represents.
        line   (int):         1-indexed line number where the token starts.
        column (int):        1-indexed column number where the token starts.
    """

    __slots__ = ("type", "value", "line", "column")

    def __init__(self, type_: TokenType, value, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return (
            self.type == other.type
            and self.value == other.value
            and self.line == other.line
            and self.column == other.column
        )
