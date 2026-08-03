"""
lexer.py
--------
The Lexer (tokenizer) turns raw Ronix source code — a flat string —
into a linear stream of Token objects, as defined in tokens.py.

It works by scanning the source one character at a time, deciding
what kind of token each character (or group of characters) starts,
and emitting the appropriate Token. It also tracks line/column
position for every token so later stages can produce precise error
messages.

Responsibilities:
  - Skip whitespace (except newlines, which are significant —
    they separate statements).
  - Skip comments (# ... to end of line).
  - Recognize number literals (integers and floats).
  - Recognize string literals ("...") with basic escape sequences.
  - Recognize identifiers and classify them as keywords or plain
    identifiers using the KEYWORDS table.
  - Recognize operators and punctuation.
  - Raise a RonixSyntaxError on any unrecognized character.
"""

from tokens import Token, TokenType, KEYWORDS
from errors import RonixSyntaxError


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []

    # ------------------------------------------------------------------
    # Low-level character helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0):
        """Return the character at pos+offset without consuming it, or None at EOF."""
        idx = self.pos + offset
        if idx >= len(self.source):
            return None
        return self.source[idx]

    def _advance(self):
        """Consume and return the current character, updating line/column."""
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _match(self, expected: str) -> bool:
        """If the current character equals expected, consume it and return True."""
        if self._peek() == expected:
            self._advance()
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self):
        """Scan the entire source and return the full list of tokens, ending in EOF."""
        while self.pos < len(self.source):
            self._scan_token()
        self._add_token(TokenType.EOF, None, self.line, self.column)
        return self.tokens

    # ------------------------------------------------------------------
    # Core scanning logic
    # ------------------------------------------------------------------

    def _scan_token(self):
        start_line = self.line
        start_col = self.column
        ch = self._advance()

        # Newlines are significant: they terminate statements.
        if ch == "\n":
            self._add_token(TokenType.NEWLINE, "\\n", start_line, start_col)
            return

        # Skip other whitespace.
        if ch in (" ", "\t", "\r"):
            return

        # Comments: '#' to end of line.
        if ch == "#":
            while self._peek() is not None and self._peek() != "\n":
                self._advance()
            return

        # String literals.
        if ch == '"':
            self._read_string(start_line, start_col)
            return

        # Numbers.
        if ch.isdigit():
            self._read_number(ch, start_line, start_col)
            return

        # Identifiers / keywords.
        if ch.isalpha() or ch == "_":
            self._read_identifier(ch, start_line, start_col)
            return

        # Operators and punctuation.
        if ch == "+":
            self._add_token(TokenType.PLUS, "+", start_line, start_col)
            return
        if ch == "-":
            self._add_token(TokenType.MINUS, "-", start_line, start_col)
            return
        if ch == "*":
            self._add_token(TokenType.STAR, "*", start_line, start_col)
            return
        if ch == "/":
            self._add_token(TokenType.SLASH, "/", start_line, start_col)
            return
        if ch == "=":
            if self._match("="):
                self._add_token(TokenType.EQ, "==", start_line, start_col)
            else:
                self._add_token(TokenType.ASSIGN, "=", start_line, start_col)
            return
        if ch == "!":
            if self._match("="):
                self._add_token(TokenType.NEQ, "!=", start_line, start_col)
                return
            raise RonixSyntaxError("Unexpected character '!' (did you mean '!='?)", start_line, start_col)
        if ch == "<":
            if self._match("="):
                self._add_token(TokenType.LTE, "<=", start_line, start_col)
            else:
                self._add_token(TokenType.LT, "<", start_line, start_col)
            return
        if ch == ">":
            if self._match("="):
                self._add_token(TokenType.GTE, ">=", start_line, start_col)
            else:
                self._add_token(TokenType.GT, ">", start_line, start_col)
            return
        if ch == "(":
            self._add_token(TokenType.LPAREN, "(", start_line, start_col)
            return
        if ch == ")":
            self._add_token(TokenType.RPAREN, ")", start_line, start_col)
            return

        raise RonixSyntaxError(f"Unexpected character '{ch}'", start_line, start_col)

    def _read_string(self, start_line, start_col):
        chars = []
        while True:
            ch = self._peek()
            if ch is None:
                raise RonixSyntaxError("Unterminated string literal", start_line, start_col)
            if ch == '"':
                self._advance()
                break
            if ch == "\n":
                raise RonixSyntaxError("Unterminated string literal", start_line, start_col)
            if ch == "\\":
                self._advance()
                escaped = self._peek()
                if escaped is None:
                    raise RonixSyntaxError("Unterminated string literal", start_line, start_col)
                self._advance()
                chars.append(self._unescape(escaped, start_line, start_col))
                continue
            chars.append(self._advance())
        self._add_token(TokenType.STRING, "".join(chars), start_line, start_col)

    @staticmethod
    def _unescape(escaped: str, line: int, col: int) -> str:
        mapping = {
            "n": "\n",
            "t": "\t",
            '"': '"',
            "\\": "\\",
        }
        if escaped not in mapping:
            raise RonixSyntaxError(f"Invalid escape sequence '\\{escaped}'", line, col)
        return mapping[escaped]

    def _read_number(self, first_digit, start_line, start_col):
        digits = [first_digit]
        is_float = False
        while self._peek() is not None and self._peek().isdigit():
            digits.append(self._advance())
        if self._peek() == "." and self._peek(1) is not None and self._peek(1).isdigit():
            is_float = True
            digits.append(self._advance())  # consume '.'
            while self._peek() is not None and self._peek().isdigit():
                digits.append(self._advance())
        text = "".join(digits)
        value = float(text) if is_float else int(text)
        self._add_token(TokenType.NUMBER, value, start_line, start_col)

    def _read_identifier(self, first_char, start_line, start_col):
        chars = [first_char]
        while self._peek() is not None and (self._peek().isalnum() or self._peek() == "_"):
            chars.append(self._advance())
        text = "".join(chars)
        token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
        self._add_token(token_type, text, start_line, start_col)

    # ------------------------------------------------------------------
    # Token emission
    # ------------------------------------------------------------------

    def _add_token(self, type_: TokenType, value, line: int, column: int):
        self.tokens.append(Token(type_, value, line, column))
