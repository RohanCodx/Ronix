"""
parser.py
---------
The Parser turns a flat list of Tokens (from lexer.py) into an AST
(nodes defined in ast_nodes.py), using recursive-descent parsing.

Grammar for Ronix v0.1 (EBNF-ish):

    program     := statement* EOF

    statement   := let_stmt | show_stmt | when_stmt | repeat_stmt | while_stmt
                 | stop_stmt | use_stmt
    let_stmt    := LET IDENTIFIER ASSIGN expression NEWLINE
    show_stmt   := SHOW "(" expression ")" NEWLINE
    when_stmt   := WHEN expression ";" NEWLINE
                   block
                   ("otherwise" ";" NEWLINE block)?
                   "end" NEWLINE
    repeat_stmt := REPEAT expression? ";" NEWLINE block "end" NEWLINE
    while_stmt  := WHILE expression ";" NEWLINE block "end" NEWLINE
    stop_stmt   := STOP NEWLINE
    use_stmt    := USE IDENTIFIER NEWLINE

    block       := statement*     (stops at 'otherwise', 'end', or EOF)

    expression  := comparison
    comparison  := addition (("<" | ">" | "<=" | ">=" | "==" | "!=") addition)?
    addition    := term (("+" | "-") term)*
    term        := factor (("*" | "/") factor)*
    factor      := NUMBER
                 | STRING
                 | "on" | "off"
                 | "ask" "(" expression? ")"
                 | IDENTIFIER
                 | "-" factor
                 | "(" expression ")"

Blank lines (stray NEWLINE tokens between statements) are skipped,
so multiple statements can be separated by one or more newlines.

Note: comparisons are intentionally non-chaining (`a < b < c` is not
supported) to keep v0.1 simple and unambiguous. Since `show(...)` now
uses `(`/`)` (shared with grouping parens) rather than `<`/`>`, it can
safely hold a full expression, comparisons included.

`repeat` with no count expression loops forever until a `stop`
statement runs inside it (or inside a nested `when`). `stop` only
makes sense inside a `repeat`/`while` body; the Interpreter is what
enforces that (a `stop` outside any loop is a runtime error, not a
parse error, since parsing has no notion of "inside a loop").

Every block-opening header line (`when ...`, `otherwise`, `repeat ...`,
`while ...`) must end with `;` — this is Ronix's equivalent of
Python's trailing `:` on `if`/`while`/`for` lines. `end` closes the
block itself and does not take a `;`.

Each parsing method consumes exactly the tokens for the grammar rule
it represents and returns the corresponding AST node. Errors raise
RonixSyntaxError with precise line/column info pulled from the
offending token.
"""

from tokens import TokenType
from ast_nodes import (
    Program,
    LetStatement,
    ShowStatement,
    WhenStatement,
    RepeatStatement,
    WhileStatement,
    StopStatement,
    UseStatement,
    NumberLiteral,
    StringLiteral,
    BooleanLiteral,
    AskExpression,
    Identifier,
    BinaryOp,
    UnaryOp,
)
from errors import RonixSyntaxError


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # ------------------------------------------------------------------
    # Token stream helpers
    # ------------------------------------------------------------------

    def _current(self):
        return self.tokens[self.pos]

    def _peek_type(self):
        return self._current().type

    def _advance(self):
        token = self.tokens[self.pos]
        if token.type != TokenType.EOF:
            self.pos += 1
        return token

    def _check(self, type_: TokenType) -> bool:
        return self._peek_type() == type_

    def _match(self, *types) -> bool:
        if self._peek_type() in types:
            self._advance()
            return True
        return False

    def _expect(self, type_: TokenType, message: str):
        if self._check(type_):
            return self._advance()
        token = self._current()
        raise RonixSyntaxError(
            f"{message} (got {token.type.name} {token.value!r} instead)",
            token.line,
            token.column,
        )

    def _skip_newlines(self):
        while self._check(TokenType.NEWLINE):
            self._advance()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        statements = []
        self._skip_newlines()
        while not self._check(TokenType.EOF):
            statements.append(self._statement())
            self._skip_newlines()
        return Program(statements)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _statement(self):
        if self._check(TokenType.LET):
            return self._let_statement()
        if self._check(TokenType.SHOW):
            return self._show_statement()
        if self._check(TokenType.WHEN):
            return self._when_statement()
        if self._check(TokenType.REPEAT):
            return self._repeat_statement()
        if self._check(TokenType.WHILE):
            return self._while_statement()
        if self._check(TokenType.STOP):
            return self._stop_statement()
        if self._check(TokenType.USE):
            return self._use_statement()

        token = self._current()
        raise RonixSyntaxError(
            f"Expected a statement, got {token.type.name} {token.value!r}",
            token.line,
            token.column,
        )

    def _let_statement(self):
        let_token = self._expect(TokenType.LET, "Expected 'let'")
        name_token = self._expect(TokenType.IDENTIFIER, "Expected variable name after 'let'")
        self._expect(TokenType.ASSIGN, "Expected '=' after variable name")
        value = self._expression()
        self._end_statement()
        return LetStatement(name_token.value, value, let_token.line, let_token.column)

    def _show_statement(self):
        show_token = self._expect(TokenType.SHOW, "Expected 'show'")
        self._expect(TokenType.LPAREN, "Expected '(' after 'show'")
        value = self._expression()
        self._expect(TokenType.RPAREN, "Expected ')' to close 'show(...)'")
        self._end_statement()
        return ShowStatement(value, show_token.line, show_token.column)

    def _expect_header_end(self, context: str):
        """Every block-opening header (when/otherwise/repeat/while) must end
        with ';' — Ronix's equivalent of Python's trailing ':'."""
        self._expect(TokenType.SEMICOLON, f"Expected ';' after {context} (Ronix uses ';' where Python uses ':')")
        self._expect(TokenType.NEWLINE, "Expected end of line after ';'")

    def _when_statement(self):
        when_token = self._expect(TokenType.WHEN, "Expected 'when'")
        condition = self._expression()
        self._expect_header_end("'when' condition")

        then_branch = self._block(stop_types=(TokenType.OTHERWISE, TokenType.END))

        otherwise_branch = None
        if self._match(TokenType.OTHERWISE):
            self._expect_header_end("'otherwise'")
            otherwise_branch = self._block(stop_types=(TokenType.END,))

        self._expect(TokenType.END, "Expected 'end' to close 'when' block")
        self._end_statement()

        return WhenStatement(condition, then_branch, otherwise_branch, when_token.line, when_token.column)

    def _repeat_statement(self):
        repeat_token = self._expect(TokenType.REPEAT, "Expected 'repeat'")

        count = None
        if not self._check(TokenType.SEMICOLON):
            count = self._expression()
        self._expect_header_end("'repeat'")

        body = self._block(stop_types=(TokenType.END,))
        self._expect(TokenType.END, "Expected 'end' to close 'repeat' block")
        self._end_statement()

        return RepeatStatement(count, body, repeat_token.line, repeat_token.column)

    def _while_statement(self):
        while_token = self._expect(TokenType.WHILE, "Expected 'while'")
        condition = self._expression()
        self._expect_header_end("'while' condition")

        body = self._block(stop_types=(TokenType.END,))
        self._expect(TokenType.END, "Expected 'end' to close 'while' block")
        self._end_statement()

        return WhileStatement(condition, body, while_token.line, while_token.column)

    def _stop_statement(self):
        stop_token = self._expect(TokenType.STOP, "Expected 'stop'")
        self._end_statement()
        return StopStatement(stop_token.line, stop_token.column)

    def _use_statement(self):
        use_token = self._expect(TokenType.USE, "Expected 'use'")
        name_token = self._expect(TokenType.IDENTIFIER, "Expected a module name after 'use'")
        self._end_statement()
        return UseStatement(name_token.value, use_token.line, use_token.column)

    def _block(self, stop_types):
        """Parse statements until one of stop_types (or EOF) is reached.
        Does not consume the stopping token."""
        statements = []
        self._skip_newlines()
        while not self._check(TokenType.EOF) and self._peek_type() not in stop_types:
            statements.append(self._statement())
            self._skip_newlines()
        return statements

    def _end_statement(self):
        """A statement must end at a NEWLINE or EOF."""
        if self._check(TokenType.EOF):
            return
        self._expect(TokenType.NEWLINE, "Expected end of line after statement")

    # ------------------------------------------------------------------
    # Expressions (precedence climbing via recursive descent)
    # ------------------------------------------------------------------

    _COMPARISON_OPS = {
        TokenType.GT: ">",
        TokenType.LT: "<",
        TokenType.GTE: ">=",
        TokenType.LTE: "<=",
        TokenType.EQ: "==",
        TokenType.NEQ: "!=",
    }

    def _expression(self):
        return self._comparison()

    def _comparison(self):
        node = self._addition()
        if self._peek_type() in self._COMPARISON_OPS:
            op_token = self._advance()
            operator = self._COMPARISON_OPS[op_token.type]
            right = self._addition()
            node = BinaryOp(node, operator, right, op_token.line, op_token.column)
        return node

    def _addition(self):
        node = self._term()
        while self._check(TokenType.PLUS) or self._check(TokenType.MINUS):
            op_token = self._advance()
            operator = "+" if op_token.type == TokenType.PLUS else "-"
            right = self._term()
            node = BinaryOp(node, operator, right, op_token.line, op_token.column)
        return node

    def _term(self):
        node = self._factor()
        while self._check(TokenType.STAR) or self._check(TokenType.SLASH):
            op_token = self._advance()
            operator = "*" if op_token.type == TokenType.STAR else "/"
            right = self._factor()
            node = BinaryOp(node, operator, right, op_token.line, op_token.column)
        return node

    def _factor(self):
        token = self._current()

        if token.type == TokenType.MINUS:
            self._advance()
            operand = self._factor()
            return UnaryOp("-", operand, token.line, token.column)

        if token.type == TokenType.NUMBER:
            self._advance()
            return NumberLiteral(token.value, token.line, token.column)

        if token.type == TokenType.STRING:
            self._advance()
            return StringLiteral(token.value, token.line, token.column)

        if token.type == TokenType.ON:
            self._advance()
            return BooleanLiteral(True, token.line, token.column)

        if token.type == TokenType.OFF:
            self._advance()
            return BooleanLiteral(False, token.line, token.column)

        if token.type == TokenType.ASK:
            self._advance()
            self._expect(TokenType.LPAREN, "Expected '(' after 'ask'")
            prompt = None
            if not self._check(TokenType.RPAREN):
                prompt = self._expression()
            self._expect(TokenType.RPAREN, "Expected ')' to close 'ask(...)'")
            return AskExpression(prompt, token.line, token.column)

        if token.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(token.value, token.line, token.column)

        if token.type == TokenType.LPAREN:
            self._advance()
            node = self._expression()
            self._expect(TokenType.RPAREN, "Expected ')' to close expression")
            return node

        raise RonixSyntaxError(
            f"Unexpected token {token.type.name} {token.value!r} in expression",
            token.line,
            token.column,
        )
