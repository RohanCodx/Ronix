"""
ast_nodes.py
------------
Defines the Abstract Syntax Tree (AST) node types for Ronix.

While the Lexer turns source text into a flat stream of tokens, the
Parser turns that stream into a tree that represents the *structure*
and *meaning* of the program — which expressions belong to which
statements, how operators nest, etc.

Each node is a simple, explicit dataclass-like object. Keeping them
dumb (no logic, just data) means the same tree can be reused by
multiple consumers later (interpreter, formatter, optimizer, etc.)
without coupling them to how the tree is evaluated.

Node hierarchy for v0.1:

    Program
      statements: list[Statement]

    Statement
      LetStatement(name, value)
      ShowStatement(value)

    Expression
      NumberLiteral(value)
      StringLiteral(value)
      Identifier(name)
      BinaryOp(left, operator, right)
      UnaryOp(operator, operand)
"""


class Node:
    """Base class for every AST node. Carries source position for error reporting."""

    def __init__(self, line: int = None, column: int = None):
        self.line = line
        self.column = column


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------

class Program(Node):
    """The root node: an ordered list of statements."""

    def __init__(self, statements):
        super().__init__(line=1, column=1)
        self.statements = statements

    def __repr__(self):
        return f"Program({self.statements!r})"


# ----------------------------------------------------------------------
# Statements
# ----------------------------------------------------------------------

class LetStatement(Node):
    """Variable declaration/assignment: `let <name> = <value>`"""

    def __init__(self, name: str, value, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.value = value

    def __repr__(self):
        return f"LetStatement(name={self.name!r}, value={self.value!r})"


class ShowStatement(Node):
    """Output statement: `show<<value>>`"""

    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self):
        return f"ShowStatement(value={self.value!r})"


class WhenStatement(Node):
    """Conditional statement:

        when <condition>
            <then_branch statements>
        otherwise
            <otherwise_branch statements>
        end

    `otherwise_branch` is `None` when no `otherwise` clause is present.
    """

    def __init__(self, condition, then_branch, otherwise_branch=None, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.then_branch = then_branch
        self.otherwise_branch = otherwise_branch

    def __repr__(self):
        return (
            f"WhenStatement(condition={self.condition!r}, "
            f"then={self.then_branch!r}, otherwise={self.otherwise_branch!r})"
        )


class RepeatStatement(Node):
    """A counted or infinite loop:

        repeat 10
            <body>
        end

        repeat            (no count -> loops forever, until 'stop')
            <body>
        end

    `count` is `None` for the bare infinite form.
    """

    def __init__(self, count, body, line=None, column=None):
        super().__init__(line, column)
        self.count = count
        self.body = body

    def __repr__(self):
        return f"RepeatStatement(count={self.count!r}, body={self.body!r})"


class WhileStatement(Node):
    """A conditional loop:

        while <condition>
            <body>
        end
    """

    def __init__(self, condition, body, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileStatement(condition={self.condition!r}, body={self.body!r})"


class StopStatement(Node):
    """Breaks out of the nearest enclosing 'repeat' or 'while' loop."""

    def __repr__(self):
        return "StopStatement()"


class UseStatement(Node):
    """Imports a stdlib module by name: `use <name>`.

    The Interpreter loads and runs stdlib/<name>.rx into the current
    global scope.
    """

    def __init__(self, name: str, line=None, column=None):
        super().__init__(line, column)
        self.name = name

    def __repr__(self):
        return f"UseStatement(name={self.name!r})"


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------

class NumberLiteral(Node):
    """A numeric literal, e.g. 10 or 3.14"""

    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self):
        return f"NumberLiteral({self.value!r})"


class StringLiteral(Node):
    """A string literal, e.g. "Hello" """

    def __init__(self, value: str, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self):
        return f"StringLiteral({self.value!r})"


class BooleanLiteral(Node):
    """A boolean literal: `on` (true) or `off` (false)."""

    def __init__(self, value: bool, line=None, column=None):
        super().__init__(line, column)
        self.value = value

    def __repr__(self):
        return f"BooleanLiteral({self.value!r})"


class AskExpression(Node):
    """Reads a line of user input, as an expression: `ask("Prompt: ")`.

    Always evaluates to a string. `prompt` is `None` for the bare
    `ask()` form (no prompt text printed before reading).
    """

    def __init__(self, prompt, line=None, column=None):
        super().__init__(line, column)
        self.prompt = prompt

    def __repr__(self):
        return f"AskExpression(prompt={self.prompt!r})"


class Identifier(Node):
    """A reference to a variable by name, e.g. `x`"""

    def __init__(self, name: str, line=None, column=None):
        super().__init__(line, column)
        self.name = name

    def __repr__(self):
        return f"Identifier({self.name!r})"


class BinaryOp(Node):
    """A binary operation, e.g. `x + y`, `10 * 2`"""

    def __init__(self, left, operator: str, right, line=None, column=None):
        super().__init__(line, column)
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.left!r} {self.operator} {self.right!r})"


class UnaryOp(Node):
    """A unary operation, e.g. `-x`"""

    def __init__(self, operator: str, operand, line=None, column=None):
        super().__init__(line, column)
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.operator}{self.operand!r})"
