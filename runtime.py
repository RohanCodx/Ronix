"""
runtime.py
----------
The Interpreter walks the AST produced by parser.py and executes it
directly (a "tree-walking interpreter" — no separate bytecode stage
in v0.1).

It owns:
  - Environment: a scope object mapping variable names to values.
  - Interpreter: dispatches on node type and evaluates/executes it.

Design notes:
  - Numbers are represented as Python int/float directly — no need
    for a custom wrapper type yet.
  - Strings are represented as Python str directly.
  - `+` supports both numeric addition and string concatenation.
    Mixing a number and a string with `+` raises a RonixTypeError
    (explicit is better than silently coercing).
  - `- * /` are numeric-only.
  - Division by zero raises RonixZeroDivisionError.
  - `show<...>` prints the evaluated value's Ronix-appropriate
    string form (numbers print without unnecessary float noise).
"""

from ast_nodes import (
    Program,
    LetStatement,
    ShowStatement,
    WhenStatement,
    NumberLiteral,
    StringLiteral,
    BooleanLiteral,
    Identifier,
    BinaryOp,
    UnaryOp,
)
from errors import RonixNameError, RonixTypeError, RonixZeroDivisionError, RonixRuntimeError


class Environment:
    """A single scope of variable bindings. v0.1 has only a global scope,
    but this class exists on its own so nested scopes (functions, blocks)
    can be layered on top in later versions without touching the rest
    of the interpreter."""

    def __init__(self, parent: "Environment" = None):
        self.parent = parent
        self.values = {}

    def define(self, name: str, value):
        self.values[name] = value

    def get(self, name: str, line=None, column=None):
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name, line, column)
        raise RonixNameError(f"Undefined variable '{name}'", line, column)

    def assign(self, name: str, value, line=None, column=None):
        if name in self.values:
            self.values[name] = value
            return
        if self.parent is not None:
            self.parent.assign(name, value, line, column)
            return
        raise RonixNameError(f"Cannot assign to undefined variable '{name}'", line, column)


class Interpreter:
    def __init__(self):
        self.globals = Environment()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(self, program: Program):
        for statement in program.statements:
            self._execute(statement)

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _execute(self, node):
        method_name = f"_exec_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is None:
            raise RonixRuntimeError(f"No executor for statement type {type(node).__name__}", node.line, node.column)
        return method(node)

    def _exec_LetStatement(self, node: LetStatement):
        value = self._evaluate(node.value)
        self.globals.define(node.name, value)

    def _exec_ShowStatement(self, node: ShowStatement):
        value = self._evaluate(node.value)
        print(self._stringify(value))

    def _exec_WhenStatement(self, node: WhenStatement):
        condition_value = self._evaluate(node.condition)
        if self._is_truthy(condition_value):
            for statement in node.then_branch:
                self._execute(statement)
        elif node.otherwise_branch is not None:
            for statement in node.otherwise_branch:
                self._execute(statement)

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, node):
        method_name = f"_eval_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is None:
            raise RonixRuntimeError(f"No evaluator for expression type {type(node).__name__}", node.line, node.column)
        return method(node)

    def _eval_NumberLiteral(self, node: NumberLiteral):
        return node.value

    def _eval_StringLiteral(self, node: StringLiteral):
        return node.value

    def _eval_BooleanLiteral(self, node: BooleanLiteral):
        return node.value

    def _eval_Identifier(self, node: Identifier):
        return self.globals.get(node.name, node.line, node.column)

    def _eval_UnaryOp(self, node: UnaryOp):
        operand = self._evaluate(node.operand)
        if node.operator == "-":
            if not isinstance(operand, (int, float)) or isinstance(operand, bool):
                raise RonixTypeError(
                    f"Unary '-' requires a number, got {self._type_name(operand)}",
                    node.line,
                    node.column,
                )
            return -operand
        raise RonixRuntimeError(f"Unknown unary operator '{node.operator}'", node.line, node.column)

    def _eval_BinaryOp(self, node: BinaryOp):
        left = self._evaluate(node.left)
        right = self._evaluate(node.right)
        op = node.operator

        if op == "+":
            return self._add(left, right, node)
        if op == "-":
            self._require_numbers(left, right, node, "-")
            return left - right
        if op == "*":
            self._require_numbers(left, right, node, "*")
            return left * right
        if op == "/":
            self._require_numbers(left, right, node, "/")
            if right == 0:
                raise RonixZeroDivisionError("Division by zero", node.line, node.column)
            result = left / right
            # Keep clean integers as ints when both operands were ints and it divides evenly.
            if isinstance(left, int) and isinstance(right, int) and left % right == 0:
                return left // right
            return result

        if op in ("<", ">", "<=", ">="):
            self._require_numbers(left, right, node, op)
            if op == "<":
                return left < right
            if op == ">":
                return left > right
            if op == "<=":
                return left <= right
            return left >= right

        if op == "==":
            return self._values_equal(left, right)
        if op == "!=":
            return not self._values_equal(left, right)

        raise RonixRuntimeError(f"Unknown binary operator '{op}'", node.line, node.column)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add(self, left, right, node):
        if self._is_number(left) and self._is_number(right):
            return left + right
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise RonixTypeError(
            f"Cannot apply '+' to {self._type_name(left)} and {self._type_name(right)}",
            node.line,
            node.column,
        )

    def _require_numbers(self, left, right, node, op):
        if not (self._is_number(left) and self._is_number(right)):
            raise RonixTypeError(
                f"Cannot apply '{op}' to {self._type_name(left)} and {self._type_name(right)}",
                node.line,
                node.column,
            )

    @staticmethod
    def _values_equal(left, right) -> bool:
        # Numbers compare across int/float; strings and booleans compare by
        # exact type+value; mismatched types (e.g. number vs string) are
        # simply unequal rather than a type error.
        if Interpreter._is_number(left) and Interpreter._is_number(right):
            return left == right
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        if type(left) is type(right):
            return left == right
        return False

    @staticmethod
    def _is_truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        if Interpreter._is_number(value):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        return True

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _type_name(value) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        return type(value).__name__

    @staticmethod
    def _stringify(value) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
