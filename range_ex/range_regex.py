from __future__ import annotations

import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional, Union

DecimalLike = Union[int, float, Decimal, str]


class Node(ABC):
    @abstractmethod
    def render(self, capturing: bool = False) -> str:
        raise NotImplementedError

    def normalize(self) -> "Node":
        return self


@dataclass(frozen=True)
class Literal(Node):
    text: str

    def render(self, capturing: bool = False) -> str:
        _ = capturing
        return re.escape(self.text)


@dataclass(frozen=True)
class DigitRange(Node):
    start: int
    end: int

    def render(self, capturing: bool = False) -> str:
        _ = capturing
        if self.start == self.end:
            return str(self.start)
        if self.start == 0 and self.end == 9:
            return r"\d"
        return f"[{self.start}-{self.end}]"

    def normalize(self) -> "Node":
        if self.start == self.end:
            return Literal(str(self.start))
        return self


@dataclass(frozen=True)
class OptionalNode(Node):
    node: Node

    def render(self, capturing: bool = False) -> str:
        open_group = "(" if capturing else "(?:"
        return f"{open_group}{self.node.render(capturing=capturing)})?"

    def normalize(self) -> "Node":
        return OptionalNode(self.node.normalize())


@dataclass(frozen=True)
class ZeroOrMore(Node):
    node: Node

    def render(self, capturing: bool = False) -> str:
        open_group = "(" if capturing else "(?:"
        return f"{open_group}{self.node.render(capturing=capturing)})*"

    def normalize(self) -> "Node":
        return ZeroOrMore(self.node.normalize())


@dataclass(frozen=True)
class Sequence(Node):
    parts: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        return "".join(part.render(capturing=capturing) for part in self.parts)

    def normalize(self) -> "Node":
        normalized_parts = [part.normalize() for part in self.parts]
        merged_parts: list[Node] = []
        for part in normalized_parts:
            if (
                merged_parts
                and isinstance(merged_parts[-1], Literal)
                and isinstance(part, Literal)
            ):
                merged_parts[-1] = Literal(merged_parts[-1].text + part.text)
            else:
                merged_parts.append(part)
        return Sequence(tuple(merged_parts))


@dataclass(frozen=True)
class Alternation(Node):
    options: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        open_group = "(" if capturing else "(?:"
        return f"{open_group}{'|'.join(option.render(capturing=capturing) for option in self.options)})"

    def normalize(self) -> Node:
        normalized_options = [option.normalize() for option in self.options]
        seen: set[str] = set()
        unique_options: list[Node] = []
        for option in normalized_options:
            key = option.render()
            if key not in seen:
                seen.add(key)
                unique_options.append(option)
        return Alternation(tuple(unique_options))


def _literal_parts(text: str) -> list[Node]:
    return [Literal(c) for c in text]


def _any_digits(count: int) -> list[Node]:
    return [DigitRange(0, 9) for _ in range(count)]


def _as_parts(node: Node) -> list[Node]:
    if isinstance(node, Sequence):
        return list(node.parts)
    return [node]


def _options(node: Node) -> tuple[Node, ...]:
    if isinstance(node, Alternation):
        return node.options
    return (node,)


def _alt(*nodes: Node) -> Node:
    if len(nodes) == 1:
        return nodes[0]
    return Alternation(tuple(nodes))


def _seq(*nodes: Node) -> Node:
    flat: list[Node] = []
    for node in nodes:
        if isinstance(node, Sequence):
            flat.extend(node.parts)
        else:
            flat.append(node)
    if len(flat) == 1:
        return flat[0]
    return Sequence(tuple(flat))


def __compute_numerical_range_ast(
    str_a: str, str_b: str, start_parts: Optional[list[Node]] = None
) -> list[Node]:
    """
    Build regex AST sequences for an inclusive integer range with equal-width endpoints.

    Assumes:
    - int(str_a) <= int(str_b)
    - len(str_a) == len(str_b)
    """
    if len(str_a) != len(str_b):
        raise Exception(
            f"The input numbers ({str_a}, {str_b}) should have equal number of digits"
        )

    parts = list(start_parts or [])
    str_len = len(str_a)

    if str_a == str_b:
        return [_seq(*parts, *_literal_parts(str_a))]
    if str_len == 1:
        return [_seq(*parts, DigitRange(int(str_a), int(str_b)))]

    check_equal = -1
    for i in range(str_len):
        if str_a[i] == str_b[i]:
            check_equal += 1
        else:
            break
    if check_equal != -1:
        return __compute_numerical_range_ast(
            str_a[check_equal + 1 :],
            str_b[check_equal + 1 :],
            start_parts=parts + _literal_parts(str_a[: check_equal + 1]),
        )

    patterns: list[Node] = []
    intermediate_range = list(range(int(str_a[0]) + 1, int(str_b[0])))
    if intermediate_range:
        patterns.append(
            _seq(
                *parts,
                DigitRange(intermediate_range[0], intermediate_range[-1]),
                *_any_digits(str_len - 1),
            )
        )

    for loop_counter in range(str_len - 1):
        prefix = parts + _literal_parts(str_a[: loop_counter + 1])
        if loop_counter == str_len - 2:
            patterns.append(_seq(*prefix, DigitRange(int(str_a[-1]), 9)))
        elif str_a[loop_counter + 1] != "9":
            patterns.append(
                _seq(
                    *prefix,
                    DigitRange(int(str_a[loop_counter + 1]) + 1, 9),
                    *_any_digits(str_len - 2 - loop_counter),
                )
            )

    for loop_counter in range(str_len - 1):
        prefix = parts + _literal_parts(str_b[: loop_counter + 1])
        if loop_counter == str_len - 2:
            patterns.append(_seq(*prefix, DigitRange(0, int(str_b[-1]))))
        elif str_b[loop_counter + 1] != "0":
            patterns.append(
                _seq(
                    *prefix,
                    DigitRange(0, int(str_b[loop_counter + 1]) - 1),
                    *_any_digits(str_len - 2 - loop_counter),
                )
            )

    return patterns


def __range_splitter(a, b):
    """
    Split an inclusive integer range into sub-ranges with equal digit width.

    Example:
        input: -15 and 256
        output: [(1, 9, '-'),
                 (10, 15, '-'),
                 (0, 9, ''),
                 (10, 99, ''),
                 (100, 256, '')]

    The third element is the sign prefix:
    - '' for positive values
    - '-' for negative values
    """
    ranges = []
    if b < 0:
        sign = "-"
        a, b = abs(a), abs(b)
        a, b = (a, b) if a < b else (b, a)
    elif a >= 0:
        sign = ""
    else:
        ranges.extend(__range_splitter(a, -1))
        ranges.extend(__range_splitter(0, b))
        return ranges

    str_a = str(a)
    str_b = str(b)
    len_str_a = len(str_a)
    len_str_b = len(str_b)

    if len_str_a == len_str_b:
        ranges.append((a, b, sign))
    else:
        for loop_counter in range(len_str_a, len_str_b + 1):
            if loop_counter == len_str_a:
                ranges.append((a, int("".join(["9"] * loop_counter)), sign))
            elif loop_counter == len_str_b:
                ranges.append((ranges[-1][1] + 1, b, sign))
            else:
                ranges.append(
                    (ranges[-1][1] + 1, int("".join(["9"] * loop_counter)), sign)
                )

    return ranges


def _range_ast(a: int, b: int) -> Node:
    a, b = (a, b) if a < b else (b, a)
    ranges = __range_splitter(a, b)
    patterns: list[Node] = []
    for start, end, sign in ranges:
        prefix = [Literal("-")] if sign else []
        patterns.extend(
            __compute_numerical_range_ast(str(start), str(end), start_parts=prefix)
        )
    return _alt(*patterns)


def _float_range_ast(a: DecimalLike, b: DecimalLike) -> Node:
    """
    Generate a regex AST that matches decimal numbers in the inclusive range [a, b].
    """
    a_decimal = Decimal(str(a))
    b_decimal = Decimal(str(b))
    a_decimal, b_decimal = (
        (a_decimal, b_decimal) if a_decimal < b_decimal else (b_decimal, a_decimal)
    )

    a_string = format(a_decimal, "f")
    b_string = format(b_decimal, "f")
    if "." not in a_string:
        a_string = f"{a_string}.0"
    if "." not in b_string:
        b_string = f"{b_string}.0"

    num_of_decimal_in_a = len(a_string) - (a_string.find(".") + 1)
    num_of_decimal_in_b = len(b_string) - (b_string.find(".") + 1)
    max_num_decimal = max(num_of_decimal_in_a, num_of_decimal_in_b)

    a_digits, b_digits = (
        "".join(c for c in a_string if c != "."),
        "".join(c for c in b_string if c != "."),
    )
    if len(a_digits) < len(b_digits):
        a_digits = a_digits + ("0" * (max_num_decimal - num_of_decimal_in_a))
    else:
        b_digits = b_digits + ("0" * (max_num_decimal - num_of_decimal_in_b))

    a_int, b_int = int(a_digits), int(b_digits)
    a_int, b_int = (a_int, b_int) if a_int < b_int else (b_int, a_int)

    scaled_int_ast = _range_ast(a_int, b_int)
    new_patterns: list[Node] = []
    for pattern_node in _options(scaled_int_ast):
        parts = _as_parts(pattern_node)
        sign_parts: list[Node] = []
        if parts and isinstance(parts[0], Literal) and parts[0].text.startswith("-"):
            sign_parts = [Literal("-")]
            leading_rest = parts[0].text[1:]
            parts = _literal_parts(leading_rest) + parts[1:]

        if len(parts) < max_num_decimal:
            parts = [Literal("0")] * (max_num_decimal - len(parts)) + parts

        split_index = len(parts) - max_num_decimal
        non_fractional = parts[:split_index]
        fractional = parts[split_index:]

        required_fractional: list[Node] = [fractional[0]]
        required_fractional.extend(OptionalNode(token) for token in fractional[1:])

        if non_fractional:
            non_fractional_parts = non_fractional
        else:
            non_fractional_parts = [OptionalNode(Literal("0"))]

        new_patterns.append(
            _seq(
                *sign_parts,
                *non_fractional_parts,
                Literal("."),
                *required_fractional,
                ZeroOrMore(DigitRange(0, 9)),
            )
        )

    return _alt(*new_patterns)


def _to_decimal(value: DecimalLike, name: str) -> Decimal:
    _ = name
    return Decimal(str(value))


def _zero() -> Literal:
    return Literal("0")


def _integer_unbounded_ast() -> Node:
    return _alt(_negative_unbounded_ast(), _positive_unbounded_ast(), _zero())

def _positive_with_min_digits_ast(extra_digits: int) -> Node:
    return _seq(
        DigitRange(1, 9),
        *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
        ZeroOrMore(DigitRange(0, 9)),
    )

def _positive_unbounded_ast() -> Node:
    return _positive_with_min_digits_ast(0)


def _negative_with_min_digits_ast(extra_digits: int) -> Node:
    return _seq(Literal("-"), _positive_with_min_digits_ast(extra_digits))


def _negative_unbounded_ast() -> Node:
    return _negative_with_min_digits_ast(0)


def _strict_decimal_unbounded_ast() -> Node:
    # Accept these strict decimal forms:
    # -?I.F+  (e.g. 1.2)
    # -?I.    (e.g. 1.)
    # -?.F+   (e.g. .2)
    # with I in {0, [1-9]\d*}
    integer_part = _alt(Literal("0"), _positive_unbounded_ast())
    return _alt(
        _seq(
            OptionalNode(Literal("-")),
            integer_part,
            Literal("."),
            DigitRange(0, 9),
            ZeroOrMore(DigitRange(0, 9)),
        ),
        _seq(OptionalNode(Literal("-")), integer_part, Literal(".")),
        _seq(
            OptionalNode(Literal("-")),
            Literal("."),
            DigitRange(0, 9),
            ZeroOrMore(DigitRange(0, 9)),
        ),
    )


def _negative_strict_decimal_with_min_int_digits_ast(extra_digits: int) -> Node:
    return _seq(Literal("-"), _positive_strict_decimal_with_min_int_digits_ast(extra_digits))


def _positive_strict_decimal_with_min_int_digits_ast(extra_digits: int) -> Node:
    return _seq(
        DigitRange(1, 9),
        *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
        ZeroOrMore(DigitRange(0, 9)),
        Literal("."),
        DigitRange(0, 9),
        ZeroOrMore(DigitRange(0, 9)),
    )


def _range_from_bounds_ast(minimum: Optional[int], maximum: Optional[int]) -> Node:
    if minimum is not None:
        minimum = operator.index(minimum)
    if maximum is not None:
        maximum = operator.index(maximum)

    if minimum is None and maximum is None:
        return _integer_unbounded_ast()
    if minimum is None:
        assert maximum is not None
        if maximum == 0:
            return _alt(_negative_unbounded_ast(), Literal("0"))
        if maximum > 0:
            upperbound_ast = _range_ast(0, maximum)
            return _alt(_negative_unbounded_ast(), upperbound_ast)

        num_digits = len(str(maximum)) - 1
        lower_bound = -int("".join(["9"] * num_digits))
        range_ast = _range_ast(lower_bound, maximum)
        return _alt(range_ast, _negative_with_min_digits_ast(num_digits))
    if maximum is None:
        assert minimum is not None
        if minimum < 0:
            lowerbound_ast = _range_ast(minimum, 0)
            return _alt(lowerbound_ast, _positive_unbounded_ast())

        num_digits = len(str(minimum))
        upperbound = int("".join(["9"] * num_digits))
        lowerbound_ast = _range_ast(minimum, upperbound)
        return _alt(lowerbound_ast, _positive_with_min_digits_ast(num_digits))

    assert minimum is not None and maximum is not None
    return _range_ast(minimum, maximum)


def _float_range_from_bounds_ast(
    minimum: Optional[DecimalLike],
    maximum: Optional[DecimalLike],
    strict: bool,
) -> Node:
    if minimum is None and maximum is None:
        decimal_ast = _strict_decimal_unbounded_ast()
        if strict:
            return decimal_ast
        return _alt(_integer_unbounded_ast(), decimal_ast)

    if minimum is None:
        assert maximum is not None
        maximum_decimal = _to_decimal(maximum, "maximum")
        if maximum_decimal >= 0:
            bounded = _float_range_ast(Decimal("0.0"), maximum_decimal)
            decimal_ast = _alt(_negative_strict_decimal_with_min_int_digits_ast(0), bounded)
        else:
            int_digits = len(str(int(abs(maximum_decimal).to_integral_value(rounding=ROUND_FLOOR))))
            lower = -(Decimal(10) ** int_digits) + Decimal("1")
            bounded = _float_range_ast(lower, maximum_decimal)
            decimal_ast = _alt(bounded, _negative_strict_decimal_with_min_int_digits_ast(int_digits))

        if strict:
            return decimal_ast

        integer_upper = int(maximum_decimal.to_integral_value(rounding=ROUND_FLOOR))
        integer_ast = _range_from_bounds_ast(None, integer_upper)
        return _alt(integer_ast, decimal_ast)

    if maximum is None:
        minimum_decimal = _to_decimal(minimum, "minimum")
        if minimum_decimal <= 0:
            bounded = _float_range_ast(minimum_decimal, Decimal("0.0"))
            decimal_ast = _alt(bounded, _positive_strict_decimal_with_min_int_digits_ast(0))
        else:
            int_digits = len(str(int(minimum_decimal.to_integral_value(rounding=ROUND_FLOOR))))
            upper = (Decimal(10) ** int_digits) - Decimal("1")
            bounded = _float_range_ast(minimum_decimal, upper)
            decimal_ast = _alt(bounded, _positive_strict_decimal_with_min_int_digits_ast(int_digits))

        if strict:
            return decimal_ast

        integer_lower = int(minimum_decimal.to_integral_value(rounding=ROUND_CEILING))
        integer_ast = _range_from_bounds_ast(integer_lower, None)
        return _alt(integer_ast, decimal_ast)

    minimum_decimal = _to_decimal(minimum, "minimum")
    maximum_decimal = _to_decimal(maximum, "maximum")
    decimal_ast = _float_range_ast(minimum_decimal, maximum_decimal)
    if strict:
        return decimal_ast

    lower_decimal, upper_decimal = (
        (minimum_decimal, maximum_decimal)
        if minimum_decimal < maximum_decimal
        else (maximum_decimal, minimum_decimal)
    )
    int_lower = int(lower_decimal.to_integral_value(rounding=ROUND_CEILING))
    int_upper = int(upper_decimal.to_integral_value(rounding=ROUND_FLOOR))
    if int_lower > int_upper:
        return decimal_ast

    integer_ast = _range_ast(int_lower, int_upper)
    return _alt(integer_ast, decimal_ast)


def range_regex(
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    capturing: bool = False,
):
    """
    Generate a regex for matching integers in an inclusive range.

    Supports integers only.

    - If both bounds are omitted, all integers are matched.
    - If ``minimum`` is omitted, values less than or equal to ``maximum`` are matched.
    - If ``maximum`` is omitted, values greater than or equal to ``minimum`` are matched.

    For floating-point ranges, use ``float_range_regex``.
    """
    return _range_from_bounds_ast(minimum, maximum).normalize().render(capturing=capturing)


def float_range_regex(
    minimum: Optional[DecimalLike] = None,
    maximum: Optional[DecimalLike] = None,
    strict: bool = True,
    capturing: bool = False,
) -> str:
    """
    Generate a regex for matching decimal numbers in an inclusive range.

    ``minimum`` and ``maximum`` can be ``int``, ``float``, ``Decimal``, or a
    string that can be parsed by ``Decimal``.

    - If ``strict`` is ``True`` (default), matches require a decimal point.
    - If ``strict`` is ``False``, both integer and decimal representations are
      matched, as long as their numeric value is in range.
    - If ``capturing`` is ``True``, grouping uses ``(...)`` instead of ``(?:...)``.
    """
    return _float_range_from_bounds_ast(minimum, maximum, strict).normalize().render(
        capturing=capturing
    )
