from __future__ import annotations

import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import floor, ceil
from typing import Optional, Union

DecimalLike = Union[int, float, Decimal, str]

def _parenthesise(s: str, capture: bool = False) -> str:
    open = "(?:" if capture else "("
    return f"{open}{s})"

class Node(ABC):
    @abstractmethod
    def render(self, capturing: bool = False) -> str:
        raise NotImplementedError

    def normalize(self) -> "Node":
        return self

    def as_parts(self) -> list["Node"]:
        return [self]

    def as_options(self) -> list["Node"]:
        return [self]

    @property
    def min_repeats(self) -> int:
        return 1

    @property
    def max_repeats(self) -> Optional[int]:
        return 1



@dataclass(frozen=True)
class Empty(Node):
    def render(self, capturing: bool = False) -> str:
        _ = capturing
        return r"(?!)"


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
        # Collapse singleton digit ranges like [7-7] into a literal node.
        if self.start == self.end:
            return Literal(str(self.start))
        return self


@dataclass(frozen=True)
class FixedRepetition(Node):
    node: Node
    _min_repeats: int
    _max_repeats: Optional[int] = None

    @property
    def min_repeats(self) -> int:
        return self._min_repeats

    @property
    def max_repeats(self) -> Optional[None]:
        return self._max_repeats

    def __post_init__(self) -> None:
        if self._min_repeats < 0:
            raise ValueError("min_repeats must be >= 0")
        if self._max_repeats is not None and self._max_repeats < self._min_repeats:
            raise ValueError("max_repeats must be >= min_repeats")

    def render(self, capturing: bool = False) -> str:
        inner = self.node.render(capturing=capturing)
        if self.min_repeats == 0 and self.max_repeats == 0:
            return ""
        elif self.min_repeats == 0 and self.max_repeats == 1:
            quantifier = "?"
        elif self.min_repeats == 0 and self.max_repeats is None:
            quantifier = "*"
        elif self.min_repeats == 1 and self.max_repeats is None:
            quantifier = "+"
        elif self.max_repeats is not None and self.min_repeats == self.max_repeats:
            quantifier = f"{{{self.min_repeats}}}"
        elif self.max_repeats is None:
            quantifier = f"{{{self.min_repeats},}}"
        else:
            quantifier = f"{{{self.min_repeats},{self.max_repeats}}}"
        return f"{inner}{quantifier}"

    def normalize(self) -> "Node":
        # Normalize repeated child
        if self.min_repeats == 0 and self.max_repeats == 0:
            return _seq()
        normalized_child = self.node.normalize()
        if self.min_repeats == 1 and self.max_repeats == 1:
            return normalized_child
        # We can fold fixed repetitions of fixed m iff the child is of the form a{n,} or a{n}
        # In those cases a{n,}{m} === a{n*m,} and a{n}{m} === a{n*m}
        if (
            isinstance(normalized_child, FixedRepetition)
            and self.min_repeats == self.max_repeats  # implies self.max_repeats is not None
            and (normalized_child.max_repeats is None or normalized_child.min_repeats == normalized_child.max_repeats)
        ):
            multiplier = self.min_repeats
            return FixedRepetition(
                normalized_child.node,
                normalized_child.min_repeats * multiplier,
                normalized_child.max_repeats * multiplier if normalized_child.max_repeats is not None else None,
            )
        return FixedRepetition(normalized_child, self.min_repeats, self.max_repeats)


@dataclass(frozen=True)
class Sequence(Node):
    parts: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        return _parenthesise("".join(part.render(capturing=capturing) for part in self.parts), capture=capturing)

    def normalize(self) -> "Node":
        # Normalize each child first so deep rewrites happen bottom-up.
        normalized_parts = [part.normalize() for part in self.parts]
        flattened_parts: list[Node] = []
        for part in normalized_parts:
            # Flatten nested sequences: (ab)(cd) -> abcd in the AST.
            flattened_parts.extend(part.as_parts())
        merged_parts: list[Node] = flattened_parts[:1]
        for part in flattened_parts[1:]:
            # Merge adjacent literals into one node to reduce AST noise.
            prev = merged_parts[-1]
            if (
                isinstance(prev, Literal)
                and isinstance(part, Literal)
            ):
                merged_parts[-1] = Literal(prev.text + part.text)
                continue
            # Merge two adjacent fixed repetitions of the same node.
            # a{n,m}a{k,l} === a{n+k,m+l}
            if (
                (prev.node if isinstance(prev, FixedRepetition) else prev) == (part.node if isinstance(part, FixedRepetition) else part)
            ):
                node = (prev.node if isinstance(prev, FixedRepetition) else prev)
                max_repeats = (
                    None
                    if prev.max_repeats is None or part.max_repeats is None
                    else prev.max_repeats + part.max_repeats
                )
                merged_parts[-1] = FixedRepetition(
                    node, prev.min_repeats + part.min_repeats, max_repeats
                )
                continue
            merged_parts.append(part)
        if len(merged_parts) == 1:
            return merged_parts[0]
        return Sequence(tuple(merged_parts))

    def as_parts(self) -> list["Node"]:
        return list(self.parts)


@dataclass(frozen=True)
class Either(Node):
    options: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        return _parenthesise('|'.join(option.render(capturing=capturing) for option in self.options), capture=capturing)

    def normalize(self) -> Node:
        # Normalize options first so each branch is internally simplified.
        normalized_options = [option.normalize() for option in self.options]
        seen = set()
        flattened_options: list[Node] = []
        for option in normalized_options:
            opts = option.as_options()
            for opt in opts:
                if opt not in seen:
                    seen.add(opt)
                    flattened_options.append(opt)
        return Either(tuple(flattened_options))

    def as_options(self):
        return list(self.options)


def _literal_parts(text: str) -> list[Node]:
    return [Literal(c) for c in text]


def _any_digits(count: Optional[int]) -> Node:
    """
    If count is not None, match exactly count digits. Otherwise, match any number of digits.
    """
    return FixedRepetition(DigitRange(0, 9), count or 0, count)


def _one_of(*nodes: Node) -> Node:
    if len(nodes) == 1:
        return nodes[0]
    return Either(tuple(nodes))


def _seq(*nodes: Node) -> Node:
    flat: list[Node] = []
    for node in nodes:
        flat.extend(node.as_parts())
    return Sequence(tuple(flat))

def optional(node: Node) -> Node:
    return FixedRepetition(node, 0, 1)

def __compute_numerical_range_ast(
    str_a: str, str_b: str, start_parts: Optional[list[Node]] = None
) -> Node:
    """
    Build regex AST sequences for an inclusive integer range with equal-width endpoints.

    Assumes:
    - int(str_a) <= int(str_b)
    - len(str_a) == len(str_b)
    """
    if len(str_a) != len(str_b):
        raise Exception(
            f"The input numbers ({str_a}, {str_b}) must have equal number of digits"
        )

    parts = list(start_parts or [])
    str_len = len(str_a)

    if str_a == str_b:
        return _seq(*parts, *_literal_parts(str_a))
    if str_len == 1:
        return _seq(*parts, DigitRange(int(str_a), int(str_b)))

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
                _any_digits(str_len - 1),
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
                    _any_digits(str_len - 2 - loop_counter),
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
                    _any_digits(str_len - 2 - loop_counter),
                )
            )

    return _seq(*patterns)


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
        a, b = abs(b), abs(a)
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
    if b < a:
        return Empty()
    ranges = __range_splitter(a, b)
    patterns: list[Node] = []
    for start, end, sign in ranges:
        prefix = [Literal("-")] if sign else []
        patterns.append(
            __compute_numerical_range_ast(str(start), str(end), start_parts=prefix)
        )
    return _one_of(*patterns)


def _fractional_precision(value: Decimal) -> int:
    text = format(value, "f")
    if "." not in text:
        return 0
    return len(text) - (text.find(".") + 1)

def _float_range_ast_within_one(a: Decimal, b: Decimal) -> Node:
    """
    Generate a regex AST that matches any number [a, b], where floor(a) <= a < b <= ceil(a)
    """
    # everything before decimals is fixed
    pre_decs = floor(a) if a > 0 else ceil(a)
    pre_decs_node = Literal(str(pre_decs))
    if a >= 0:
        str_a = str(a - pre_decs)[2:]  # skip "0."
        if b != ceil(a):
            # after decimals, match the range from equalized width strings + optional trailing digits
            str_b_orig = str(b - pre_decs)[2:]  # skip "0."
            str_b = str_b_orig
            # equalize widths by padding with zeros, then compute the range AST for the fractional part
            str_a = str_a.rstrip("0") or "0"
            str_b = str_b.rstrip("0") or "0"
            max_decimals = max(len(str_a), len(str_b))
            str_a = str_a.ljust(max_decimals, "0")
            str_b = str_b.ljust(max_decimals, "0")
            str_b = str(int(str_b) - 1)
        else:
            # if b is exactly ceil(a), we match a to infinity, so the decimal part can be any number with the same precision as a
            str_b_orig = "9" * _fractional_precision(a)
            str_b = str_b_orig
        decimal_ast = __compute_numerical_range_ast(str_a, str_b)
        decimal_ast = _one_of(_seq(decimal_ast, _any_digits(None)), Literal(str_b_orig))
    else:
        str_b = str(abs(b - pre_decs))[2:]  # skip "0."
        if floor(b) != a:
            # after decimals, match the range from equalized width strings + optional trailing digits
            str_a_orig = str(abs(a - pre_decs))[2:]  # skip "0."
            str_a = str(int(str_a_orig) - 1)
            # equalize widths by padding with zeros, then compute the range AST for the fractional part
            str_a = str_a.rstrip("0") or "0"
            str_b = str_b.rstrip("0") or "0"
            max_decimals = max(len(str_a), len(str_b))
            str_a = str_a.ljust(max_decimals, "0")
            str_b = str_b.ljust(max_decimals, "0")
            decimal_ast = __compute_numerical_range_ast(str_b, str_a)
            decimal_ast = _one_of(_seq(decimal_ast, _any_digits(None)), Literal(str_a_orig))
        else:
            decimal_ast = _range_from_bounds_ast(int(str_b or "0"), None)
    if a - pre_decs == 0 and pre_decs == 0:
        # handle the special case that just "." is not allowed
        return _one_of(
            _seq(optional(pre_decs_node), Literal("."), decimal_ast),
            _seq(pre_decs_node, Literal("."), optional(decimal_ast)),
        )
    # in the other cases we can optionally allow the missing part (e.g. ".2" can be written as "0.2", and "1." can be written as "1")
    elif pre_decs == 0:
        pre_decs_node = optional(pre_decs_node)
    elif (a - pre_decs == 0 and a >= 0) or (b - pre_decs == 0 and a < 0):
        decimal_ast = optional(decimal_ast)
    sign = [Literal("-")] if pre_decs < 0 else []
    return _seq(*sign, pre_decs_node, Literal("."), decimal_ast)

def _float_range_ast(a: DecimalLike, b: DecimalLike, strict=False) -> Node:
    """
    Generate a regex AST that matches decimal numbers in the inclusive range [a, b].
    """
    a_decimal = Decimal(a)
    b_decimal = Decimal(b)
    if b_decimal < a_decimal:
        return Empty()

    # there are three ranges here:
    # [a, ceil(a)], [ceil(a), floor(b)], [floor(b), b]
    first_pattern = _float_range_ast_within_one(a_decimal, ceil(a_decimal))
    decimals = _seq(Literal("."), _any_digits(None))
    if not strict:
        decimals = optional(decimals)
    # through appending decimals, in the positive, this matches a - (b-1).9999...
    # in the negative likewise, it matches (a+1).0000... - b
    second_pattern = _seq(_range_from_bounds_ast(ceil(a_decimal)+(1 if a_decimal < 0 else 0), floor(b_decimal)-(1 if b_decimal > 0 else 0)), decimals)
    # we therefore need to explicitly add a and b in non-strict mode since it may not covered by the above pattern
    if not strict:
        second_pattern = _one_of(second_pattern, Literal(str(floor(b_decimal))), Literal(str(ceil(a_decimal))))
    third_pattern = _float_range_ast_within_one(floor(b_decimal), b_decimal)
    return _one_of(first_pattern, second_pattern, third_pattern)


def _zero() -> Literal:
    return Literal("0")


def _integer_unbounded_ast() -> Node:
    return _one_of(_negative_unbounded_ast(), _positive_unbounded_ast(), _zero())

def _positive_with_min_digits_ast(extra_digits: int) -> Node:
    if extra_digits < 0:
        raise ValueError("extra_digits must be >= 0")
    return _seq(
        DigitRange(1, 9),
        *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
        FixedRepetition(DigitRange(0, 9), 0, None),
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
    integer_part = _one_of(Literal("0"), _positive_unbounded_ast())
    return _one_of(
        _seq(
            FixedRepetition(Literal("-"), 0, 1),
            integer_part,
            Literal("."),
            DigitRange(0, 9),
            FixedRepetition(DigitRange(0, 9), 0, None),
        ),
        _seq(FixedRepetition(Literal("-"), 0, 1), integer_part, Literal(".")),
        _seq(
            FixedRepetition(Literal("-"), 0, 1),
            Literal("."),
            DigitRange(0, 9),
            FixedRepetition(DigitRange(0, 9), 0, None),
        ),
    )

def _negative_strict_decimal_with_min_int_digits_ast(extra_digits: int) -> Node:
    return _seq(Literal("-"), _positive_strict_decimal_with_min_int_digits_ast(extra_digits))


def _positive_strict_decimal_with_min_int_digits_ast(extra_digits: int) -> Node:
    if extra_digits < 0:
        raise ValueError("extra_digits must be >= 0")
    if extra_digits == 0:
        integer_part = _one_of(Literal("0"), _positive_unbounded_ast())
        return _one_of(
            _seq(
                integer_part,
                Literal("."),
                DigitRange(0, 9),
                FixedRepetition(DigitRange(0, 9), 0, None),
            ),
            _seq(integer_part, Literal(".")),
            _seq(
                Literal("."),
                DigitRange(0, 9),
                FixedRepetition(DigitRange(0, 9), 0, None),
            ),
        )

    integer_part = _seq(
        DigitRange(1, 9),
        *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
        FixedRepetition(DigitRange(0, 9), 0, None),
    )
    return _one_of(
        _seq(
            integer_part,
            Literal("."),
            DigitRange(0, 9),
            FixedRepetition(DigitRange(0, 9), 0, None),
        ),
        _seq(integer_part, Literal(".")),
    )


def _range_from_bounds_ast(minimum: Optional[int], maximum: Optional[int]) -> Node:
    if minimum is not None:
        minimum = operator.index(minimum)
    if maximum is not None:
        maximum = operator.index(maximum)
    if minimum is not None and maximum is not None and minimum > maximum:
        return Empty()

    if minimum is None and maximum is None:
        return _integer_unbounded_ast()
    if minimum is None:
        assert maximum is not None
        if maximum == 0:
            return _one_of(_negative_unbounded_ast(), Literal("0"))
        if maximum > 0:
            upperbound_ast = _range_ast(0, maximum)
            return _one_of(_negative_unbounded_ast(), upperbound_ast)

        num_digits = len(str(maximum)) - 1
        lower_bound = -int("".join(["9"] * num_digits))
        range_ast = _range_ast(lower_bound, maximum)
        return _one_of(range_ast, _negative_with_min_digits_ast(num_digits))
    if maximum is None:
        assert minimum is not None
        if minimum < 0:
            lowerbound_ast = _range_ast(minimum, 0)
            return _one_of(lowerbound_ast, _positive_unbounded_ast())

        num_digits = len(str(minimum))
        upperbound = int("".join(["9"] * num_digits))
        lowerbound_ast = _range_ast(minimum, upperbound)
        return _one_of(lowerbound_ast, _positive_with_min_digits_ast(num_digits))

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
        return _one_of(_integer_unbounded_ast(), decimal_ast)

    if minimum is None:
        assert maximum is not None
        maximum_decimal = Decimal(maximum)
        if maximum_decimal >= 0:
            bounded = _float_range_ast(Decimal("0.0"), maximum_decimal, strict=strict)
            decimal_ast = _one_of(_negative_strict_decimal_with_min_int_digits_ast(0), bounded)
        else:
            int_digits = len(
                str(int(floor(abs(maximum_decimal))))
            )
            lower = -(Decimal(10) ** int_digits)
            bounded = _float_range_ast(lower, maximum_decimal, strict)
            decimal_ast = _one_of(bounded, _negative_strict_decimal_with_min_int_digits_ast(int_digits))

        integer_upper = int(floor(maximum_decimal))
        integer_ast = _range_from_bounds_ast(None, integer_upper)
        integer_dot_ast = _seq(integer_ast, Literal("."))
        if strict:
            return _one_of(decimal_ast, integer_dot_ast)
        return _one_of(integer_ast, integer_dot_ast, decimal_ast)

    if maximum is None:
        minimum_decimal = Decimal(minimum)
        if minimum_decimal <= 0:
            bounded = _float_range_ast(minimum_decimal, Decimal("0.0"))
            decimal_ast = _one_of(bounded, _positive_strict_decimal_with_min_int_digits_ast(0))
        else:
            int_digits = len(str(int(minimum_decimal.to_integral_value(rounding=ROUND_FLOOR))))
            upper = Decimal(10) ** int_digits
            bounded = _float_range_ast(minimum_decimal, upper)
            decimal_ast = _one_of(bounded, _positive_strict_decimal_with_min_int_digits_ast(int_digits))

        integer_lower = int(minimum_decimal.to_integral_value(rounding=ROUND_CEILING))
        integer_ast = _range_from_bounds_ast(integer_lower, None)
        integer_dot_ast = _seq(integer_ast, Literal("."))
        if strict:
            return _one_of(decimal_ast, integer_dot_ast)
        return _one_of(integer_ast, integer_dot_ast, decimal_ast)

    decimal_ast = _float_range_ast(minimum, maximum)
    integer_ast = _range_ast(ceil(minimum), floor(maximum))
    integer_dot_ast = _seq(integer_ast, Literal("."))
    if strict:
        return _one_of(decimal_ast, integer_dot_ast)
    return _one_of(integer_ast, integer_dot_ast, decimal_ast)


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
