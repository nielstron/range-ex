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
    open = "(" if capture else "(?:"
    return f"{open}{s})"


def _to_decimal(value: DecimalLike) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


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
        if isinstance(self.node, (Either, Sequence, FixedRepetition)):
            inner = _parenthesise(inner, capture=capturing)
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
        if (
            (self.min_repeats == 0 and self.max_repeats == 0)
            or (isinstance(self.node, Sequence) and not self.node.parts)
            or (isinstance(self.node, Empty) and self.min_repeats == 0)
        ):
            return _seq()
        if isinstance(self.node, Empty) and self.min_repeats > 0:
            return Empty()
        normalized_child = self.node.normalize()
        # (a+)? === a* and (a*)? === a*
        if (
            isinstance(normalized_child, FixedRepetition)
            and self.min_repeats == 0
            and self.max_repeats == 1
            and normalized_child.max_repeats is None
            and normalized_child.min_repeats in (0, 1)
        ):
            return FixedRepetition(normalized_child.node, 0, None)
        if self.min_repeats == 1 and self.max_repeats == 1:
            return normalized_child
        # We can fold fixed repetitions of fixed m iff the child is of the form a{n,} or a{n}
        # In those cases a{n,}{m} === a{n*m,} and a{n}{m} === a{n*m}
        if (
            isinstance(normalized_child, FixedRepetition)
            and self.min_repeats
            == self.max_repeats  # implies self.max_repeats is not None
            and (
                normalized_child.max_repeats is None
                or normalized_child.min_repeats == normalized_child.max_repeats
            )
        ):
            multiplier = self.min_repeats
            return FixedRepetition(
                normalized_child.node,
                normalized_child.min_repeats * multiplier,
                normalized_child.max_repeats * multiplier
                if normalized_child.max_repeats is not None
                else None,
            )
        return FixedRepetition(normalized_child, self.min_repeats, self.max_repeats)


@dataclass(frozen=True)
class Sequence(Node):
    parts: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        return "".join(part.render(capturing=capturing) for part in self.parts)

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
            if isinstance(prev, Literal) and isinstance(part, Literal):
                merged_parts[-1] = Literal(prev.text + part.text)
                continue
            # Merge two adjacent fixed repetitions of the same node.
            # a{n,m}a{k,l} === a{n+k,m+l}
            if (prev.node if isinstance(prev, FixedRepetition) else prev) == (
                part.node if isinstance(part, FixedRepetition) else part
            ):
                node = prev.node if isinstance(prev, FixedRepetition) else prev
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
        if any(isinstance(part, Empty) for part in merged_parts):
            return Empty()
        return Sequence(tuple(merged_parts))

    def as_parts(self) -> list["Node"]:
        return list(self.parts)


@dataclass(frozen=True)
class Either(Node):
    options: tuple[Node, ...]

    def render(self, capturing: bool = False) -> str:
        return _parenthesise(
            "|".join(option.render(capturing=capturing) for option in self.options),
            capture=capturing,
        )

    def normalize(self) -> Node:
        # Normalize options first so each branch is internally simplified.
        normalized_options = [option.normalize() for option in self.options]
        seen = set()
        flattened_options: list[Node] = []
        for option in normalized_options:
            opts = option.as_options()
            for opt in opts:
                if isinstance(opt, Sequence) and not opt.parts:
                    # Remove empty sequence options since they don't contribute to the language.
                    continue
                if opt not in seen:
                    seen.add(opt)
                    flattened_options.append(opt)
        # Merge alternatives by unioning overlapping/adjacent repeat intervals for
        # the same base node. Plain nodes are treated as {1,1}.
        repetition_groups: dict[Node, list[tuple[int, Optional[int]]]] = {}
        repetition_group_order: list[Node] = []

        for option in flattened_options:
            base_node = option.node if isinstance(option, FixedRepetition) else option
            if base_node not in repetition_groups:
                repetition_groups[base_node] = []
                repetition_group_order.append(base_node)
            repetition_groups[base_node].append(
                (option.min_repeats, option.max_repeats)
            )

        merged_repetition_options: list[Node] = []
        for base_node in repetition_group_order:
            reps = repetition_groups[base_node]
            intervals = sorted(
                reps,
                key=lambda interval: interval[0],
            )
            merged: list[tuple[int, Optional[int]]] = []
            for interval_min, interval_max in intervals:
                if not merged:
                    merged.append((interval_min, interval_max))
                    continue

                prev_min, prev_max = merged[-1]
                can_merge = prev_max is None or interval_min <= prev_max + 1
                if can_merge:
                    if prev_max is None or interval_max is None:
                        merged[-1] = (prev_min, None)
                    else:
                        merged[-1] = (prev_min, max(prev_max, interval_max))
                else:
                    merged.append((interval_min, interval_max))

            for interval_min, interval_max in merged:
                if interval_min == 1 and interval_max == 1:
                    merged_repetition_options.append(base_node)
                else:
                    merged_repetition_options.append(
                        FixedRepetition(base_node, interval_min, interval_max)
                    )

        result_options = merged_repetition_options
        if len(result_options) == 1:
            return result_options[0]
        return Either(tuple(result_options))

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

    return _one_of(*patterns)


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


def _fixed_width_int_range_ast(lower: int, upper: int, width: int) -> Node:
    if upper < lower:
        return Empty()
    return __compute_numerical_range_ast(
        str(lower).zfill(width), str(upper).zfill(width)
    )


def _fractional_interval_ast(
    lower: Decimal,
    upper: Decimal,
    *,
    upper_open_one: bool = False,
) -> Node:
    """
    Match characters after "." for:
    - [lower, upper] if upper_open_one is False
    - [lower, 1) if upper_open_one is True
    """
    if upper_open_one:
        if not (Decimal("0") <= lower < Decimal("1")):
            return Empty()
    else:
        if upper < lower or not (Decimal("0") <= lower <= upper <= Decimal("1")):
            return Empty()

    p = max(_fractional_precision(lower), _fractional_precision(upper))
    if p == 0:
        p = 1

    scale = 10**p
    scaled_lower = int(lower * scale)
    scaled_upper = (scale - 1) if upper_open_one else int(upper * scale)
    patterns: list[Node] = []

    # Shorter forms (q < p), e.g. ".2" equivalent to ".2000".
    for q in range(1, p):
        factor = 10 ** (p - q)
        q_lower = (scaled_lower + factor - 1) // factor
        q_upper = scaled_upper // factor
        if q_lower <= q_upper:
            patterns.append(_fixed_width_int_range_ast(q_lower, q_upper, q))

    # Exact p-digit forms.
    patterns.append(_fixed_width_int_range_ast(scaled_lower, scaled_upper, p))

    # Longer forms (q > p).
    if upper_open_one:
        if scaled_lower <= scale - 1:
            full_prefixes = _fixed_width_int_range_ast(scaled_lower, scale - 1, p)
            patterns.append(_seq(full_prefixes, _any_digits(None)))
    else:
        if scaled_lower <= scaled_upper - 1:
            full_prefixes = _fixed_width_int_range_ast(
                scaled_lower, scaled_upper - 1, p
            )
            patterns.append(_seq(full_prefixes, _any_digits(None)))
        upper_text = str(scaled_upper).zfill(p)
        patterns.append(
            _seq(
                *_literal_parts(upper_text),
                FixedRepetition(Literal("0"), 1, None),
            )
        )

    return _one_of(*patterns)


def _float_range_ast_within_one(a: Decimal, b: Decimal) -> Node:
    """
    Generate a regex AST that matches any number [a, b], where floor(a) <= a < b <= ceil(a)
    """
    pre_decs = floor(a) if a > 0 else ceil(a)
    pre_decs_node = Literal(str(abs(pre_decs)) if pre_decs < 0 else str(pre_decs))

    if a < 0 and pre_decs == 0:
        negative_fractional_lower = abs(b)
        negative_fractional_upper = abs(a)
        negative_decimal_ast = _fractional_interval_ast(
            negative_fractional_lower,
            negative_fractional_upper,
            upper_open_one=False,
        )
        negative_part = _seq(
            Literal("-"),
            optional(Literal("0")),
            Literal("."),
            negative_decimal_ast,
        )
        if b == 0:
            zero_decimal_ast = _fractional_interval_ast(Decimal("0"), Decimal("0"))
            zero_part = _one_of(
                _seq(optional(Literal("0")), Literal("."), zero_decimal_ast),
                _seq(Literal("0"), Literal("."), optional(zero_decimal_ast)),
            )
            return _one_of(negative_part, zero_part)
        return negative_part

    if pre_decs < 0:
        fractional_lower = abs(b - pre_decs)
        fractional_upper = abs(a - pre_decs)
    else:
        fractional_lower = a - pre_decs
        fractional_upper = b - pre_decs

    decimal_ast = _fractional_interval_ast(
        fractional_lower,
        fractional_upper,
        upper_open_one=(pre_decs >= 0 and b == pre_decs + 1),
    )
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
    a_decimal = _to_decimal(a)
    b_decimal = _to_decimal(b)
    if b_decimal < a_decimal:
        return Empty()

    # Handle negative ranges by mirroring to positive magnitudes.
    if b_decimal < 0:
        return _seq(
            Literal("-"),
            _float_range_ast(abs(b_decimal), abs(a_decimal), strict=strict),
        )
    if a_decimal < 0 <= b_decimal:
        patterns: list[Node] = []
        patterns.append(
            _seq(
                Literal("-"),
                _float_range_ast(Decimal("0"), abs(a_decimal), strict=strict),
            )
        )
        patterns.append(_float_range_ast(Decimal("0"), b_decimal, strict=strict))
        return _one_of(*patterns)

    start_bucket = floor(a_decimal)
    end_bucket = floor(b_decimal)
    if start_bucket == end_bucket:
        return _float_range_ast_within_one(a_decimal, b_decimal)

    patterns: list[Node] = []
    patterns.append(_float_range_ast_within_one(a_decimal, Decimal(start_bucket + 1)))

    middle_min = start_bucket + 1
    middle_max = end_bucket - 1
    if middle_min <= middle_max:
        decimals = _seq(Literal("."), _any_digits(None))
        if not strict:
            decimals = optional(decimals)
        patterns.append(_seq(_range_from_bounds_ast(middle_min, middle_max), decimals))

    patterns.append(_float_range_ast_within_one(Decimal(end_bucket), b_decimal))
    return _one_of(*patterns)


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
    return _seq(
        Literal("-"), _positive_strict_decimal_with_min_int_digits_ast(extra_digits)
    )


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
        maximum_decimal = _to_decimal(maximum)
        if maximum_decimal >= 0:
            bounded = _float_range_ast(Decimal("0.0"), maximum_decimal, strict=strict)
            decimal_ast = _one_of(
                _negative_strict_decimal_with_min_int_digits_ast(0), bounded
            )
        else:
            int_digits = len(str(int(floor(abs(maximum_decimal)))))
            lower = -(Decimal(10) ** int_digits)
            bounded = _float_range_ast(lower, maximum_decimal, strict)
            decimal_ast = _one_of(
                bounded, _negative_strict_decimal_with_min_int_digits_ast(int_digits)
            )

        integer_upper = int(floor(maximum_decimal))
        integer_ast = _range_from_bounds_ast(None, integer_upper)
        integer_dot_ast = _seq(integer_ast, Literal("."))
        if strict:
            return _one_of(decimal_ast, integer_dot_ast)
        return _one_of(integer_ast, integer_dot_ast, decimal_ast)

    if maximum is None:
        minimum_decimal = _to_decimal(minimum)
        if minimum_decimal <= 0:
            bounded = _float_range_ast(minimum_decimal, Decimal("0.0"), strict=strict)
            decimal_ast = _one_of(
                bounded, _positive_strict_decimal_with_min_int_digits_ast(0)
            )
        else:
            int_digits = len(
                str(int(minimum_decimal.to_integral_value(rounding=ROUND_FLOOR)))
            )
            upper = Decimal(10) ** int_digits
            bounded = _float_range_ast(minimum_decimal, upper, strict=strict)
            decimal_ast = _one_of(
                bounded, _positive_strict_decimal_with_min_int_digits_ast(int_digits)
            )

        integer_lower = int(minimum_decimal.to_integral_value(rounding=ROUND_CEILING))
        integer_ast = _range_from_bounds_ast(integer_lower, None)
        integer_dot_ast = _seq(integer_ast, Literal("."))
        if strict:
            return _one_of(decimal_ast, integer_dot_ast)
        return _one_of(integer_ast, integer_dot_ast, decimal_ast)

    minimum_decimal = _to_decimal(minimum)
    maximum_decimal = _to_decimal(maximum)
    decimal_ast = _float_range_ast(minimum_decimal, maximum_decimal, strict=strict)
    integer_ast = _range_ast(ceil(minimum_decimal), floor(maximum_decimal))
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
    return (
        _range_from_bounds_ast(minimum, maximum).normalize().render(capturing=capturing)
    )


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
    return (
        _float_range_from_bounds_ast(minimum, maximum, strict)
        .normalize()
        .render(capturing=capturing)
    )
