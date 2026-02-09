from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Optional, Union

DecimalLike = Union[int, float, Decimal, str]


class Node(ABC):
    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    def normalize(self) -> "NodeType":
        return self


@dataclass(frozen=True)
class Literal(Node):
    text: str

    def render(self) -> str:
        return re.escape(self.text)


@dataclass(frozen=True)
class DigitRange(Node):
    start: int
    end: int

    def render(self) -> str:
        if self.start == self.end:
            return str(self.start)
        if self.start == 0 and self.end == 9:
            return r"\d"
        return f"[{self.start}-{self.end}]"

    def normalize(self) -> "NodeType":
        if self.start == self.end:
            return Literal(str(self.start))
        return self


@dataclass(frozen=True)
class OptionalNode(Node):
    node: "NodeType"

    def render(self) -> str:
        return f"(?:{self.node.render()})?"

    def normalize(self) -> "NodeType":
        return OptionalNode(self.node.normalize())


@dataclass(frozen=True)
class ZeroOrMore(Node):
    node: "NodeType"

    def render(self) -> str:
        return f"(?:{self.node.render()})*"

    def normalize(self) -> "NodeType":
        return ZeroOrMore(self.node.normalize())


@dataclass(frozen=True)
class Sequence(Node):
    parts: tuple["NodeType", ...]

    def render(self) -> str:
        return "".join(part.render() for part in self.parts)

    def normalize(self) -> "NodeType":
        normalized_parts = [part.normalize() for part in self.parts]
        merged_parts: list[NodeType] = []
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
    options: tuple[Sequence, ...]

    def render(self) -> str:
        return f"(?:{'|'.join(option.render() for option in self.options)})"

    def normalize(self) -> "NodeType":
        normalized_options = [option.normalize() for option in self.options]
        seen: set[str] = set()
        unique_options: list[Sequence] = []
        for option in normalized_options:
            assert isinstance(option, Sequence)
            key = option.render()
            if key not in seen:
                seen.add(key)
                unique_options.append(option)
        return Alternation(tuple(unique_options))


NodeType = Union[Literal, DigitRange, OptionalNode, ZeroOrMore, Sequence, Alternation]


def _literal_parts(text: str) -> list[NodeType]:
    return [Literal(c) for c in text]


def _any_digits(count: int) -> list[NodeType]:
    return [DigitRange(0, 9) for _ in range(count)]


def __compute_numerical_range_ast(
    str_a: str, str_b: str, start_parts: Optional[list[NodeType]] = None
) -> list[Sequence]:
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
        return [Sequence(tuple(parts + _literal_parts(str_a)))]
    if str_len == 1:
        return [Sequence(tuple(parts + [DigitRange(int(str_a), int(str_b))]))]

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

    patterns: list[Sequence] = []
    intermediate_range = list(range(int(str_a[0]) + 1, int(str_b[0])))
    if intermediate_range:
        patterns.append(
            Sequence(
                tuple(
                    parts
                    + [DigitRange(intermediate_range[0], intermediate_range[-1])]
                    + _any_digits(str_len - 1)
                )
            )
        )

    for loop_counter in range(str_len - 1):
        prefix = parts + _literal_parts(str_a[: loop_counter + 1])
        if loop_counter == str_len - 2:
            patterns.append(
                Sequence(tuple(prefix + [DigitRange(int(str_a[-1]), 9)]))
            )
        elif str_a[loop_counter + 1] != "9":
            patterns.append(
                Sequence(
                    tuple(
                        prefix
                        + [DigitRange(int(str_a[loop_counter + 1]) + 1, 9)]
                        + _any_digits(str_len - 2 - loop_counter)
                    )
                )
            )

    for loop_counter in range(str_len - 1):
        prefix = parts + _literal_parts(str_b[: loop_counter + 1])
        if loop_counter == str_len - 2:
            patterns.append(
                Sequence(tuple(prefix + [DigitRange(0, int(str_b[-1]))]))
            )
        elif str_b[loop_counter + 1] != "0":
            patterns.append(
                Sequence(
                    tuple(
                        prefix
                        + [DigitRange(0, int(str_b[loop_counter + 1]) - 1)]
                        + _any_digits(str_len - 2 - loop_counter)
                    )
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


def _range_ast(a: int, b: int) -> Alternation:
    a, b = (a, b) if a < b else (b, a)
    ranges = __range_splitter(a, b)
    patterns: list[Sequence] = []
    for start, end, sign in ranges:
        prefix = [Literal("-")] if sign else []
        patterns.extend(
            __compute_numerical_range_ast(str(start), str(end), start_parts=prefix)
        )
    return Alternation(tuple(patterns))


def _float_range_ast(a: DecimalLike, b: DecimalLike) -> Alternation:
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
    new_patterns: list[Sequence] = []
    for pattern in scaled_int_ast.options:
        parts = list(pattern.parts)
        sign_parts: list[NodeType] = []
        if parts and isinstance(parts[0], Literal) and parts[0].text.startswith("-"):
            sign_parts = [Literal("-")]
            leading_rest = parts[0].text[1:]
            parts = _literal_parts(leading_rest) + parts[1:]

        if len(parts) < max_num_decimal:
            parts = [Literal("0")] * (max_num_decimal - len(parts)) + parts

        split_index = len(parts) - max_num_decimal
        non_fractional = parts[:split_index]
        fractional = parts[split_index:]

        required_fractional: list[NodeType] = [fractional[0]]
        required_fractional.extend(OptionalNode(token) for token in fractional[1:])

        if non_fractional:
            non_fractional_parts = non_fractional
        else:
            non_fractional_parts = [OptionalNode(Literal("0"))]

        new_patterns.append(
            Sequence(
                tuple(
                    sign_parts
                    + non_fractional_parts
                    + [Literal("."), *required_fractional, ZeroOrMore(DigitRange(0, 9))]
                )
            )
        )

    return Alternation(tuple(new_patterns))


def _to_decimal(value: DecimalLike) -> Decimal:
    return Decimal(str(value))


def _range_regex(a: int, b: int) -> str:
    """Generate a regex that matches integers in the inclusive range [a, b]."""
    return _range_ast(a, b).normalize().render()


def _integer_unbounded_ast() -> Alternation:
    return Alternation(
        (
            Sequence((OptionalNode(Literal("-")), DigitRange(1, 9), ZeroOrMore(DigitRange(0, 9)))),
            Sequence((Literal("0"),)),
        )
    )


def _negative_unbounded_ast() -> Sequence:
    return Sequence((Literal("-"), DigitRange(1, 9), ZeroOrMore(DigitRange(0, 9))))


def _positive_unbounded_ast() -> Sequence:
    return Sequence((DigitRange(1, 9), ZeroOrMore(DigitRange(0, 9))))


def _negative_with_min_digits_ast(extra_digits: int) -> Sequence:
    return Sequence(
        (
            Literal("-"),
            DigitRange(1, 9),
            *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
            ZeroOrMore(DigitRange(0, 9)),
        )
    )


def _positive_with_min_digits_ast(extra_digits: int) -> Sequence:
    return Sequence(
        (
            DigitRange(1, 9),
            *tuple(DigitRange(0, 9) for _ in range(extra_digits)),
            ZeroOrMore(DigitRange(0, 9)),
        )
    )


def range_regex(minimum: Optional[int] = None, maximum: Optional[int] = None):
    """
    Generate a regex for matching integers in an inclusive range.

    Supports integers only.

    - If both bounds are omitted, all integers are matched.
    - If ``minimum`` is omitted, values less than or equal to ``maximum`` are matched.
    - If ``maximum`` is omitted, values greater than or equal to ``minimum`` are matched.

    For floating-point ranges, use ``float_range_regex``.
    """
    if minimum is None and maximum is None:
        return _integer_unbounded_ast().normalize().render()
    if minimum is None:
        assert maximum is not None
        if maximum == 0:
            return Alternation((_negative_unbounded_ast(), Sequence((Literal("0"),)))).normalize().render()
        if maximum > 0:
            upperbound_ast = _range_ast(0, maximum)
            return Alternation((_negative_unbounded_ast(), *upperbound_ast.options)).normalize().render()

        num_digits = len(str(maximum)) - 1
        lower_bound = -int("".join(["9"] * num_digits))
        range_ast = _range_ast(lower_bound, maximum)
        return Alternation(
            (*range_ast.options, _negative_with_min_digits_ast(num_digits))
        ).normalize().render()
    if maximum is None:
        assert minimum is not None
        if minimum < 0:
            lowerbound_ast = _range_ast(minimum, 0)
            return Alternation((*lowerbound_ast.options, _positive_unbounded_ast())).normalize().render()

        num_digits = len(str(minimum))
        upperbound = int("".join(["9"] * num_digits))
        lowerbound_ast = _range_ast(minimum, upperbound)
        return Alternation(
            (*lowerbound_ast.options, _positive_with_min_digits_ast(num_digits))
        ).normalize().render()

    assert minimum is not None and maximum is not None
    return _range_regex(minimum, maximum)


def float_range_regex(
    minimum: DecimalLike, maximum: DecimalLike, strict: bool = True
) -> str:
    """
    Generate a regex for matching decimal numbers in an inclusive range.

    ``minimum`` and ``maximum`` can be ``int``, ``float``, ``Decimal``, or a
    string that can be parsed by ``Decimal``.

    - If ``strict`` is ``True`` (default), matches require a decimal point.
    - If ``strict`` is ``False``, both integer and decimal representations are
      matched, as long as their numeric value is in range.
    """
    minimum_decimal = _to_decimal(minimum)
    maximum_decimal = _to_decimal(maximum)

    decimal_ast = _float_range_ast(minimum_decimal, maximum_decimal)
    if strict:
        return decimal_ast.normalize().render()

    lower_decimal, upper_decimal = (
        (minimum_decimal, maximum_decimal)
        if minimum_decimal < maximum_decimal
        else (maximum_decimal, minimum_decimal)
    )
    int_lower = int(lower_decimal.to_integral_value(rounding=ROUND_CEILING))
    int_upper = int(upper_decimal.to_integral_value(rounding=ROUND_FLOOR))
    if int_lower > int_upper:
        return decimal_ast.normalize().render()

    integer_ast = _range_ast(int_lower, int_upper)
    combined = Alternation(tuple(integer_ast.options + decimal_ast.options)).normalize()
    return combined.render()
