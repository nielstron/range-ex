from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Optional, Union

DecimalLike = Union[int, float, Decimal, str]

ANY_DIGIT = r"\d"


def __digit_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    if start == 0 and end == 9:
        return ANY_DIGIT
    return f"[{start}-{end}]"


def __tokenize_numeric_pattern(pattern: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "[":
            end = pattern.find("]", i)
            if end == -1:
                raise ValueError(f"Malformed range expression: {pattern}")
            tokens.append(pattern[i : end + 1])
            i = end + 1
        elif i + 1 < len(pattern) and pattern[i] == "\\" and pattern[i + 1] == "d":
            tokens.append(r"\d")
            i += 2
        else:
            tokens.append(pattern[i])
            i += 1
    return tokens


def __compute_numerical_range(str_a, str_b, start_appender_str=""):
    """
    Build a regex fragment for an inclusive integer range with equal-width endpoints.

    Assumes:
    - int(str_a) <= int(str_b)
    - len(str_a) == len(str_b)
    """
    str_len = len(str_a)
    if len(str_a) != len(str_b):
        raise (
            Exception(
                f"The input numbers ({str_a}, {str_b}) should have equal number of digits"
            )
        )

    # Three edge cases
    if str_a == str_b:
        return start_appender_str + str_a
    if str_len == 1:
        return f"{start_appender_str}{__digit_range(int(str_a), int(str_b))}"
    # Counting index position till the characteres are equal
    check_equal = -1
    for i in range(str_len):
        if str_a[i] == str_b[i]:
            check_equal += 1
        else:
            break
    if check_equal != -1:
        return __compute_numerical_range(
            str_a[check_equal + 1 :],
            str_b[check_equal + 1 :],
            start_appender_str=start_appender_str + str_a[: check_equal + 1],
        )

    # Example 1: 169 - 543
    # Intermediate range
    intermediate_range = list(range(int(str_a[0]) + 1, int(str_b[0])))
    patterns = []
    if intermediate_range:
        patterns.append(
            f"{start_appender_str}{__digit_range(intermediate_range[0], intermediate_range[-1])}{ANY_DIGIT * (str_len-1)}"
        )
    # patterns for the above part ['[2-4][0-9][0-9]']

    # Case for str_a
    for loop_counter in range(str_len - 1):  # no_of_digits-1 units
        if loop_counter == str_len - 2:  # Find the last loop
            patterns.append(
                f"{start_appender_str}{str_a[:loop_counter+1]}{__digit_range(int(str_a[-1]), 9)}"
            )
        else:
            if (
                str_a[loop_counter + 1] != "9"
            ):  # if 599 then avoid 10 in '[6-8]...|5[10-9]..|59[9-9].|598[9-9]'
                patterns.append(
                    f"{start_appender_str}{str_a[:loop_counter+1]}{__digit_range(int(str_a[loop_counter+1]) + 1, 9)}{ANY_DIGIT * (str_len-2-loop_counter)}"
                )
    # patterns for the above part ['1[7-9][0-9]','16[9-9]']

    # Case for str_b
    for loop_counter in range(str_len - 1):  # no_of_digits-1 units
        if loop_counter == str_len - 2:  # Find the last loop
            patterns.append(
                f"{start_appender_str}{str_b[:loop_counter+1]}{__digit_range(0, int(str_b[-1]))}"
            )
        else:
            if (
                str_b[loop_counter + 1] != "0"
            ):  # if 1102 then avoid -1 in '11[0--1].|110[0-2]'
                patterns.append(
                    f"{start_appender_str}{str_b[:loop_counter+1]}{__digit_range(0, int(str_b[loop_counter+1]) - 1)}{ANY_DIGIT * (str_len-2-loop_counter)}"
                )
    # patterns for the above part ['5[0-3][0-9]','54[0-3]']

    return "|".join(patterns)


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
    # Entire range negative
    if b < 0:  # a <= 0 implicit
        sign = "-"
        a, b = abs(a), abs(b)
        a, b = (a, b) if a < b else (b, a)
    # Entire range positive
    elif a >= 0:  # b >= 0 implicit
        sign = ""
    # Range between negative and positive
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


def _float_range_regex(a: DecimalLike, b: DecimalLike) -> str:
    """
    Generate a regex that matches decimal numbers in the inclusive range [a, b].

    This function is used by ``float_range_regex`` and always emits a decimal
    pattern (for example, values like ``0.0`` or ``1.25``).
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

    # Properly removing floating point and converting to integer
    a, b = (
        "".join([c for c in a_string if c != "."]),
        "".join([c for c in b_string if c != "."]),
    )
    if len(str(a)) < len(str(b)):
        a = a + f"{'0'*(max_num_decimal-num_of_decimal_in_a)}"
    else:
        b = b + f"{'0'*(max_num_decimal-num_of_decimal_in_b)}"
    a, b = int(a), int(b)
    a, b = (a, b) if a < b else (b, a)

    # Generate regex by treating float as integer
    ranges = __range_splitter(a, b)
    intermediate_regex = "|".join(
        [
            __compute_numerical_range(str(r[0]), str(r[1]), start_appender_str=r[2])
            for r in ranges
        ]
    )

    # Modifying the integer supported regex to support float
    new_regex = []
    for p in intermediate_regex.split("|"):
        x = __tokenize_numeric_pattern(p[1:] if p.startswith("-") else p)

        # If x = ['[0-9]'] and max_num_decimal = 2, We need x = ['0','[0-9]']
        if len(x) < max_num_decimal:
            x = (["0"] * (max_num_decimal - len(x))) + x

        # Example x = ['3', '2', '[0-1]', '[0-9]'] for p=32[0-1][0-9]
        start_appender_str = "-" if p.startswith("-") else ""
        # Add a decimal point inbetween, keep the next digit mandatory and others optional (32.[0-1][0-9]?[0-9]*)
        fractional_part = (
            [x[-max_num_decimal]] + [z + "?" for z in x[-max_num_decimal + 1 :]]
            if max_num_decimal > 1
            else [z for z in x[-max_num_decimal:]]
        )
        non_fractional_part = (
            "".join(x[:-max_num_decimal]) if "".join(x[:-max_num_decimal]) else "0?"
        )
        new_regex.append(
            rf"{start_appender_str}{non_fractional_part}\.{''.join(fractional_part)}\d*"
        )
    regex = f"(?:{'|'.join(new_regex)})"
    return regex


def _to_decimal(value: DecimalLike, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise TypeError(
            f"{name} must be int, float, Decimal, or parseable string, got {value!r}"
        ) from exc


def _range_regex(a: int, b: int):
    """Generate a regex that matches integers in the inclusive range [a, b]."""
    a, b = (a, b) if a < b else (b, a)
    ranges = __range_splitter(a, b)
    regex = f"(?:{'|'.join([__compute_numerical_range(str(r[0]),str(r[1]),start_appender_str=r[2]) for r in ranges])})"
    return regex


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
        return r"-?(?:[1-9]\d*|0)"
    if minimum is None:
        if maximum == 0:
            return r"(?:-[1-9]\d*|0)"
        elif maximum > 0:
            upperbound_regex = _range_regex(0, maximum)
            return rf"(?:-[1-9]\d*|{upperbound_regex})"
        else:
            # choose the smallest number with the same number of digits as lowerbound,
            # and allow all negative numbers with strictly more digits
            num_digits = len(str(maximum)) - 1
            lower_bound = -int("".join(["9"] * num_digits))
            range_expression = _range_regex(lower_bound, maximum)
            # now match any number with at least one more digit
            return rf"(?:{range_expression}|-[1-9]\d{{{num_digits}}}\d*)"
    if maximum is None:
        if minimum < 0:
            lowerbound_regex = _range_regex(minimum, 0)
            return rf"(?:{lowerbound_regex}|[1-9]\d*)"
        else:
            # choose the highest number with the same number of digits as upperbound,
            # and allow all numbers with strictly more digits
            num_digits = len(str(minimum))
            upperbound = int("".join(["9"] * num_digits))
            lowerbound_regex = _range_regex(minimum, upperbound)
            # now match any number with at least one more digit
            return rf"(?:{lowerbound_regex}|[1-9]\d{{{num_digits}}}\d*)"
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
    minimum_decimal = _to_decimal(minimum, "minimum")
    maximum_decimal = _to_decimal(maximum, "maximum")

    decimal_regex = _float_range_regex(minimum_decimal, maximum_decimal)
    if strict:
        return decimal_regex

    lower_decimal = minimum_decimal
    upper_decimal = maximum_decimal
    lower_decimal, upper_decimal = (
        (lower_decimal, upper_decimal)
        if lower_decimal < upper_decimal
        else (upper_decimal, lower_decimal)
    )
    int_lower = int(lower_decimal.to_integral_value(rounding=ROUND_CEILING))
    int_upper = int(upper_decimal.to_integral_value(rounding=ROUND_FLOOR))
    if int_lower > int_upper:
        return decimal_regex

    integer_regex = _range_regex(int_lower, int_upper)
    return f"(?:{integer_regex}|{decimal_regex})"
