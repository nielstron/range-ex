from decimal import Decimal, InvalidOperation
import re

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import one_of
import pytest

from range_ex import float_range_regex, range_regex

NUM_EXAMPLES = 1000


@st.composite
def ranges_samples(draw):
    lower_bound = draw(st.integers())
    upper_bound = draw(st.integers(min_value=lower_bound))
    return (lower_bound, upper_bound)


@st.composite
def ranges_samples_inside(draw):
    lower_bound, upper_bound = draw(ranges_samples())
    inside = draw(st.integers(min_value=lower_bound, max_value=upper_bound))
    return (lower_bound, upper_bound, inside)


@st.composite
def ranges_samples_below(draw):
    lower_bound, upper_bound = draw(ranges_samples())
    outside = draw(st.integers(max_value=lower_bound - 1))
    return (lower_bound, upper_bound, outside)


@st.composite
def ranges_samples_above(draw):
    lower_bound, upper_bound = draw(ranges_samples())
    outside = draw(st.integers(min_value=upper_bound + 1))
    return (lower_bound, upper_bound, outside)


@st.composite
def float_ranges_and_values(draw):
    lower_bound = draw(st.integers(min_value=-10000, max_value=10000))
    upper_bound = draw(st.integers(min_value=lower_bound, max_value=10000))
    value = draw(st.integers(min_value=-10000, max_value=10000))
    return (lower_bound / 10, upper_bound / 10, value / 10)


@st.composite
def optional_float_bounds_and_integer(draw):
    minimum_tenths = draw(st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)))
    maximum_tenths = draw(st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)))
    integer_value = draw(st.integers(min_value=-2000, max_value=2000))
    return minimum_tenths, maximum_tenths, integer_value


@given(ranges_samples_inside())
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_numerical_range(pair):
    (start_range, end_range, value_inside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_inside)) is not None


@given(one_of(ranges_samples_below(), ranges_samples_above()))
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_numerical_range_outside(pair):
    (start_range, end_range, value_outside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_outside)) is None


@given(st.integers(), st.integers())
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_range_lower_bounded(lower_bound, value):
    generated_regex = range_regex(minimum=lower_bound)
    assert (re.compile(generated_regex).fullmatch(str(value)) is not None) == (
        value >= lower_bound
    )


@given(
    st.integers(),
    st.integers(),
)
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_range_upper_bounded(upper_bound, value):
    generated_regex = range_regex(maximum=upper_bound)
    assert (re.compile(generated_regex).fullmatch(str(value)) is not None) == (
        value <= upper_bound
    )


@given(
    st.integers(),
)
@settings(max_examples=NUM_EXAMPLES)
def test_range_no_bound(value):
    generated_regex = range_regex()
    assert re.compile(generated_regex).fullmatch(str(value)) is not None


def test_single_digit_class_uses_shorthand():
    assert range_regex(0, 9) == r"\d"


def test_redundant_single_value_ranges_are_collapsed():
    generated_regex = range_regex(169, 543)
    assert re.search(r"\[([0-9])-\1\]", generated_regex) is None


def test_range_regex_with_reversed_bounds_is_empty():
    compiled = re.compile(range_regex(5, 4))
    assert compiled.fullmatch("4") is None
    assert compiled.fullmatch("5") is None


def test_range_regex_capturing_rendering_toggle():
    assert "(?:" in range_regex(-65, 12, capturing=False)
    assert "(?:" not in range_regex(-65, 12, capturing=True)
    assert "(" in range_regex(-65, 12, capturing=True)


@given(float_ranges_and_values())
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_float_range(pair):
    (start_range, end_range, value) = pair
    generated_regex = float_range_regex(start_range, end_range, strict=True)
    matched = re.compile(generated_regex).fullmatch(f"{value:.1f}") is not None
    assert matched == (start_range <= value <= end_range)


@given(optional_float_bounds_and_integer(), st.booleans())
@settings(max_examples=NUM_EXAMPLES, deadline=None)
def test_float_range_dot_integer_forms_follow_numeric_bounds(data, strict):
    minimum_tenths, maximum_tenths, integer_value = data
    minimum = None if minimum_tenths is None else minimum_tenths / 10
    maximum = None if maximum_tenths is None else maximum_tenths / 10

    generated_regex = float_range_regex(minimum=minimum, maximum=maximum, strict=strict)
    matched = re.compile(generated_regex).fullmatch(f"{integer_value}.") is not None

    if minimum is not None and maximum is not None and minimum > maximum:
        assert matched is False
        return

    value = Decimal(integer_value)
    lower_ok = minimum is None or value >= Decimal(str(minimum))
    upper_ok = maximum is None or value <= Decimal(str(maximum))
    assert matched == (lower_ok and upper_ok)


def test_range_regex_does_not_match_decimal_strings():
    assert re.compile(range_regex()).fullmatch("0.0") is None
    assert re.compile(range_regex(-10, 10)).fullmatch("0.0") is None
    assert re.compile(range_regex(0, 0)).fullmatch("0.0") is None


def test_float_range_strict_requires_decimal_point():
    generated_regex = float_range_regex(0, 2, strict=True)
    compiled = re.compile(generated_regex)
    assert compiled.fullmatch("1") is None
    assert compiled.fullmatch("1.0") is not None


def test_float_range_unbounded_strict_accepts_dot_edge_forms():
    compiled = re.compile(float_range_regex(strict=True))
    assert compiled.fullmatch("1.") is not None
    assert compiled.fullmatch(".1") is not None


def test_float_range_bounded_strict_accepts_trailing_dot_for_integers_in_range():
    compiled = re.compile(float_range_regex(-2, 2, strict=True))
    assert compiled.fullmatch("-2.") is not None
    assert compiled.fullmatch("-1.") is not None
    assert compiled.fullmatch("0.") is not None
    assert compiled.fullmatch("1.") is not None
    assert compiled.fullmatch("2.") is not None
    assert compiled.fullmatch("-3.") is None
    assert compiled.fullmatch("3.") is None


def test_float_range_bounded_strict_accepts_trailing_dot_for_inner_integer():
    compiled = re.compile(float_range_regex(0.5, 1.5, strict=True))
    assert compiled.fullmatch("1.") is not None


def test_float_range_minimum_only_strict_accepts_trailing_dot_integers():
    compiled = re.compile(float_range_regex(minimum=2, strict=True))
    assert compiled.fullmatch("2.") is not None
    assert compiled.fullmatch("3.") is not None
    assert compiled.fullmatch("1.") is None


def test_float_range_maximum_only_strict_accepts_trailing_dot_integers():
    compiled = re.compile(float_range_regex(maximum=-2, strict=True))
    assert compiled.fullmatch("-2.") is not None
    assert compiled.fullmatch("-3.") is not None
    assert compiled.fullmatch("-1.") is None


def test_float_range_non_strict_matches_int_and_decimal():
    generated_regex = float_range_regex(0, 2, strict=False)
    compiled = re.compile(generated_regex)
    assert compiled.fullmatch("1") is not None
    assert compiled.fullmatch("1.5") is not None


def test_float_range_capturing_rendering_toggle():
    assert "(?:" in float_range_regex(0.5, 1.5, strict=True, capturing=False)
    assert "(?:" not in float_range_regex(0.5, 1.5, strict=True, capturing=True)
    assert "(" in float_range_regex(0.5, 1.5, strict=True, capturing=True)


def test_float_range_supports_decimal_bounds_strict():
    generated_regex = float_range_regex(Decimal("0.5"), Decimal("1.5"), strict=True)
    compiled = re.compile(generated_regex)
    assert compiled.fullmatch("1.0") is not None
    assert compiled.fullmatch("1") is None


def test_float_range_supports_decimal_bounds_non_strict():
    generated_regex = float_range_regex(Decimal("0.5"), Decimal("1.5"), strict=False)
    compiled = re.compile(generated_regex)
    assert compiled.fullmatch("1") is not None
    assert compiled.fullmatch("1.25") is not None


def test_float_range_supports_parseable_string_bounds():
    generated_regex = float_range_regex("0.5", "1.5", strict=False)
    compiled = re.compile(generated_regex)
    assert compiled.fullmatch("1") is not None
    assert compiled.fullmatch("1.25") is not None


def test_float_range_non_strict_minimum_decimal_accepts_trailing_dot_higher_integers():
    compiled = re.compile(float_range_regex(minimum="1.5", strict=False))
    assert compiled.fullmatch("2.") is not None
    assert compiled.fullmatch("3.") is not None


def test_float_range_minimum_decimal_includes_9_point_1():
    compiled = re.compile(float_range_regex(minimum=1.5))
    assert compiled.fullmatch("9.1") is not None


def test_float_range_rejects_values_above_upper_bound_with_extra_fractional_digits():
    compiled = re.compile(float_range_regex(0.5, 1.5))
    assert compiled.fullmatch("1.51") is None


def test_float_range_rejects_values_above_upper_bound_with_extra_digits_from_integer_minimum():
    compiled = re.compile(float_range_regex(1, 1.5))
    assert compiled.fullmatch("1.51") is None

def test_pos_float_range_accepts_dot():
    compiled = re.compile(float_range_regex(0.1, 1.5))
    assert compiled.fullmatch(".51") is not None

def test_float_range_rejects_non_parseable_string_bounds():
    with pytest.raises(InvalidOperation):
        float_range_regex("foo", "1.5")


def test_float_range_with_reversed_bounds_is_empty():
    compiled = re.compile(float_range_regex(1.5, 1.0))
    assert compiled.fullmatch("1.0") is None
    assert compiled.fullmatch("1.5") is None
