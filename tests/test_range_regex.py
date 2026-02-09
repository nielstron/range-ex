import re

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import one_of
import pytest

from range_ex import float_range_regex, range_regex

NUM_EXAMPLES = 5000


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


def _one_decimal_str(value: float) -> str:
    return "0.0" if value == 0 else f"{value:.1f}"


@given(ranges_samples_inside())
@settings(max_examples=NUM_EXAMPLES)
def test_numerical_range(pair):
    (start_range, end_range, value_inside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_inside)) is not None


@given(one_of(ranges_samples_below(), ranges_samples_above()))
@settings(max_examples=NUM_EXAMPLES)
def test_numerical_range_outside(pair):
    (start_range, end_range, value_outside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_outside)) is None


@given(st.integers(), st.integers())
@settings(max_examples=NUM_EXAMPLES)
def test_range_lower_bounded(lower_bound, value):
    generated_regex = range_regex(minimum=lower_bound)
    assert (re.compile(generated_regex).fullmatch(str(value)) is not None) == (
        value >= lower_bound
    )


@given(
    st.integers(),
    st.integers(),
)
@settings(max_examples=NUM_EXAMPLES)
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
    assert range_regex(0, 9) == r"(?:\d)"


def test_redundant_single_value_ranges_are_collapsed():
    generated_regex = range_regex(169, 543)
    assert re.search(r"\[([0-9])-\1\]", generated_regex) is None


@given(float_ranges_and_values())
@settings(max_examples=NUM_EXAMPLES)
def test_float_range(pair):
    (start_range, end_range, value) = pair
    generated_regex = float_range_regex(start_range, end_range)
    matched = re.compile(generated_regex).fullmatch(_one_decimal_str(value)) is not None
    assert matched == (start_range <= value <= end_range)


def test_range_regex_rejects_float_bounds():
    with pytest.raises(TypeError):
        range_regex(0.0, 10)
    with pytest.raises(TypeError):
        range_regex(0, 10.0)


def test_range_regex_does_not_match_decimal_strings():
    assert re.compile(range_regex()).fullmatch("0.0") is None
    assert re.compile(range_regex(-10, 10)).fullmatch("0.0") is None
    assert re.compile(range_regex(0, 0)).fullmatch("0.0") is None
