import re

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import one_of

from range_ex import range_regex


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


@given(ranges_samples_inside())
@settings(max_examples=10000)
def test_numerical_range(pair):
    (start_range, end_range, value_inside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_inside)) is not None


@given(one_of(ranges_samples_below(), ranges_samples_above()))
@settings(max_examples=10000)
def test_numerical_range_outside(pair):
    (start_range, end_range, value_outside) = pair
    generated_regex = range_regex(start_range, end_range)
    assert re.compile(generated_regex).fullmatch(str(value_outside)) is None


@given(st.integers(max_value=-1), st.integers())
@settings(max_examples=10000)
def test_range_lower_bounded(lower_bound, value):
    generated_regex = range_regex(minimum=lower_bound)
    assert (re.compile(generated_regex).fullmatch(str(value)) is not None) == (
        value >= lower_bound
    )


@given(
    st.integers(min_value=1),
    st.integers(),
)
@settings(max_examples=10000)
def test_range_upper_bounded(upper_bound, value):
    generated_regex = range_regex(maximum=upper_bound)
    assert (re.compile(generated_regex).fullmatch(str(value)) is not None) == (
        value <= upper_bound
    )


@given(
    st.integers(),
)
@settings(max_examples=10000)
def test_range_no_bound(value):
    generated_regex = range_regex()
    assert re.compile(generated_regex).fullmatch(str(value)) is not None
