Range-Ex
=======
[![Tests & QA](https://github.com/nielstron/range-ex/actions/workflows/tests.yml/badge.svg)](https://github.com/nielstron/range-ex/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/nielstron/range-ex/badge.svg?branch=master)](https://coveralls.io/github/nielstron/range-ex?branch=master)
[![PyPI version](https://badge.fury.io/py/range-ex.svg)](https://pypi.org/project/range-ex/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/range-ex.svg)
[![PyPI - Status](https://img.shields.io/pypi/status/range-ex.svg)](https://pypi.org/project/range-ex/)

This tool builds a regular expression for a numerical range.


### Installation

```sh
pip install range-ex
```


<!-- USAGE EXAMPLES -->
## Usage

Use `range_regex` for integer ranges and `float_range_regex` for decimal ranges.

### Integer ranges (`range_regex`)

`range_regex` matches only integers (including negatives). The range is inclusive on both ends.

```python
from range_ex import range_regex

regex1 = range_regex(5, 89)
# (?:[5-9]|[2-7]\d|1\d|8\d)

regex2 = range_regex(-65, 12)
# (?:\-[1-9]|\-[2-5]\d|\-1\d|\-6[0-5]|\d|1[0-2])
```

If you pass only one bound, the other is unbounded:

```python
regex3 = range_regex(minimum=5)
# (?:[5-9]|[1-9]\d(?:\d)*)

regex4 = range_regex(maximum=89)
# (?:\-[1-9](?:\d)*|\d|[2-7]\d|1\d|8\d)

regex5 = range_regex()
# (?:\-[1-9](?:\d)*|[1-9](?:\d)*|0)
```

### Decimal ranges (`float_range_regex`)

`float_range_regex` accepts `int`, `float`, `Decimal`, and decimal-parseable `str` bounds.

```python
from range_ex import float_range_regex

regex6 = float_range_regex(0.5, 1.5, strict=True)
# (?:(?:0)?\.[5-9](?:\d)*|1\.[0-5](?:\d)*)

regex7 = float_range_regex(0.5, 1.5, strict=False)
# (?:1|(?:0)?\.[5-9](?:\d)*|1\.[0-5](?:\d)*)

regex8 = float_range_regex(maximum="1.5", strict=True)
# (?:\-[1-9](?:\d)*\.\d(?:\d)*|(?:0)?\.\d(?:\d)*|1\.[0-5](?:\d)*)

regex9 = float_range_regex(strict=True)
# (?:(?:\-)?(?:0|[1-9](?:\d)*)\.\d(?:\d)*|(?:\-)?(?:0|[1-9](?:\d)*)\.|(?:\-)?\.\d(?:\d)*)
```

### Rendering options

Both APIs support `capturing=True` to emit capturing groups (`(...)`) instead of non-capturing groups (`(?:...)`):

```python
range_regex(-65, 12, capturing=True)
# (\-[1-9]|\-[2-5]\d|\-1\d|\-6[0-5]|\d|1[0-2])
```

> Note: Generated regexes can still match inside larger strings (for example, inside `abc25def`). Use anchors like `^...$` if you need whole-string matches.


### Contributing

Contributions are very welcome. Please open an issue or a pull request if you have any suggestions or improvements.

To test your changes, run the following command:

```sh
pytest -n 5
```

### Acknowledgements

This project is based on [regex_engine](https://github.com/raj-kiran-p/regex_engine). Feel free to check it out.
