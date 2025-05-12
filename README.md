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

Pass a minimum and maximum value to the `range_regex` function to generate a regex that matches numbers in that range. The range is inclusive, meaning both the minimum and maximum values are included in the regex.


Supports integer numbers and negative range.

```python
from range_ex import range_regex

regex1 = range_regex(5,89)
# ([5-9]|[2-7][0-9]|1[0-9]|8[0-9])

regex2 = range_regex(-65,12)
# (-[1-9]|-[2-5][0-9]|-1[0-9]|-6[0-5]|[0-9]|1[0-2])
```

> Note: This will still find matches in strings like `1234` or `abc25def53`, so you may want to wrap it in `^` and `$` to match the whole string or `\b...\b` to ensure word boundaries are matched.

If you only pass one of the two arguments, the other will be set to `None`, which means it will not be constrained.
In this case, the regex will match any number that is greater than or equal to the minimum or less than or equal to the maximum.

```python
regex3 = range_regex(minimum=5)
# (([5-9])|[1-9]\\d{1}\\d*)

regex4 = range_regex(maximum=89)
# (-[1-9]\\d*|([0-9]|[2-7][0-9]|1[0-9]|8[0-9]))
```


### Contributing

Contributions are very welcome. Please open an issue or a pull request if you have any suggestions or improvements.

To test your changes, run the following command:

```sh
pytest -n 5
```

### Acknowledgements

This project is based on [regex_engine](https://github.com/raj-kiran-p/regex_engine). Feel free to check it out.