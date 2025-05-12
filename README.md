Range-Ex
=======
[![Build Status](https://app.travis-ci.com/nielstron/range-ex.svg?branch=master)](https://app.travis-ci.com/nielstron/range-ex)
[![PyPI version](https://badge.fury.io/py/range-ex.svg)](https://pypi.org/project/range-ex/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/range-ex.svg)
[![PyPI - Status](https://img.shields.io/pypi/status/range-ex.svg)](https://pypi.org/project/range-ex/)

This tool builds a regular expression for your numerical range.



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Coded With Language](#coded-with-language)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



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
regex2 = range_regex(-65,12)

regex3 = range_regex(minimum=5)
regex4 = range_regex(maximum=89)
```

Example regex generated for 25-53
```
([3-4][0-9]|2[5-9]|5[0-3])
```

> Note: This will still find matches in strings like `1234` or `abc25def53`, so you may want to wrap it in `^` and `$` to match the whole string or `\b...\b` to ensure word boundaries are matched.

### Contributing

Contributions are very welcome. Please open an issue or a pull request if you have any suggestions or improvements.

### Acknowledgements

This project is based on [regex_engine](https://github.com/raj-kiran-p/regex_engine). Feel free to check it out.