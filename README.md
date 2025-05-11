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

If given numbers are integers you get a regex that will only match with integer and if floating-point numbers are given it only match with floating-point number.

Supports integer and floating-point numbers and negative range.

You can use `geq_regex` and `leq_regex` to generate regex for greater than or equal to and less than or equal to respectively (i.e., ranges without an upper or lower bound).

```python
from range_ex import range_regex, geq_regex, leq_regex
regex1 = range_regex(5,89)
regex2 = range_regex(81.78,250.23)
regex3 = range_regex(-65,12)

regex4 = geq_regex(5)
regex5 = leq_regex(89)
```

Example regex generated for 25-53
```
([3-4][0-9]|2[5-9]|5[0-3])
```


The regex might not be optimal but it will surely serve the purpose.

### Contributing

Contributions are very welcome. Please open an issue or a pull request if you have any suggestions or improvements.

### Acknowledgements

This project is based on [regex_engine](https://github.com/raj-kiran-p/regex_engine). Feel free to check it out.