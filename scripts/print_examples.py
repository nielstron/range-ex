from range_ex import float_range_regex, range_regex


def build_examples() -> list[tuple[str, str]]:
    return [
        ("range_regex(0, 9)", range_regex(0, 9)),
        ("range_regex(169, 543)", range_regex(169, 543)),
        ("range_regex(-65, 12)", range_regex(-65, 12)),
        ("range_regex(minimum=5)", range_regex(minimum=5)),
        ("range_regex(maximum=89)", range_regex(maximum=89)),
        ("range_regex()", range_regex()),
        (
            "float_range_regex(0.5, 1.5, strict=True)",
            float_range_regex(0.5, 1.5, strict=True),
        ),
        (
            "float_range_regex(0.5, 1.5, strict=False)",
            float_range_regex(0.5, 1.5, strict=False),
        ),
        (
            "float_range_regex(maximum='1.5', strict=True)",
            float_range_regex(maximum="1.5", strict=True),
        ),
        (
            "float_range_regex(minimum='1.5', strict=False)",
            float_range_regex(minimum="1.5", strict=False),
        ),
        ("float_range_regex(strict=True)", float_range_regex(strict=True)),
        (
            "range_regex(-65, 12, capturing=True)",
            range_regex(-65, 12, capturing=True),
        ),
    ]


def main() -> None:
    for label, regex in build_examples():
        print(label)
        print(f"  {regex}")
        print()


if __name__ == "__main__":
    main()
