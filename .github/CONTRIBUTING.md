# Contributing to NeuralCompression

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

We will periodically release updates to the core `neuralcompression` package to
PyPI.

## Building

Using `pip`, you can install the package in development mode by running:

```sh
pip install -e .
pip install -r dev-requirements.txt
```

## Testing

We test the package using `pytest`, which you can run locally by typing

```sh
pytest tests
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

We use `black` for formatting. We ask for type hints for all code committed to
`neuralcompression` and check this with `mypy`. Imports should be sorted with
`isort`. We also lint with `flake8`. The CI system should check of this when
you submit your pull requests.

## License

By contributing to NeuralCompression, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
