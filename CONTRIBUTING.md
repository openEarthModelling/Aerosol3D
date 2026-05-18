# Contributing to Aerosol3D

Thank you for your interest in contributing! This document outlines the process and guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/openEarthModelling/Aerosol3D.git`
3. Install in development mode: `pip install -e ".[dev,docs]"`
4. Install pre-commit hooks: `pre-commit install`

## Development Setup

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

## Running Tests

```bash
pytest
```

Skip Julia-dependent tests:

```bash
SKIP_JULIA_TESTS=1 pytest
```

With coverage:

```bash
pytest --cov=Aerosol3D --cov-report=term-missing
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and import sorting. Run before committing:

```bash
ruff check src tests
ruff format src tests
```

Pre-commit hooks are configured:

```bash
pre-commit install
pre-commit run --all-files
```

## Documentation

Build the Sphinx documentation locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

Open `_build/html/index.html` in your browser to preview.

When adding new modules or public functions, keep the docs in sync:

- **API reference**: Add `.. automodule::` directives to the appropriate `docs/api-reference/*.rst` file
- **User guide**: Update the relevant `docs/user-guide/*.rst` file
- **Tutorials**: Create a new `docs/tutorials/*.rst` file and register it in `docs/tutorials/index.rst`
- **README**: Update Features, API Overview, and Examples sections
- **CHANGELOG**: Add entries under `[Unreleased]` in `CHANGELOG.md`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- Aerosol3D version
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback if applicable
