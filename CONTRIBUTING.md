# Contributing to aerosol3d

Thank you for your interest in contributing! This document outlines the process and guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/openEarthModelling/aerosol3d.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=aerosol3d --cov-report=term-missing
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and import sorting. Run before committing:

```bash
ruff check src tests
ruff format src tests
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- aerosol3d version
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback if applicable
