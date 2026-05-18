Contributing
============

Development Setup
-----------------

Clone the repository and install in editable mode::

    git clone https://github.com/openEarthModelling/Aerosol3D.git
    cd Aerosol3D
    pip install -e ".[dev,docs]"

Running Tests
-------------

Run the full test suite::

    pytest

Skip Julia-dependent tests::

    SKIP_JULIA_TESTS=1 pytest

Code Style
----------

This project uses `ruff` for linting and formatting::

    ruff check src tests
    ruff format src tests

Pre-commit hooks are configured::

    pre-commit install
    pre-commit run --all-files

Documentation
-------------

Build documentation locally::

    cd docs
    sphinx-build -b html . _build/html

Open ``_build/html/index.html`` in your browser to preview.

Updating Documentation
-----------------------

When adding new modules or public functions, keep the docs in sync:

- **API reference**: Add ``.. automodule::`` directives to the appropriate
  ``docs/api-reference/*.rst`` file (e.g., ``optics.rst`` for new optics
  submodules).
- **User guide**: Update the relevant ``docs/user-guide/*.rst`` file to describe
  new features or changed behavior.
- **Tutorials**: Create a new ``docs/tutorials/*.rst`` file, add a
  ``.. literalinclude::`` to the example script, and register it in
  ``docs/tutorials/index.rst``.
- **README**: Update the Features, API Overview, and Examples sections to
  reflect user-facing changes.
- **CHANGELOG**: Add entries under ``[Unreleased]`` in ``CHANGELOG.md``.

After updating, rebuild the docs to verify::

    cd docs && sphinx-build -b html . _build/html
