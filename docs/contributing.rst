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
