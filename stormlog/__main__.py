"""Module execution support for `python -m stormlog`."""

from .entrypoint import main

if __name__ == "__main__":
    raise SystemExit(main())
