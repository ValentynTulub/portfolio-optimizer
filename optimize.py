"""Thin entry point — preserves `python optimize.py …` after the package split."""

from portfolio_optimizer.cli import main

if __name__ == "__main__":
    main()
