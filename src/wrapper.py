#!/usr/bin/env python3
"""
Thin wrapper so the container / CI can simply call `python -m wrapper`
instead of `python macd_unified.py`
"""
from macd_unified import main
if __name__ == "__main__":
    raise SystemExit(main())
