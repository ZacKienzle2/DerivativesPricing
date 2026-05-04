"""Shared pytest fixtures."""

import os

os.environ.setdefault("DERIVATIVES_PRICING_NO_WARMUP", "1")
