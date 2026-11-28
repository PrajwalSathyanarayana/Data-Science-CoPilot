"""Shared test setup. Runs before any test module imports the backend."""
import os

# config.py requires a download signing key; tests use a fixed one when .env has none (the real key is never needed)
os.environ.setdefault("DOWNLOAD_SIGNING_KEY", "test-signing-key-not-a-secret")
