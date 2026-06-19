"""Utility — validates, reads, and normalises CSV/XLSX uploads into a pandas DataFrame."""
import io
import pandas as pd
import chardet
from ..config import ALLOWED_FILE_TYPE, LARGE_FILE_THRESHOLD, SAMPLE_SIZE

def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    # Validate file type
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_FILE_TYPE):
        raise ValueError(f"Unsupported file type. Allowed types are: {ALLOWED_FILE_TYPE}")

    # Detect encoding
    result = chardet.detect(file_bytes)
    encoding = result['encoding'] if result['encoding'] else 'utf-8'

    # Read file into DataFrame
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
    else:  # Excel files
        df = pd.read_excel(io.BytesIO(file_bytes))

    # Normalise DataFrame (e.g., handle missing values, standardise column names)
    df.columns = [col.strip().lower() for col in df.columns]  # Standardise column names

    # Sample down if too large
    if len(df) > LARGE_FILE_THRESHOLD:
        df = df.sample(n=SAMPLE_SIZE, random_state=42) # fallback; stratified sampling happens in the planner when a target_column in known.

    return df