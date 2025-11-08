import sys
maj, minor = sys.version_info[:2]
if not ((maj == 3 and minor in (10, 11))):
    sys.stderr.write(
        f"ERROR: Python {maj}.{minor} detected. Use Python 3.10 or 3.11 for TensorFlow.\n"
    )
    sys.exit(1)
print(f"OK: Python {maj}.{minor}")

