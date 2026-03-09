#!/usr/bin/env bash
set -euo pipefail

echo "Checking Vector Search resources from config.py..."
python scripts/create_vector_search.py
echo "Done. Existing resources were reused; missing resources were created."
