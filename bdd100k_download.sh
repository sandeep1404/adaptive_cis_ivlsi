#!/bin/bash
# bdd100k_download.sh
# Downloads BDD100K via Kaggle API (fastest method, no registration wall)
# Prerequisites: pip install kaggle
# Get your kaggle.json from https://www.kaggle.com/settings → API → Create New Token

set -e

DOWNLOAD_DIR="data/bdd100k_raw"
mkdir -p "$DOWNLOAD_DIR"

echo "========================================================"
echo "  BDD100K Download Script"
echo "========================================================"

# ── Check kaggle CLI is installed ─────────────────────────────────────────────
if ! command -v kaggle &> /dev/null; then
    echo "[→] Installing kaggle CLI..."
    pip install kaggle --quiet
fi

# ── Check kaggle.json exists ──────────────────────────────────────────────────
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "[!] kaggle.json not found. Please do the following:"
    echo "    1. Go to: https://www.kaggle.com/settings"
    echo "    2. Scroll to 'API' section"
    echo "    3. Click 'Create New Token' → downloads kaggle.json"
    echo "    4. Run:"
    echo "         mkdir -p ~/.kaggle"
    echo "         mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "         chmod 600 ~/.kaggle/kaggle.json"
    echo "    5. Re-run this script"
    exit 1
fi

echo "[✓] kaggle.json found"

# ── Download BDD100K images (~5.3 GB) ─────────────────────────────────────────
echo ""
echo "[→] Downloading BDD100K images (~5.3 GB)..."
echo "    This will take 10–30 minutes depending on connection speed."
echo ""

kaggle datasets download -d marquis03/bdd100k \
    --path "$DOWNLOAD_DIR" \
    --unzip

echo ""
echo "[✓] Images downloaded and extracted to: $DOWNLOAD_DIR"

# ── Download BDD100K detection labels (~50 MB) separately ────────────────────
# Labels are inside a separate Kaggle source — download and place:
echo ""
echo "[→] Downloading BDD100K detection labels..."

kaggle datasets download -d solesensei/solesensei_bdd100k \
    --path "$DOWNLOAD_DIR/labels_raw" \
    --unzip

echo ""
echo "[✓] Labels downloaded"

# ── Show final structure ──────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Download complete. Structure:"
echo "========================================================"
echo "  $DOWNLOAD_DIR/"
find "$DOWNLOAD_DIR" -maxdepth 4 -type d | head -20
echo ""
echo "  NEXT: python bdd100k_prepare.py"
