#!/bin/bash
# Archive a completed competition

set -e

SLUG=$1
if [ -z "$SLUG" ]; then
    echo "Usage: $0 <competition-slug>"
    exit 1
fi

COMP_DIR="competitions/$SLUG"
if [ ! -d "$COMP_DIR" ]; then
    echo "Error: Competition directory not found: $COMP_DIR"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_NAME="${SLUG}-archive-${TIMESTAMP}.tar.gz"
ARCHIVE_DIR="archives"

mkdir -p "$ARCHIVE_DIR"

echo "Archiving $SLUG..."
tar -czf "$ARCHIVE_DIR/$ARCHIVE_NAME" \
    "$COMP_DIR"

echo "✓ Archived to: $ARCHIVE_DIR/$ARCHIVE_NAME"
echo "✓ Size: $(du -h "$ARCHIVE_DIR/$ARCHIVE_NAME" | cut -f1)"
echo ""
echo "Archive contains:"
echo "  - challenge_config.json"
echo "  - SKILL_STATE.json"
echo "  - data/raw/"
echo "  - data/processed/"
echo "  - reports/"
echo "  - submissions/"
