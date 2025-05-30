# utils/file_utils.py
import os
import json
import logging

logger = logging.getLogger(__name__)


def load_json(filepath):
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return {}


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")

