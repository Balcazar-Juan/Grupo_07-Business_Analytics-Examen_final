from pathlib import Path
import sys

# Permite ejecutar este archivo directamente con: python scripts/<archivo>.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import ping, get_db, ensure_indexes
if __name__ == "__main__":
    ping()
    ensure_indexes()
    print("✅ MongoDB Atlas conectado.")
    print("Base:", get_db().name)
    print("Colecciones:", get_db().list_collection_names())
