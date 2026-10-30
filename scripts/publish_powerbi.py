from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import ensure_indexes
from src.etl.powerbi import publish_powerbi_layer

if __name__ == "__main__":
    ensure_indexes()
    result = publish_powerbi_layer()
    print("Capa Power BI actualizada:", result)
