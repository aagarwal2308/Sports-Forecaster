"""Project-wide settings: where files live and how seasons are split.

Every function that touches disk takes its directory as an argument and only
*defaults* to these paths, so tests can point everything at a temp folder.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # exact bytes as downloaded; never edited
CLEAN_DIR = DATA_DIR / "clean"    # validated, standardized tables
DB_PATH = DATA_DIR / "lab.sqlite"  # prediction records
JOURNAL_PATH = PROJECT_ROOT / "journal" / "experiments.jsonl"

# Chronological splits, by NFL season (a season is labeled by the year it
# starts; e.g. the 2025 season's Super Bowl was played in Feb 2026).
#   train      -> fit / estimate model parameters
#   validation -> compare models and tune settings
#   test       -> LOCKED. Touched only for a final, pre-declared evaluation.
TRAIN_SEASONS = tuple(range(1999, 2016))       # 1999-2015
VALIDATION_SEASONS = tuple(range(2016, 2021))  # 2016-2020
TEST_SEASONS = tuple(range(2021, 2026))        # 2021-2025
# Seasons after 2025 are "live": only prospective forecasts, scored as games finish.
