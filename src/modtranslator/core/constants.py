"""Constants for the TES4 binary format used by Fallout 3, Fallout NV, and Skyrim."""

from enum import IntEnum, IntFlag

# Magic type tags
GRUP_TYPE = b"GRUP"
TES4_TYPE = b"TES4"

# Record header sizes (bytes)
RECORD_HEADER_SIZE_FO3 = 24  # Type(4) + DataSize(4) + Flags(4) + FormID(4) + VCS1(4) + VCS2(4)

# GRUP header is always 24 bytes across all games
GRUP_HEADER_SIZE = 24  # "GRUP"(4) + GroupSize(4) + Label(4) + GroupType(4) + Stamp(4) + VCS(4)

# Subrecord header: Type(4) + Size(2)
SUBRECORD_HEADER_SIZE = 6

# HEDR version floats to detect game
HEDR_VERSION_FO3 = 0.94      # Fallout 3 and New Vegas
HEDR_VERSION_SKYRIM = 1.70   # Skyrim (LE and SE)


class Game(IntEnum):
    """Detected game based on HEDR version."""
    UNKNOWN = 0
    FALLOUT3 = 2   # Also covers Fallout NV (same format)
    SKYRIM = 5     # Skyrim (LE and SE)


class RecordFlag(IntFlag):
    """Common record flags."""
    MASTER = 0x00000001
    COMPRESSED = 0x00040000
    LOCALIZED = 0x00000080  # Skyrim+ only, not used in FO3/FNV


# Default string encoding for FO3/FNV
DEFAULT_ENCODING = "cp1252"  # Windows-1252
