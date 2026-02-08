"""Registry mapping (record_type, subrecord_type) → translatable.

Defines which subrecords contain user-visible strings that should be translated.
"""

from __future__ import annotations

# Subrecord types that are ALWAYS translatable regardless of parent record type
_ALWAYS_TRANSLATABLE_SUBS: set[bytes] = set()

# Record types where FULL (display name) is visible to the player.
# Many record types have FULL but it's internal/editor-only (EXPL, PROJ,
# WATR, ARMA, etc.).  Only translate FULL in records where the name is
# shown in the HUD, Pip-Boy, inventory, dialogue, menus, or map.
# Source: fopdoc (tes5edit.github.io/fopdoc/Fallout3) + GECK wiki.
_FULL_ALLOWED_RECORDS: set[bytes] = {
    b"ACTI",  # Activator — crosshair prompt ("Activate X")
    b"ALCH",  # Ingestible — Pip-Boy inventory
    b"AMMO",  # Ammunition — Pip-Boy inventory
    b"ARMO",  # Armor — Pip-Boy inventory
    b"AVIF",  # Actor Value Info — SPECIAL/skills screen
    b"BOOK",  # Book — Pip-Boy inventory / reading screen
    b"CELL",  # Cell — HUD location display, Pip-Boy map
    b"CLAS",  # Class — character creation / Pip-Boy stats
    b"COBJ",  # Constructible Object — crafting menu
    b"CONT",  # Container — crosshair prompt ("Search X")
    b"CREA",  # Creature — HUD crosshair, V.A.T.S.
    b"DIAL",  # Dialog Topic — dialogue topic choices
    b"DOOR",  # Door — crosshair prompt ("Open X")
    b"ENCH",  # Object Effect — effect name on enchanted items
    b"EYES",  # Eyes — character creation
    b"FACT",  # Faction — Pip-Boy reputation display
    b"FURN",  # Furniture — crosshair prompt ("Sit X", "Sleep X")
    b"HAIR",  # Hair — character creation
    b"HDPT",  # Head Part — character creation (beards, scars, eyepatch)
    b"INGR",  # Ingredient — Pip-Boy inventory
    b"KEYM",  # Key — Pip-Boy inventory
    b"LIGH",  # Light — some are carryable inventory items
    b"MESG",  # Message — in-game popups / notifications
    b"MGEF",  # Magic Effect — Pip-Boy effect descriptions
    b"MISC",  # Miscellaneous Item — Pip-Boy inventory
    b"MSTT",  # Moveable Static — crosshair prompt (Tranquility Lane objects)
    b"NOTE",  # Note/Holotape — Pip-Boy inventory
    b"NPC_",  # NPC — HUD crosshair, V.A.T.S., dialogue
    b"PERK",  # Perk — Pip-Boy perk list, level-up screen
    b"QUST",  # Quest — Pip-Boy quest list, HUD tracker
    b"RACE",  # Race — character creation, Pip-Boy stats
    b"REFR",  # Placed Reference — map marker names on Pip-Boy world map
    b"SPEL",  # Actor Effect — Pip-Boy effects screen
    b"TACT",  # Talking Activator — crosshair prompt
    b"TERM",  # Terminal — crosshair prompt, terminal header
    b"WEAP",  # Weapon — Pip-Boy inventory
    b"WRLD",  # Worldspace — Pip-Boy world map header
}

# Mapping: subrecord_type → set of record types where it's translatable.
# An empty set means translatable in ANY record type.
_TRANSLATABLE_MAP: dict[bytes, set[bytes]] = {
    b"FULL": _FULL_ALLOWED_RECORDS,
    b"DESC": {
        b"AVIF", b"BOOK", b"WEAP", b"ARMO", b"ALCH", b"PERK", b"SPEL",
        b"MESG", b"CLAS", b"RACE", b"FACT", b"MISC", b"AMMO",
        b"ENCH", b"MGEF", b"LSCR", b"QUST", b"TERM",
    },
    b"NAM1": {b"INFO"},  # Dialog response text
    b"RNAM": {b"INFO", b"TERM"},  # Dialog prompt / terminal menu text
    b"TNAM": {b"NOTE"},  # Note/holotape text
    b"NNAM": {b"QUST"},  # Quest objective text
    b"ITXT": {b"MESG"},  # Message button text
    b"CNAM": {b"QUST"},  # Quest stage log entry
}

# Subrecord types that should NEVER be translated
_NEVER_TRANSLATE: set[bytes] = {
    b"EDID",  # Editor ID
    b"SCPT",  # Script reference
    b"SCHR",  # Script header
    b"SCDA",  # Compiled script data
    b"SCTX",  # Script source text
    b"MODL",  # Model filename
    b"ICON",  # Icon filename
    b"MICO",  # Small icon filename
}


def is_translatable(record_type: bytes, subrecord_type: bytes) -> bool:
    """Check if a subrecord should be translated given its parent record type."""
    if subrecord_type in _NEVER_TRANSLATE:
        return False

    if subrecord_type not in _TRANSLATABLE_MAP:
        return False

    allowed_records = _TRANSLATABLE_MAP[subrecord_type]
    # Empty set means translatable in any record
    if not allowed_records:
        return True

    return record_type in allowed_records


def get_translatable_subrecord_types() -> list[bytes]:
    """Return all subrecord types that can be translatable."""
    return list(_TRANSLATABLE_MAP.keys())
