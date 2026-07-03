"""Lookup tables shared by the Silver transforms.

Plain Python dicts derived from src.config.constants. Each transform loads the
ones it needs into DuckDB as temp tables (see db.register_lookup) and LEFT JOINs
against them — the SQL-native equivalent of a dictionary lookup.
"""

from src.config.constants import EVENT_GROUPS, CAMEO_CATEGORIES, FIPS_TO_ISO


def _invert_event_groups() -> dict[str, str]:
    """EVENT_GROUPS is {group: [codes]}; invert to {code: group}."""
    return {code: group for group, codes in EVENT_GROUPS.items() for code in codes}


# CAMEO root code ("01".."20") → high-level group ("violent_conflict", ...)
EVENT_GROUP_BY_CODE: dict[str, str] = _invert_event_groups()

# CAMEO root code → human label ("fight", "consult", ...)
CAMEO_LABEL_BY_CODE: dict[str, str] = dict(CAMEO_CATEGORIES)

# FIPS 10-4 country code → ISO 3166-1 alpha-3
FIPS_BY_CODE: dict[str, str] = dict(FIPS_TO_ISO)
