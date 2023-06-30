import collections

LENGTH_UNITS = collections.OrderedDict(
    [
        ("km", 1e-12),
        ("m", 1e-9),
        ("mm", 1e-6),
        ("um", 1e-3),
        ("nm", 1.0),
        ("pm", 1e3),
    ]
)
"""List of physical units of length."""


def ceil_div(a, b):
    """Ceil integer division (``ceil(a / b)`` using integer arithmetic)."""
    return (a - 1) // b + 1


def format_length(length_nm, unit):
    """Format a length according to the provided unit (input in nanometres).

    :param float length_nm: a length in nanometres
    :param str unit: must be one of ``LENGTH_UNITS.keys``
    :return: the formatted length, rounded to the specified unit (no fractional
             part is printed)
    :rtype: str
    """
    return format(length_nm * LENGTH_UNITS[unit], ".0f") + unit
