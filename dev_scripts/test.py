from typing import List, Tuple, Type
import pandas as pd


def get_allowed_classes(frequencies: List[str]) -> List[Tuple[Type, ...]]:
    """
    Given a list of frequency strings, return a list of unique Method Resolution Orders (MROs)
    associated with their corresponding pandas offset objects.

    Parameters:
    frequencies (List[str]): A list of frequency strings (e.g., ["AS", "QS", "MS", "D", "H", "15T"])

    Returns:
    List[Tuple[Type, ...]]: A list of unique MROs, where each MRO is a tuple of classes representing
                            the inheritance hierarchy of the corresponding offset object.
    """
    unique_classes = []

    for freq in frequencies:
        offset_obj = pd.tseries.frequencies.to_offset(freq)
        mro = offset_obj.__class__.__mro__

        # Add the first class from the MRO to the list of unique classes
        if mro[0] not in unique_classes:
            unique_classes.append(mro[0])

    return unique_classes


FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]
ALLOWED_CLASSES = get_allowed_classes(FREQUENCIES)
print(ALLOWED_CLASSES)
# Get the MROs for the given frequency strings
result_mros = get_allowed_classes(["AS-FEB"])
# print(result_mros)
