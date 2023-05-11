"""Module with decorators to catch (and possibly correct) common situations."""


import warnings

from ... import tools


def assert_longest_allowed_freq(freq):
    def decorator(fn):
        def wrapped(self, *args, **kwargs):
            if tools.freq.longer_or_shorter(self.index.freq, freq) == 1:
                raise ValueError(
                    "The frequency of the index is too long; longest allowed:"
                    f" {freq}; passed: {self.index.freq}."
                )
            return fn(self, *args, **kwargs)

        return wrapped

    return decorator


def assert_shortest_allowed_freq(freq):
    def decorator(fn):
        def wrapped(self, *args, **kwargs):
            if tools.freq.longer_or_shorter(self.index.freq, freq) == -1:
                raise ValueError(
                    "The frequency of the index is too short; shortest allowed:"
                    f" {freq}; passed: {self.index.freq}."
                )
            return fn(self, *args, **kwargs)

        return wrapped

    return decorator


def map_to_year_warning(map_to_year):
    def wrapped(self, *args, **kwargs):
        if tools.freq.shortest(self.index.freq, "MS") == "MS":
            warnings.warn(
                "This PfLine has a monthly frequency or longer; changing the year is inaccurate, as"
                " details (number of holidays, weekends, offpeak hours, etc) cannot be taken into account."
            )
        return map_to_year(self, *args, **kwargs)

    return wrapped
