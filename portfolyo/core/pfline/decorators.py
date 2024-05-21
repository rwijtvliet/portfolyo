"""Module with decorators to catch (and possibly correct) common situations."""

from ... import tools


def assert_longest_allowed_freq(freq):
    def decorator(fn):
        def wrapped(self, *args, **kwargs):
            if tools.freq.up_or_down(self.index.freq, freq) == 1:
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
            if tools.freq.up_or_down(self.index.freq, freq) == -1:
                raise ValueError(
                    "The frequency of the index is too short; shortest allowed:"
                    f" {freq}; passed: {self.index.freq}."
                )
            return fn(self, *args, **kwargs)

        return wrapped

    return decorator
