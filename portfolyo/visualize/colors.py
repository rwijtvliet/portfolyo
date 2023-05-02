"""Creating colors for use in plotting."""

import colorsys
from collections import namedtuple

import matplotlib as mpl
import numpy as np


class Color(namedtuple("RGB", ["r", "g", "b"])):
    """Class to create an rgb color tuple, with additional methods."""

    def lighten(self, value):
        """Lighten the color by fraction ``value`` (between 0 and 1). If < 0, darken."""
        hu, li, sa = np.array(colorsys.rgb_to_hls(*mpl.colors.to_rgb(self)))
        li += value * ((1 - li) if value > 0 else li)
        return Color(*[min(max(p, 0), 1) for p in colorsys.hls_to_rgb(hu, li, sa)])

    def darken(self, value):
        """Darken the color by fraction ``value`` (between 0 and 1)."""
        return self.lighten(self, -value)

    light = property(lambda self: self.lighten(0.3))
    xlight = property(lambda self: self.lighten(0.6))
    dark = property(lambda self: self.lighten(-0.3))
    xdark = property(lambda self: self.lighten(-0.6))


class Colors:
    class General:
        PURPLE = Color(0.549, 0.110, 0.706)
        GREEN = Color(0.188, 0.463, 0.165)
        BLUE = Color(0.125, 0.247, 0.600)
        ORANGE = Color(0.961, 0.533, 0.114)
        RED = Color(0.820, 0.098, 0.114)
        YELLOW = Color(0.945, 0.855, 0.090)
        LBLUE = Color(0.067, 0.580, 0.812)
        LGREEN = Color(0.325, 0.773, 0.082)
        BLACK = Color(0, 0, 0)
        WHITE = Color(1, 1, 1)

    class Wqpr:  # Standard colors when plotting a portfolio
        w = Color(*mpl.colors.to_rgb("#0E524F")).lighten(0.15)
        q = Color(*mpl.colors.to_rgb("#0E524F"))
        r = Color(*mpl.colors.to_rgb("#8B7557"))
        p = Color(*mpl.colors.to_rgb("#E53454"))
