"""Creating colors for use in plotting."""

import colorsys
from collections import namedtuple
from enum import Enum

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
        return self.lighten(-value)

    light = property(lambda self: self.lighten(0.3))
    xlight = property(lambda self: self.lighten(0.6))
    dark = property(lambda self: self.lighten(-0.3))
    xdark = property(lambda self: self.lighten(-0.6))


class Colors:
    class General(Enum):
        DARK_BURGUNDY = Color(0.2667, 0.0000, 0.0745)
        BROWN_SUGAR = Color(0.3765, 0.2902, 0.1804)
        DUSTY_GRAY = Color(0.5529, 0.5176, 0.3765)
        MOONSTONE = Color(0.7647, 0.7569, 0.6118)
        CORAL = Color(0.9608, 0.3412, 0.4980)
        ORANGE = Color(0.9608, 0.5098, 0.1140)
        RED = Color(0.8196, 0.0980, 0.1137)
        YELLOW = Color(0.9451, 0.8549, 0.0902)
        LBLUE = Color(0.0667, 0.5804, 0.8118)
        LGREEN = Color(0.3255, 0.7725, 0.0824)
        BLACK = Color(0.0000, 0.0000, 0.0000)
        WHITE = Color(1.0000, 1.0000, 1.0000)

    class Wqpr:  # Standard colors when plotting a portfolio
        w = Color(*mpl.colors.to_rgb("#0E524F")).lighten(0.15)
        q = Color(*mpl.colors.to_rgb("#0E524F"))
        r = Color(*mpl.colors.to_rgb("#8B7557"))
        p = Color(*mpl.colors.to_rgb("#cd3759"))
