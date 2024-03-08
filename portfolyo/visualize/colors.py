"""Creating colors for use in plotting."""

import colorsys
from enum import Enum
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
        return self.lighten(-value)

    light = property(lambda self: self.lighten(0.3))
    xlight = property(lambda self: self.lighten(0.6))
    dark = property(lambda self: self.lighten(-0.3))
    xdark = property(lambda self: self.lighten(-0.6))


class Colors:
    class General(Enum):
        Color1 = Color(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        Color2 = Color(0.6823529411764706, 0.7803921568627451, 0.9098039215686274)
        Color3 = Color(1.0, 0.4980392156862745, 0.054901960784313725)
        Color4 = Color(1.0, 0.7333333333333333, 0.47058823529411764)
        Color5 = Color(0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
        Color6 = Color(0.596078431372549, 0.8745098039215686, 0.5411764705882353)
        Color7 = Color(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
        Color8 = Color(1.0, 0.596078431372549, 0.5882352941176471)
        Color9 = Color(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
        Color10 = Color(0.7725490196078432, 0.6901960784313725, 0.8352941176470589)
        Color11 = Color(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
        Color12 = Color(0.7686274509803922, 0.611764705882353, 0.5803921568627451)
        Color13 = Color(0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
        Color14 = Color(0.9686274509803922, 0.7137254901960784, 0.8235294117647058)
        Color15 = Color(0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
        Color16 = Color(0.7803921568627451, 0.7803921568627451, 0.7803921568627451)
        Color17 = Color(0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
        Color18 = Color(0.8588235294117647, 0.8588235294117647, 0.5529411764705883)
        Color19 = Color(0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
        Color20 = Color(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)

    class Wqpr:  # Standard colors when plotting a portfolio
        w = Color(*mpl.colors.to_rgb("#0E524F")).lighten(0.15)
        q = Color(*mpl.colors.to_rgb("#0E524F"))
        r = Color(*mpl.colors.to_rgb("#8B7557"))
        p = Color(*mpl.colors.to_rgb("#E53454"))
