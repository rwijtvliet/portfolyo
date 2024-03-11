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
        Color1 = Color(0.2235294117647059, 0.23137254901960785, 0.4745098039215686)
        Color2 = Color(0.3215686274509804, 0.32941176470588235, 0.6392156862745098)
        Color3 = Color(0.4196078431372549, 0.43137254901960786, 0.8117647058823529)
        Color4 = Color(0.611764705882353, 0.6196078431372549, 0.8705882352941177)
        Color5 = Color(0.19215686274509805, 0.5098039215686274, 0.7411764705882353)
        Color6 = Color(0.4196078431372549, 0.6823529411764706, 0.8392156862745098)
        Color7 = Color(0.6196078431372549, 0.792156862745098, 0.8823529411764706)
        Color8 = Color(0.7764705882352941, 0.8588235294117647, 0.9372549019607843)
        Color9 = Color(0.5490196078431373, 0.42745098039215684, 0.19215686274509805)
        Color10 = Color(0.7411764705882353, 0.6196078431372549, 0.2235294117647059)
        Color11 = Color(0.9058823529411765, 0.7294117647058823, 0.3215686274509804)
        Color12 = Color(0.9058823529411765, 0.796078431372549, 0.5803921568627451)
        Color13 = Color(0.5176470588235295, 0.23529411764705882, 0.2235294117647059)
        Color14 = Color(0.6784313725490196, 0.28627450980392155, 0.2901960784313726)
        Color15 = Color(0.8392156862745098, 0.3803921568627451, 0.4196078431372549)
        Color16 = Color(0.9058823529411765, 0.5882352941176471, 0.611764705882353)
        Color17 = Color(0.4823529411764706, 0.2549019607843137, 0.45098039215686275)
        Color18 = Color(0.6470588235294118, 0.3176470588235294, 0.5803921568627451)
        Color19 = Color(0.807843137254902, 0.42745098039215684, 0.7411764705882353)
        Color20 = Color(0.8705882352941177, 0.6196078431372549, 0.8392156862745098)
        Color21 = Color(0.38823529411764707, 0.38823529411764707, 0.38823529411764707)
        Color22 = Color(0.5882352941176471, 0.5882352941176471, 0.5882352941176471)
        Color23 = Color(0.7411764705882353, 0.7411764705882353, 0.7411764705882353)
        Color24 = Color(0.8509803921568627, 0.8509803921568627, 0.8509803921568627)

    class Wqpr:  # Standard colors when plotting a portfolio
        w = Color(*mpl.colors.to_rgb("#0E524F")).lighten(0.15)
        q = Color(*mpl.colors.to_rgb("#0E524F"))
        r = Color(*mpl.colors.to_rgb("#8B7557"))
        p = Color(*mpl.colors.to_rgb("#E53454"))
