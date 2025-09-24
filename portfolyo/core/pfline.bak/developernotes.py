# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where, for some timestamp, the volume q == 0 but the revenue
#   r != 0, because this occasionally arises for the sourced volume, e.g. after buying
#   and selling the same volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# Both cases can be catered to. The first as a 'FlatPfLine', where the timeseries for
# q and r are used in the instance creation. The price is not defined at the timestamp in
# the example, but can be calculated for other timestamps, and downsampling is also still
# possble.
# The second is a bit more complex. It is possible as a 'NestedPfLine'. This has then 2
# 'FlatPfLine' instances as its children: one made from each of the timeseries for q
# and p.
