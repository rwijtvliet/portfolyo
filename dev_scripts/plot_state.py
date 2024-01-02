import matplotlib.pyplot as plt
import portfolyo as pf


"""Plot PfState to test of adding axes for un/sourced prices."""

index = pf.dev.get_index(freq="D")
pfl1 = pf.dev.get_pfline(index, nlevels=2, childcount=2)
pfs = pf.dev.get_pfstate(index)
pfs2 = pfs.asfreq("MS")
# unsourced = pfs.unsourcedprice
# unsourced.print()

# sourced = pfs.sourcedprice
# sourced.print()

pfs2.print()
pfs2.plot()
plt.show()
