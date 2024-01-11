import matplotlib.pyplot as plt
import portfolyo as pf


"""Plot PfState to test of adding axes for un/sourced prices."""

index = pf.dev.get_index(freq="D")
pfs = pf.dev.get_pfstate(index)
pfs2 = pfs.asfreq("MS")

# pfl2 = create.nestedpfline(index)
# pfs3 = pf.dev.get_nested_pfstate(index)


# sourced = pfs.sourcedprice
# sourced.print()

# pfl2.print()
pfs2.plot()
plt.show()
