import matplotlib.pyplot as plt
import portfolyo as pf
from portfolyo import dev
from portfolyo.core.pfline.enums import Kind
from portfolyo.core.pfstate.pfstate import PfState


"""Plot PfState to test of adding axes for un/sourced prices."""

index = pf.dev.get_index(freq="D")

pfs = pf.dev.get_pfstate(index)
pfs2 = pfs.asfreq("MS")
offtakevolume = dev.get_nestedpfline(index, kind=Kind.VOLUME)
sourced = dev.get_nestedpfline(index, kind=Kind.COMPLETE)
unsourcedprice = dev.get_nestedpfline(index, kind=Kind.PRICE)
pfs3 = PfState(-1 * offtakevolume, unsourcedprice, sourced)
# pfl2 = create.nestedpfline(index)
# pfs3 = pf.dev.get_nested_pfstate(index)
offtakevolume2 = pfs3.offtakevolume
print(offtakevolume2)
# sourced = pfs.sourcedprice
# sourced.print()

# pfl2.print()

pfs3.plot(children=True)
plt.show()
