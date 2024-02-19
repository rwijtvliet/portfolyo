import portfolyo as pf
from portfolyo import tools

pfl = pf.dev.get_nestedpfline()
pfl2 = pf.dev.get_nestedpfline()
pfls = [pfl, pfl2]
children_names_sets = [{name for name in pfl.children} for pfl in pfls]
start_of_day = tools.startofday.get(pfl.index, "str")
print(pfl)
