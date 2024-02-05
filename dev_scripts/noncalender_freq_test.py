import portfolyo as pf

index = pf.dev.get_index(freq="D")
pfl1 = pf.dev.get_flatpfline(index)
pfl2 = pfl1.asfreq("AS-APR")
