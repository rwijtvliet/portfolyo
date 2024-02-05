import portfolyo as pf
import pandas as pd

index = pd.date_range("2020", "2024", freq="QS", inclusive="left")


# FlatPfline
def flat_test():
    pfl1 = pf.dev.get_flatpfline(index)
    # pfl2 = pf.dev.get_flatpfline(index)
    # pfl3 = pfl1
    # pfl4 = pfl2
    # pfl1 = pfl1.loc[:"2021"]
    # pfl3 = pfl2.slice[:"2022"]
    # print("Loc version", pfl1)
    # print("\n")
    # print("Slice version", pfl3)
    # print(pfl1 == pfl3)
    pfls_to_concat = [pfl1.slice[:"2022"], pfl1.slice["2022":]]
    pfls_to_concat2 = [pfl1.loc[:"2021"], pfl1.loc["2022":]]

    print("The slice version", pfls_to_concat)
    print("\n")
    print("Loc version", pfls_to_concat2)
    print(pfls_to_concat == pfls_to_concat2)


def nested_test():
    pfl1 = pf.dev.get_nestedpfline(index, childcount=2)
    pfl2 = pf.dev.get_nestedpfline(index, childcount=2)

    pfls_to_concat = [pfl1.slice[:"2022"], pfl2.slice["2022":]]
    pfls_to_concat2 = [pfl1.loc[:"2021"], pfl2.loc["2022":]]

    print("The slice version", pfls_to_concat)
    print("\n")
    print("Loc version", pfls_to_concat2)
    print(pfls_to_concat == pfls_to_concat2)


def state_test():
    pfs = pf.dev.get_pfstate(index)
    pfs_to_concat2 = [pfs.loc[:"2021"], pfs.loc["2022":]]
    pfs_to_concat = [pfs.slice[:"2022"], pfs.slice["2022":]]

    print("The slice version", pfs_to_concat)
    print("\n")
    print("Loc version", pfs_to_concat2)
    print(pfs_to_concat == pfs_to_concat2)


flat_test()
# nested_test()
# state_test()
