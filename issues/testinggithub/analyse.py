from collections import defaultdict

packages = defaultdict(dict)
for v in 38, 39, 310:
    print(f"\n\n{v}")
    for line in open(f"{v}.txt").readlines():
        if line.startswith("Requirement already satisfied: "):
            print(line)
            packvers = line[31:-1].split(" ")
            pack, ver = packvers[0], packvers[-1][1:-1]
            if (found := pack.find(">")) > -1:
                pack = pack[:found]
            if (found := pack.find("!")) > -1:
                pack = pack[:found]
            if pack == "c":
                raise Exception()
            packages[pack][v] = ver

    for line in open(f"{v}.txt").readlines():
        if line.startswith("Successfully installed"):
            print(line)
            packvers = line[23:-1].split(" ")
            for packver in packvers:
                splitpoint = -packver[::-1].find("-")
                pack, ver = packver[: splitpoint - 1], packver[splitpoint:]
                packages[pack][v] = ver


# check version diffs
suspicious = {}
for pack, versions in packages.items():
    if len(versions) < 3 or len(set(versions.items())) < 3:
        suspicious[pack] = versions
