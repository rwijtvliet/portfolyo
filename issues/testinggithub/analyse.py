from collections import defaultdict

FILES = "39finalok", "39firstnok"
FILES = "38", "39", "310"

packages = defaultdict(dict)
for file in FILES:
    print(f"\n\n{file}")
    for line in open(f"{file}.txt").readlines():
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
            packages[pack][file] = ver

    for line in open(f"{file}.txt").readlines():
        if line.startswith("Successfully installed"):
            print(line)
            packvers = line[23:-1].split(" ")
            for packver in packvers:
                splitpoint = -packver[::-1].find("-")
                pack, ver = packver[: splitpoint - 1], packver[splitpoint:]
                packages[pack][file] = ver


# check version diffs
suspicious_missing = {}
suspicious_diffrnt = {}
for pack, versions in packages.items():
    if len(versions) < len(FILES):
        suspicious_missing[pack] = versions
    if len(set(versions.values())) > 1:
        suspicious_diffrnt[pack] = versions
suspicious = {"missing": suspicious_missing, "different": suspicious_diffrnt}
