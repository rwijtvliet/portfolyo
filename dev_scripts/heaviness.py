# import cProfile
# import portfolyo


# def main():
#     # Your code here
#     pass


# if __name__ == "__main__":
#     cProfile.run("main()", sort="cumulative")

import timeit

t = timeit.Timer("import portfolyo")
print(t.timeit(number=1000000))
t2 = timeit.Timer("import pandas")
print(t2.timeit(number=1000000))
