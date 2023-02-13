import ptxcompiler
from time import perf_counter


def get_ptx(fname):
    with open(fname, 'r') as f:
        return f.read()


ptxes = [get_ptx(fname) for fname in ('a.ptx', 'b.ptx', 'c.ptx')]
options = ('--gpu-name', 'sm_75', '-c')

start = perf_counter()

for ptx in ptxes:
    ptxcompiler.compile_ptx(ptx, options)

end = perf_counter()

total = end - start
print(total)
