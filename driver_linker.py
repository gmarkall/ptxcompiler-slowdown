from time import perf_counter
from numba.cuda.cudadrv.driver import Driver, Linker


def get_ptx(fname):
    with open(fname, 'r') as f:
        return f.read()


ptxes = [get_ptx(fname) for fname in ('a.ptx', 'b.ptx', 'c.ptx')]
options = ('--gpu-name', 'sm_75', '-c')

driver = Driver()
driver.ensure_initialized()
driver.get_device().get_primary_context().push()

start = perf_counter()

for ptx in ptxes:
    linker = Linker.new(cc=(7, 5))
    linker.add_ptx(ptx.encode())
    linker.complete()

end = perf_counter()

total = end - start
print(total)
