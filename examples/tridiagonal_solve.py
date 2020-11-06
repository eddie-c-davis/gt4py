#!/usr/bin/env python

import sys
import time
import traceback as tb

import numpy as np

import gt4py
from gt4py import gtscript


try:
    import cupy as cp
except ImportError:
    cp = None


# py_backend = 'debug'
cpu_backend = "cxxopt"
gpu_backend = "cuda"
py_backend = "numpy"
# cpu_backend = 'gtmc'
# gpu_backend = 'gtcuda'
rebuild = True  # True
managed_mem = False  # True

Field3D = gtscript.Field[float]


@gtscript.stencil(backend=py_backend, rebuild=rebuild)
def tridiagonal_solve_py(inf: Field3D, diag: Field3D, sup: Field3D, rhs: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            sup = sup / diag
            rhs = rhs / diag
        with interval(1, None):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
    with computation(BACKWARD):
        with interval(-1, None):
            out = rhs
        with interval(0, -1):
            out = rhs - sup * out[0, 0, 1]


@gtscript.stencil(backend=cpu_backend, rebuild=rebuild)
def tridiagonal_solve_mc(inf: Field3D, diag: Field3D, sup: Field3D, rhs: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            sup = sup / diag
            rhs = rhs / diag
        with interval(1, None):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
    with computation(BACKWARD):
        with interval(-1, None):
            out = rhs
        with interval(0, -1):
            out = rhs - sup * out[0, 0, 1]


if cp:

    @gtscript.stencil(backend=gpu_backend, rebuild=rebuild)
    def tridiagonal_solve_cu(
        inf: Field3D, diag: Field3D, sup: Field3D, rhs: Field3D, out: Field3D
    ):
        with computation(FORWARD):
            with interval(0, 1):
                sup = sup / diag
                rhs = rhs / diag
            with interval(1, None):
                sup = sup / (diag - sup[0, 0, -1] * inf)
                rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
        with computation(BACKWARD):
            with interval(-1, None):
                out = rhs
            with interval(0, -1):
                out = rhs - sup * out[0, 0, 1]


def main():
    # Allocate storages
    hsize = int(sys.argv[1])  # vsize * 2
    vsize = int(sys.argv[2]) if len(sys.argv) > 2 else hsize
    shape = (hsize, hsize, vsize)
    nhalo = 3

    inf_data = np.fromfunction(lambda i, j, k: np.sin(i) * np.sin(j), shape, dtype=float)
    diag_data = np.fromfunction(lambda i, j, k: np.tan(i) * np.tan(j), shape, dtype=float)
    sup_data = np.fromfunction(lambda i, j, k: np.cos(i) * np.cos(j), shape, dtype=float)
    rhs_data = np.fromfunction(lambda i, j, k: np.sin(i) * np.cos(j), shape, dtype=float)

    inf_fld_py = gt4py.storage.from_array(
        data=inf_data, backend=py_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    diag_fld_py = gt4py.storage.from_array(
        data=diag_data, backend=py_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    sup_fld_py = gt4py.storage.from_array(
        data=sup_data, backend=py_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    rhs_fld_py = gt4py.storage.from_array(
        data=rhs_data, backend=py_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    out_fld_py = gt4py.storage.zeros(
        shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=py_backend
    )

    inf_fld_mc = gt4py.storage.from_array(
        data=inf_data, backend=cpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    diag_fld_mc = gt4py.storage.from_array(
        data=diag_data, backend=cpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    sup_fld_mc = gt4py.storage.from_array(
        data=sup_data, backend=cpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    rhs_fld_mc = gt4py.storage.from_array(
        data=rhs_data, backend=cpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )
    out_fld_mc = gt4py.storage.zeros(
        shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=cpu_backend
    )

    if cp:
        inf_fld_cu = gt4py.storage.from_array(
            data=inf_data,
            backend=gpu_backend,
            default_origin=(nhalo, nhalo, 0),
            shape=shape,
            managed_memory=managed_mem,
        )
        diag_fld_cu = gt4py.storage.from_array(
            data=diag_data, backend=gpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
        )
        sup_fld_cu = gt4py.storage.from_array(
            data=sup_data, backend=gpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
        )
        rhs_fld_cu = gt4py.storage.from_array(
            data=rhs_data, backend=gpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
        )
        out_fld_cu = gt4py.storage.zeros(
            shape=shape,
            default_origin=(nhalo, nhalo, 0),
            dtype=float,
            backend=gpu_backend,
            managed_memory=managed_mem,
        )

    domain = (hsize - nhalo * 2, hsize - nhalo * 2, vsize)

    t_start = time.perf_counter()
    tridiagonal_solve_py(
        inf_fld_py, diag_fld_py, sup_fld_py, rhs_fld_py, out_fld_py, domain=domain
    )
    t_run = (time.perf_counter() - t_start) * 1e3
    print("%s time: %.6lf" % (py_backend, t_run))
    np.savetxt("tridiag_solve_py_output.csv", out_fld_py.data.flatten(), delimiter=",")

    t_start = time.perf_counter()
    tridiagonal_solve_mc(
        inf_fld_mc, diag_fld_mc, sup_fld_mc, rhs_fld_mc, out_fld_mc, domain=domain
    )
    t_run = (time.perf_counter() - t_start) * 1e3
    print("%s time: %.6lf" % (cpu_backend, t_run))
    np.savetxt("tridiag_solve_cpu_output.csv", out_fld_mc.data.flatten(), delimiter=",")

    if cp:
        t_start = time.perf_counter()
        tridiagonal_solve_cu(
            inf_fld_cu, diag_fld_cu, sup_fld_cu, rhs_fld_cu, out_fld_cu, domain=domain
        )
        t_run = (time.perf_counter() - t_start) * 1e3
        print("%s time: %.6lf" % (gpu_backend, t_run))
        out_fld_cu.synchronize()
        np.savetxt("tridiag_solve_gpu_output.csv", np.asarray(out_fld_cu).flatten(), delimiter=",")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:  # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")
    except Exception as e:
        print("ERROR: " + str(e))
        tb.print_exc()
