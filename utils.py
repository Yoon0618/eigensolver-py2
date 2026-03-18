import argparse
from parameters import Params
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", nargs=3, type=int, default=[Params.n_start, Params.n_delta, Params.n_end],
                        help="toroidal mode number: n_start, n_delta, n_end")
    parser.add_argument("--m", type=int, default=Params.m,
                        help="poloidal mode number upper bound")
    parser.add_argument("--p", type=int, default=Params.p,
                        help="radial mode number count")
    parser.add_argument("--basis", type=str, default=Params.basis,
                        choices=["bessel", "hermite"])
    parser.add_argument("--method", type=str, default=Params.method,
                        choices=["eigenproblem", "time_evolution"])
    parser.add_argument("--dt", type=float, default=Params.dt)
    parser.add_argument("--suffix", type=str, default=Params.suffix)

    return parser


def parse_params():
    parser = make_parser()
    args = parser.parse_args()

    param = Params(
        n_start=args.n[0],
        n_delta=args.n[1],
        n_end=args.n[2],
        m=args.m,
        p=args.p,
        basis=args.basis,
        method=args.method,
        dt=args.dt,
        suffix=args.suffix,
    )
    return param

def save_result(param, profiles, mode_data, mat_data, solve_data):
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 저장 경로가 없으면 생성
    import os
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)
    file_name = f"n{param.n_start}-{param.n_end}_m{param.m}_p{param.p}_{param.basis}_{date}" # ex) n4_48_m50_p10_bessel_20240601_123456
    save_path = f"{param.save_dir}/{file_name}"
    print(f"saving result as {file_name}")

    # save parameters as json
    import json
    with open(f"{save_path}.json", "w", encoding="utf-8") as f:
        json.dump(param.__dict__, f, indent=4)

    # save raw data as npz
    np.savez_compressed(f"{save_path}.npz", solve_data=solve_data)

    # save plots
    from plot import plot_eigenvalues, plot_eigenmodes
    plot_eigenvalues(param, profiles, solve_data, save=True, show=True)
    plot_eigenmodes(param, profiles, mode_data, mat_data, solve_data, save=True, show=True)

    # save note.txt for 약간의 메모 남기기
    memo_context = input("메모를 입력하세요 (엔터로 종료): ")
    with open(f"{save_path}_note.txt", "w") as f:
        f.write(memo_context)

import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[TIMER] {func.__name__:<24} {t1 - t0:10.6f} s")
        return out
    return wrapper