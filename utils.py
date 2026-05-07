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
    
    # 저장 경로가 없으면 생성
    import os
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)
    
    if param.basis == "bessel":
        basis = "b"
    elif param.basis == "hermite":
        basis = "h"
    
    # 빈 값으로 되어 있는 파일명 base를 템플릿에 맞추어 자동으로 생성
    if param.file_name == "":
        from datetime import datetime
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        param.file_name = (
            f"{date}_n{param.n_start}-{param.n_end}"
            f"_dn{param.n_delta}"
            f"_m{param.m}_p{param.p}_{basis}"
        )
    
    save_path = f"{param.save_dir}/{param.file_name}"
    print(f"saving result at {save_path} ...")

    # save parameters as json
    import json
    with open(f"{save_path}_params.json", "w", encoding="utf-8") as f:
        json.dump(param.__dict__, f, indent=4)
    
    # save plots
    if param.method == "eigenproblem":
        pass
    
    elif param.method == "time_evolution":
        from plot import plot_time_evolution
        plot_time_evolution(param, profiles, solve_data, save=True, show=True)

    from plot import plot_eigenmodes, plot_eigenvalues
    plot_eigenmodes(param, profiles, mode_data, mat_data, solve_data, save=True, show=True)
    eigenvalues_data = plot_eigenvalues(param, profiles, solve_data, save=True, show=True)
    
    # time evolution의 경우 나중에 최종 상태에서 계산을 이어갈 수 있게 최종 상태 저장
    if param.method == "time_evolution":
        F_block_final_state = solve_data["F_block_final_state"]
        eigenvalues_data["F_block_final_state"] = F_block_final_state

    # save eigenvalues data as npz
    np.savez_compressed(
        f"{save_path}_eigenvalues.npz",
        **{f"{k}": v for k, v in eigenvalues_data.items()}
    )

    # # save note.txt, 약간의 메모 남기기
    # memo_context = input("메모를 입력하세요 (엔터로 종료): ")
    # with open(f"{save_path}_note.txt", "w") as f:
    #     f.write(memo_context)

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

# def downsample(x, l):
#     x = np.asarray(x, dtype=float)
#     L = len(x)
#     edges = np.linspace(0, L, l + 1)
#     csum = np.concatenate(([0.0], np.cumsum(x)))
#     integral = np.interp(edges, np.arange(L + 1), csum)
#     return np.diff(integral) / np.diff(edges)