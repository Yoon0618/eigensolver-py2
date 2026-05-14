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
                        choices=["eigenproblem", "time_evolution", "matrix_exponential"])
    parser.add_argument("--dt", type=float, default=Params.dt)

    parser.add_argument("--expm-chunk-steps", type=int, default=Params.expm_chunk_steps,
                        help="number of time samples computed per scipy.sparse.linalg.expm_multiply call")
    
    parser.add_argument("--load-mat", type=str, default=Params.load_mat_path,
                        help="path to .mat file to load, if not specified, compute from scratch")
    
    parser.add_argument("--save_mat", 
                        nargs="?",
                        const=True,
                        default=False,
                        metavar="PATH",
                        help="save matrix data. If PATH is omitted, use an auto-generated filename.",
    )


    parser.add_argument("--suffix", type=str, default=Params.suffix)
    
    return parser

def parse_params():
    parser = make_parser()
    args = parser.parse_args()

    if args.save_mat is False:
        save_mat = False
        save_mat_path = ""
    elif args.save_mat is True:
        save_mat = True
        save_mat_path = ""
    else:
        save_mat = True
        save_mat_path = args.save_mat

    load_mat_path = args.load_mat or ""
    load_mat = bool(load_mat_path)

    param = Params(
        n_start=args.n[0],
        n_delta=args.n[1],
        n_end=args.n[2],
        m=args.m,
        p=args.p,
        basis=args.basis,
        method=args.method,
        dt=args.dt,
        expm_chunk_steps=args.expm_chunk_steps,

        load_mat=load_mat,
        load_mat_path=load_mat_path,
        save_mat=save_mat,
        save_mat_path=save_mat_path,

        suffix=args.suffix,
    )
    return param

def load_mat(path):
    with np.load(path) as data:
        A_matrix = {key: data[key] for key in data.files}

    return A_matrix

def save_mat(A_matrix, path):
    path = param.mat_save_folder_name + "/" + param.file_name + ".npz"
    np.savez_compressed(path, **A_matrix)


def save_result(param, profiles, mode_data, selected_mat_data, solve_data):
    
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
    
    time_like_methods = {"time_evolution", "matrix_exponential"}

    # save plots
    if param.method == "eigenproblem":
        pass
    
    elif param.method in time_like_methods:
        from plot import plot_time_evolution
        plot_time_evolution(param, profiles, solve_data, save=True, show=True)

    from plot import plot_eigenmodes, plot_eigenvalues
    plot_eigenmodes(param, profiles, mode_data, selected_mat_data, solve_data, save=True, show=True)
    eigenvalues_data = plot_eigenvalues(param, profiles, solve_data, save=True, show=True)
    
    # time evolution의 경우 나중에 최종 상태에서 계산을 이어갈 수 있게 최종 상태 저장
    if param.method in time_like_methods:
        F_block_final_state = solve_data.get("F_block_final_state", solve_data.get("F_final_state"))

        if F_block_final_state is not None:
            F_blocks = [np.asarray(F) for F in F_block_final_state]
            F_lengths = np.asarray([F.size for F in F_blocks], dtype=np.int64)
            F_offsets = np.concatenate(([0], np.cumsum(F_lengths))).astype(np.int64)

            eigenvalues_data["F_block_final_state"] = np.concatenate(F_blocks)
            eigenvalues_data["F_block_final_state_lengths"] = F_lengths
            eigenvalues_data["F_block_final_state_offsets"] = F_offsets

        # plot_eigenvalues()가 반환하는 gammas/omegas는 normalize된 값이므로,
        # time-domain solver에서 얻은 raw 값들은 별도 key로 보존한다.
        for key in ("n_values", "ts", "lnFs", "alphas"):
            if key in solve_data:
                eigenvalues_data[key] = np.asarray(solve_data[key])

        if "gammas" in solve_data:
            eigenvalues_data["gammas_raw"] = np.asarray(solve_data["gammas"])
        if "omegas" in solve_data:
            eigenvalues_data["omegas_raw"] = np.asarray(solve_data["omegas"])
        if "fit_info" in solve_data:
            eigenvalues_data["fit_info_json"] = np.asarray(json.dumps(solve_data["fit_info"], ensure_ascii=False))

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
