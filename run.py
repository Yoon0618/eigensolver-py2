import argparse
import numpy as np
from parameters import Params

def main():
    parser = argparse.ArgumentParser()
    
    default = {
        "n_start": Params.n_start,
        "n_delta": Params.n_delta,
        "n_end": Params.n_end,
        "m": Params.m,
        "p": Params.p,
        "basis": Params.basis,
        "method": Params.method,
        "dt": Params.dt,
        "suffix": Params.suffix,
    }

    # n
    parser.add_argument("--n", nargs=3, type=int, default=[default["n_start"], default["n_delta"], default["n_end"]], help="toroidal mode number: n_start, n_delta, n_end")

    # m
    parser.add_argument("--m", type=int, default=default["m"], help="poloidal mode number, [1, 2, 3,... m]")

    # p
    parser.add_argument("--p", type=int, default=default["p"], help="radial mode number, [0, 1, 2,... p-1]")

    # basis
    parser.add_argument("--basis", type=str, default=default["basis"], choices=["bessel", "hermite"], help="basis functions for radial direction")

    # method
    parser.add_argument("--method", type=str, default=default["method"], choices=["eigenproblem", "time_evolution"], help="solution method")

    # dt
    parser.add_argument("--dt", type=float, default=default["dt"], help="time step for time evolution method")

    # file name suffix for saving images
    parser.add_argument("--suffix", type=str, default=default["suffix"], help="file name suffix for saving images")

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

    print(f"parameter: {param}")

    # build profiles
    from parameters import build_profiles
    profiles = build_profiles(param)

    # build modes
    from modes import build_modes
    mode_data = build_modes(param, profiles)
    
    # build matrices
    from matrices import build_matrices
    mat_data = build_matrices(param, profiles, mode_data)

    from solve import construct_A_matrix
    A_matrix = construct_A_matrix(mode_data, mat_data)

    # solve
    # method 1. eigenvalue problem
    if param.method == "eigenproblem":
        from solve import solve_eigenvalue_problem
        solve_data = solve_eigenvalue_problem(A_matrix)

    # method 2. time evolution
    elif param.method == "time_evolution":
        from solve import solve_time_evolution
        solve_data = solve_time_evolution(param, A_matrix)

    # save results
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 저장 경로가 없으면 생성
    import os
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)
    file_name = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}_{date}" # ex) n4_48_m50_p10_bessel_20240601_123456
    save_path = f"{param.save_dir}/{file_name}"
    print(f"saving result as {file_name}")

    # save parameters as json
    import json
    

    # save raw data as npz
    np.savez(f"{save_path}.npz", solve_data=solve_data)

    # save plots
    from plot import plot_eigenvalues, plot_eigenmodes
    plot_eigenvalues(param, profiles, solve_data, save=True, show=True)
    plot_eigenmodes(param, profiles, mode_data, mat_data, solve_data, save=True, show=True)

    # save note.txt for 약간의 메모 남기기
    memo_context = input("메모를 입력하세요 (엔터로 종료): ")
    with open(f"{save_path}_note.txt", "w") as f:
        f.write(memo_context)

if __name__ == "__main__":
    main()