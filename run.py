
def main():
    # parse parameters
    from utils import parse_params
    param = parse_params()
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

    # # sparse check
    # import numpy as np
    # print(f"A matrix shape: {A_matrix['A'].shape}, nnz: {np.count_nonzero(A_matrix['A'])}, density: {np.count_nonzero(A_matrix['A'])/A_matrix['A'].size:.2e}")
    # from plot import plot_matrices
    # plot_matrices([A_matrix["A"].real, A_matrix["A"].imag], ["Real part of A", "Imaginary part of A"])

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
    from utils import save_result
    save_result(param, profiles, mode_data, mat_data, solve_data)

if __name__ == "__main__":
    main()