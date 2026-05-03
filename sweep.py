from parameters import Params, build_profiles
from modes import build_modes
from matrices import build_matrices
from solve import construct_A_matrix, solve_eigenvalue_problem, solve_time_evolution
from utils import save_result

# Parameters to sweep
p_sweep_values = [30, 40, 50]

for p in p_sweep_values:
    # Update parameters
    param = Params(
            p=p,
            method="time_evolution",
            # suffix=f"_mu1{mu1}_p{p}",
        )

    profiles = build_profiles(param)
    mode_data = build_modes(param, profiles)
    mat_data = build_matrices(param, profiles, mode_data)
    A_matrix = construct_A_matrix(mode_data, mat_data)

    if param.method == "eigenproblem":
        solve_data = solve_eigenvalue_problem(A_matrix)
    elif param.method == "time_evolution":
        solve_data = solve_time_evolution(param, A_matrix)

    save_result(param, profiles, mode_data, mat_data, solve_data)