# eigensolver-py

선형 ITG(ion-temperature-gradient) gyrofluid 모델을 spectral basis에서
푸는 Python 연구 코드다.

`run.py`는 Cyclone-like equilibrium을 만들고, monotonic `q(r)` profile에서
공명 조건을 만족하는 `(n, m, p)` 모드를 고른 뒤, Hermite 또는 Bessel radial
basis 위에서 선형 연산자 `A`를 조립한다. 이후 각 toroidal mode block을 따로
풀어 가장 불안정한 모드의 growth rate와 frequency를 구한다.

시간 전개 방법에서는 다음 식을 적분하고,

```text
dF/dt = -i A F
```

late-time 구간의 `ln ||F(t)||`를 fit해서 growth rate를 구한다.

모델 이론은 [`documents/eigenvalue_ITG.md`](documents/eigenvalue_ITG.md) 참고.

## Repository layout

| Path | 역할 |
| --- | --- |
| [`run.py`](run.py) | 실행 진입점. profile, mode, matrix를 만들고 solver와 저장까지 호출한다. |
| [`parameters.py`](parameters.py) | 물리/수치 파라미터의 기본값. |
| [`modes.py`](modes.py) | `(n, m, p)` mode set과 lookup table을 만든다. |
| [`matrices.py`](matrices.py) | basis function과 operator matrix를 조립한다. |
| [`solve.py`](solve.py) | block matrix를 만들고 eigenproblem 또는 time evolution을 푼다. |
| [`plot.py`](plot.py) | growth rate, frequency, eigenmode, time evolution figure를 저장한다. |
| [`sweep.py`](sweep.py) | parameter sweep을 스크립트로 돌리는 예시. |
| [`documents/`](documents) | 모델 유도와 배경 메모. |
| `results/` | 로컬 실행 결과. |

패키지로 설치해서 쓰는 라이브러리는 아니고, 스크립트 중심의 연구 코드다.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy sympy matplotlib
```

## Quick start

smoke run:

```bash
python3 run.py --n 4 4 4 --m 8 --p 2 --basis hermite --method eigenproblem
```

Hermite basis, time evolution:

```bash
python3 run.py --n 4 4 48 --m 150 --p 50 --basis hermite --method time_evolution --dt 1e-2
```

인자 없이 실행 시 [`parameters.py`](parameters.py)의 `Params` 기본값 사용:

```bash
python3 run.py
```

현재 기본값은 다음과 같다.

```text
n = 4:4:48
m = 50
p = 10
basis = hermite
method = time_evolution
dt = 1e-2
T = 30.0
```

`T`, grid size, profile, damping switch, physical constant 대부분은 [`parameters.py`](parameters.py)에서 수정 가능.

## Solver paths

| Method | 동작 | 주 출력 |
| --- | --- | --- |
| `eigenproblem` | 각 `n` block의 eigenvalue problem을 직접 푼다. | block별 most unstable eigenvalue/eigenmode |
| `time_evolution` | RK4로 `dF/dt = -iAF`를 적분한다. | late-time fit에서 얻은 growth rate/frequency |
| `matrix_exponential` | `scipy.sparse.linalg.expm_multiply`를 chunk 단위로 호출한다. | `time_evolution`과 같은 형태의 결과 |

## Command line

| Option | 기본값 | 설명 |
| --- | --- | --- |
| `--n n_start n_delta n_end` | `4 4 48` | toroidal mode number 범위. 예를 들어 `4 4 16`은 `4, 8, 12, 16`을 뜻한다. |
| `--m M` | `50` | poloidal mode number 상한. 후보는 `1..M`. |
| `--p P` | `10` | 선택된 `(n, m)`마다 붙는 radial mode 개수. index는 `0..P-1`. |
| `--basis {hermite,bessel}` | `hermite` | radial basis. |
| `--method {eigenproblem,time_evolution,matrix_exponential}` | `time_evolution` | solver 선택. |
| `--dt DT` | `1e-2` | time-domain solver의 time step. |
| `--expm-chunk-steps N` | `1000` | `matrix_exponential`에서 한 번에 계산할 output step 수. |
| `--suffix TEXT` | empty | *미구현 |
| `--load-mat PATH` | empty | *미구현 |
| `--save_mat [PATH]` | disabled | *미구현 |

## Outputs

결과는 `Params.save_dir` 아래에 저장된다. 기본값은 `results/`다.
`Params.file_name`은 다음 양식으로 만들어진다.

```text
YYYYMMDD_HHMMSS_n{n_start}-{n_delta}-{n_end}_m{m}_p{p}_{basis}_{method}
```

`basis`는 `h` 또는 `b`, `method`는 `eg`, `te`, `me`로 들어간다.

실행 후 다음 파일이 생긴다.

| File | 내용 |
| --- | --- |
| `*_params.json` | 실행에 사용한 `Params` 값. 기본값과 실행 중 채워진 값이 같이 들어간다. |
| `*_eigenvalues.npz` | growth rate/frequency 배열. time-domain run이면 시간 이력도 포함한다. |
| `*_eigenvalues.png` | `k_theta rho_i`에 대한 growth rate와 frequency plot. |
| `*_eigenmodes.png` | 각 `n`에서 dominant potential mode의 real part. |
| `*_time_evolution.png` | time-domain method에서만 저장된다. `ln ||F||`, fit interval, frequency estimate를 보여준다. |

`*_eigenvalues.npz`:

| Key | 의미 |
| --- | --- |
| `k_thetas_rho_i` | plot x-axis에 쓰는 normalized poloidal wave number. |
| `gammas` | plot normalization이 적용된 growth rate. |
| `omegas` | plot normalization이 적용된 frequency. figure에서는 `Frequency/4`로 표시한다. |

`time_evolution` 또는 `matrix_exponential`에서:

| Key | 의미 |
| --- | --- |
| `n_values` | 실제로 계산한 toroidal mode numbers. |
| `ts` | time samples. |
| `lnFs` | 각 `n` block의 log amplitude history. |
| `alphas` | 각 `n` block의 instantaneous complex alpha estimate. |
| `gammas_raw` | plot normalization 전 fitted growth rate. |
| `omegas_raw` | plot normalization 전 fitted frequency. |
| `fit_info_json` | fit 구간, R2, standard deviation 등을 담은 JSON string. |
| `F_block_final_state` | 마지막 time step의 block state를 이어 붙인 배열. |
| `F_block_final_state_lengths` | 각 final-state block의 길이. |
| `F_block_final_state_offsets` | `F_block_final_state` 안에서 각 block이 시작하는 위치. |
