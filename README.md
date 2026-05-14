# eigensolver-py 실행 인자

`run.py`는 `utils.parse_params()`에서 command line argument를 읽어 `Params` 객체를 만든 뒤 계산을 실행합니다.

## 기본 실행

```bash
python3 run.py
```

기본값은 `parameters.py`의 `Params` dataclass에 정의되어 있습니다.

## Arguments

| Argument | 기본값 | 형식 / 선택지 | 설명 |
| --- | --- | --- | --- |
| `--n` | `4 12 48` | 정수 3개: `n_start n_delta n_end` | 사용할 toroidal mode number 범위입니다. 실제 값은 `n_start`부터 `n_end`까지 `n_delta` 간격으로 생성됩니다. 예: `4 12 48` -> `4, 16, 28, 40`. |
| `--m` | `50` | 정수 | poloidal mode number의 상한입니다. 실제 후보는 `1`부터 `m`까지입니다. |
| `--p` | `50` | 정수 | 각 `(n, m)` 모드마다 사용할 radial mode 개수입니다. 실제 index는 `0`부터 `p - 1`까지입니다. |
| `--basis` | `hermite` | `bessel`, `hermite` | radial basis 종류를 선택합니다. |
| `--method` | `time_evolution` | `eigenproblem`, `time_evolution` | 해를 구하는 방법을 선택합니다. |
| `--dt` | `1e-4` | 실수 | `time_evolution` 방법에서 사용하는 시간 간격입니다. |
| `--suffix` | `""` | 문자열 | 결과 구분용 suffix 값입니다. 현재 `run.py`에서는 `Params.suffix`에 저장됩니다. |

## 실행 예시

```bash
python3 run.py --n 4 4 16 --m 50 --p 20 --basis hermite --method time_evolution --dt 1e-4
```

```bash
python3 run.py --n 4 12 48 --m 150 --p 10 --basis bessel --method eigenproblem
```

결과는 기본적으로 `results/` 디렉터리에 저장됩니다.

## 출력 파일

`save_result()`는 `Params.save_dir`에 지정된 디렉터리, 기본값 `results/`에 결과를 저장합니다.
`Params.file_name`이 비어 있으면 다음 형식의 기본 파일명이 자동 생성됩니다.

```text
YYYYMMDD_HHMMSS_n{n_start}-{n_end}_dn{n_delta}_m{m}_p{p}_{basis}
```

여기서 `{basis}`는 `bessel`이면 `b`, `hermite`이면 `h`입니다.

### `*_params.json`

실행에 사용된 `Params` 객체의 값을 JSON으로 저장한 파일입니다. CLI에서 넘긴 값과 `parameters.py`의 기본값, 계산 중 채워지는 `dr`, 자동 생성된 `file_name` 등이 포함됩니다.

예:

```text
results/20260512_120408_n4-48_dn12_m50_p10_h_params.json
```

### `*_eigenvalues.npz`

고유값 또는 시간 전개에서 얻은 성장률/진동수 데이터를 NumPy compressed archive 형식으로 저장한 파일입니다. `np.load()`로 읽을 수 있습니다.

항상 포함되는 주요 배열은 다음과 같습니다.

| Key | 설명 |
| --- | --- |
| `k_thetas_rho_i` | 각 `n` 값에 대응하는 normalized poloidal wave number입니다. |
| `gammas` | 플롯에 사용되는 정규화된 growth rate입니다. |
| `omegas` | 플롯에 사용되는 정규화된 frequency 값입니다. 플롯에서는 `Frequency/4`로 표시됩니다. |

`time_evolution` 또는 `matrix_exponential` 방법으로 실행한 경우에는 이어서 분석하거나 재시작할 수 있도록 다음 데이터가 추가됩니다.

| Key | 설명 |
| --- | --- |
| `n_values` | 계산에 사용한 toroidal mode number 배열입니다. |
| `ts` | 시간 샘플 배열입니다. |
| `lnFs` | 각 `n` 모드의 amplitude 로그 값입니다. |
| `alphas` | 각 `n` 모드의 phase 값입니다. |
| `gammas_raw` | time-domain solver가 계산한 정규화 전 growth rate입니다. |
| `omegas_raw` | time-domain solver가 계산한 정규화 전 frequency입니다. |
| `fit_info_json` | growth rate와 frequency fitting에 대한 부가 정보를 JSON 문자열로 저장한 값입니다. |
| `F_block_final_state` | 마지막 시간 스텝의 mode coefficient들을 하나로 이어 붙인 배열입니다. |
| `F_block_final_state_lengths` | `F_block_final_state`에서 각 `n` block의 길이입니다. |
| `F_block_final_state_offsets` | `F_block_final_state`에서 각 `n` block의 시작 offset입니다. |

예:

```python
import json
import numpy as np

with open("results/20260512_120408_n4-48_dn12_m50_p10_h_params.json", encoding="utf-8") as f:
    params = json.load(f)

data = np.load("results/20260512_120408_n4-48_dn12_m50_p10_h_eigenvalues.npz")
print(data.files)
print(data["gammas"])
```
