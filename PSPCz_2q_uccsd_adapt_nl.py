# PSPCz (HOMO–LUMO 2-qubit, 6-31G(d)) — VQD×2: UCCSD(lib, tapered-equivalent 2q) → real_amplitudes(func)
# Python 3.11.13 / Qiskit 2.1.1 / NumPy 2.3.x / SciPy 1.16.x / Matplotlib 3.10.x
# Env: macOS (Apple M3 Pro), conda env: TADF_211

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.optimize import minimize

# =============================
# 실행/최적화/시각화 전역 설정
# =============================
# 난수 고정(재현성)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# VQD penalty(직교화 항의 가중치)
PENALTY_BETA = 3.0

# SciPy minimize 옵션
OPTIMIZATION_METHOD = "COBYLA"   # COBYLA 권장(제약 없는 실수 파라미터 최적화)
MAXIMUM_ITERATION = 10000
COBYLA_RHOBEG = 0.2
ATTEMPT_COUNT = 2                # 동일 상태를 서로 다른 초기값으로 여러 번 시도하여 최적값 선택

# Stage 2(real_amplitudes) 하이퍼파라미터
RA_ENTANGLEMENT = "linear"       # {'linear', 'full', ...} Qiskit real_amplitudes 인자
RA_REPS = 2                      # 반복 깊이
INIT_EPS = 0                  # zeros 초기화 시 작은 jitter 크기 (1e-4 등)

# 회로 시각화 제어(전역 토글)
DRAW_CIRCUITS = True             # 회로 그림 생성 여부
SHOW_CIRCUIT_FIGURES = True      # 생성한 회로 그림을 화면에 표시할지 여부

# Matplotlib 백엔드: 화면 표시(off면 Agg, on이면 TkAgg)
import matplotlib
matplotlib.use('TkAgg' if SHOW_CIRCUIT_FIGURES else 'Agg')
import matplotlib.pyplot as plt

# 회로 도식 스타일
FIGSIZE = (10, 5)                # 가로/세로(inch)
FONTSIZE = 7                     # 기본 폰트 크기
FOLD = -1                        # 한 줄에 접는 게이트 수(가로 폭 제어)

# 계산할 상태 개수(인덱스 0/1/2 → S0/T1/S1로 해석)
NUM_STATES = 3

# 내부 figure 레지스트리
_OPEN_FIGS: List[plt.Figure] = []
def _register_fig(fig: Optional[plt.Figure]) -> None:
    """생성된 matplotlib Figure를 내부 리스트에 등록."""
    if fig is not None:
        _OPEN_FIGS.append(fig)

def show_all_figures_blocking() -> None:
    """
    생성된 모든 회로 그림을 한 번에 처리.
    - SHOW_CIRCUIT_FIGURES=True: plt.show() 호출로 화면 표시(블로킹)
    - SHOW_CIRCUIT_FIGURES=False: 화면 미표시, 생성만 하고 정리
    """
    if not _OPEN_FIGS:
        return
    if SHOW_CIRCUIT_FIGURES:
        plt.show()
    plt.close('all')
    _OPEN_FIGS.clear()

# =============================
# PSPCz 2-qubit 해밀토니안(행렬/라벨)
# =============================
def format_hamiltonian_equation(h):
    """
    2-qubit 유효 Hamiltonian을 사람이 읽기 좋은 수식 문자열로 변환.
    입력: 계수 dict (h_II, h_ZI, h_ZZ, h_XI, h_XX)
    """
    return (
        "H = h_II·II + h_ZI·(ZI - IZ) + h_ZZ·ZZ + h_XX·XX + h_XI·(XI + IX + XZ - ZX)\n"
        f"H = {h['h_II']:+.6f}·II  {h['h_ZI']:+.6f}·ZI  {-h['h_ZI']:+.6f}·IZ  "
        f"{h['h_ZZ']:+.6f}·ZZ  {h['h_XX']:+.6f}·XX  {h['h_XI']:+.6f}·XI  "
        f"{h['h_XI']:+.6f}·IX  {h['h_XI']:+.6f}·XZ  {-h['h_XI']:+.6f}·ZX  (Ha)"
    )

def build_h_pspcz():
    """
    PSPCz(HOMO–LUMO, 6-31G(d))에서 얻은 2-qubit 유효 Hamiltonian을 구성.
    - 반환: (라벨/계수/행렬을 보유한 경량 연산자 객체, 계수 dict)
    """
    coeffs = {"h_II": -0.518418, "h_ZI": -0.136555, "h_ZZ": -0.025866,
              "h_XI": -0.000296, "h_XX": 0.015725}

    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1],[1, 0]], dtype=complex)
    Z  = np.array([[1, 0],[0,-1]], dtype=complex)
    kron = np.kron

    # H = h_II·II + h_ZI·(ZI - IZ) + h_ZZ·ZZ + h_XX·XX + h_XI·(XI + IX + XZ - ZX)
    H = (
            coeffs["h_II"] * kron(I2, I2) +
            coeffs["h_ZI"] * (kron(Z, I2) - kron(I2, Z)) +
            coeffs["h_ZZ"] * kron(Z, Z) +
            coeffs["h_XX"] * (kron(X, X)) +
            coeffs["h_XI"] * (kron(X, I2) + kron(I2, X) + kron(X, Z) - kron(Z, X))
    )

    labels = ["II", "ZI", "IZ", "ZZ", "XX", "XI", "IX", "XZ", "ZX"]
    c = [coeffs["h_II"], coeffs["h_ZI"], -coeffs["h_ZI"], coeffs["h_ZZ"],
         coeffs["h_XX"], coeffs["h_XI"], coeffs["h_XI"], coeffs["h_XI"], -coeffs["h_XI"]]

    class Op2:
        """라벨/계수/행렬을 최소한으로 보관하는 경량 연산자 래퍼."""
        def __init__(self, labels, coeffs, mat):
            self.labels = labels
            self.coeffs = np.array(coeffs, dtype=complex)
            self._mat = mat
            self.num_qubits = 2
        def to_matrix(self) -> np.ndarray:
            return self._mat

    return Op2(labels, c, H), coeffs

def label_to_matrix(lbl: str) -> np.ndarray:
    """
    2-qubit Pauli 라벨(예: 'XZ')을 밀도행렬 연산용 4x4 ndarray로 변환.
    - 지원: I, X, Y, Z
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    base = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    return np.kron(base[lbl[0]], base[lbl[1]])

# =============================
# 표 출력 유틸(단계별/최종 비교)
# =============================
def _stats(err_µ: np.ndarray) -> Tuple[float, float, float]:
    """에러(µHa)의 중앙값, 최댓값, RMSE를 계산."""
    med = float(np.median(np.abs(err_µ)))
    mx  = float(np.max(np.abs(err_µ)))
    rmse = float(np.sqrt(np.mean(err_µ**2)))
    return med, mx, rmse

def print_table_stage(header: str, exact: np.ndarray, calc: List[float]) -> None:
    """
    단일 Stage의 결과 테이블을 출력.
    - 열: Exact(Ha), Calc(Ha), Error(µHa)
    - ΔE_ST(S1 - T1) 및 에러(µHa) 추가 표기
    - 통계치(µHa): median/max/RMSE
    """
    calc_arr = np.array(calc[:len(exact)], dtype=float)
    ex_arr = np.array(exact[:len(calc_arr)], dtype=float)
    err_µ = (calc_arr - ex_arr) * 1e6

    print(f"\n[{header}]")
    print(" State |     Exact (Ha) |      Calc (Ha) |  Error (µHa)")
    print("-------|----------------|----------------|-------------")
    for i, (ex, ca, eµ) in enumerate(zip(ex_arr, calc_arr, err_µ)):
        sign = "+" if eµ >= 0 else "-"
        print(f" {i:>5d} | {ex:14.8f} | {ca:14.8f} | {sign}{abs(eµ):12.3f}")

    if len(ex_arr) >= 3:
        exact_gap = float(ex_arr[2] - ex_arr[1])
        calc_gap  = float(calc_arr[2] - calc_arr[1])
        gap_err_µ = (calc_gap - exact_gap) * 1e6
        s = "+" if gap_err_µ >= 0 else "-"
        print("-------|----------------|----------------|-------------")
        print(f" ΔE_ST | {exact_gap:14.8f} | {calc_gap:14.8f} | {s}{abs(gap_err_µ):12.3f}")

    med, mx, rmse = _stats(err_µ)
    print("-------|----------------|----------------|-------------")
    print(f" median|                |                | {med:12.3f}")
    print(f" max   |                |                | {mx:12.3f}")
    print(f" RMSE  |                |                | {rmse:12.3f}")

def print_table_combined(header: str, exact: np.ndarray, calc1: List[float], calc2: List[float]) -> None:
    """
    Stage1/Stage2를 함께 비교하는 테이블 출력.
    - 각 상태에 대해 Exact, Stage1, Stage2, 그리고 각 에러(µHa)
    - ΔE_ST 비교 및 Stage별 통계(µHa)
    """
    ex = np.array(exact, dtype=float)
    c1 = np.array(calc1[:len(ex)], dtype=float)
    c2 = np.array(calc2[:len(ex)], dtype=float)
    e1_µ = (c1 - ex) * 1e6
    e2_µ = (c2 - ex) * 1e6

    print(f"\n[{header}]")
    print(" State |     Exact (Ha) | Stage1_Calc (Ha) | Stage1_Error (µHa) | Stage2_Calc (Ha) | Stage2_Error (µHa)")
    print("-------|----------------|------------------|---------------------|------------------|---------------------")
    for i in range(len(ex)):
        s1 = "+" if e1_µ[i] >= 0 else "-"
        s2 = "+" if e2_µ[i] >= 0 else "-"
        print(f" {i:>5d} | {ex[i]:14.8f} | {c1[i]:16.8f} | {s1}{abs(e1_µ[i]):19.3f} | {c2[i]:16.8f} | {s2}{abs(e2_µ[i]):19.3f}")

    if len(ex) >= 3:
        ex_gap = float(ex[2] - ex[1])
        c1_gap = float(c1[2] - c1[1])
        c2_gap = float(c2[2] - c2[1])
        e1_gap_µ = (c1_gap - ex_gap) * 1e6
        e2_gap_µ = (c2_gap - ex_gap) * 1e6
        s1 = "+" if e1_gap_µ >= 0 else "-"
        s2 = "+" if e2_gap_µ >= 0 else "-"
        print("-------|----------------|------------------|---------------------|------------------|---------------------")
        print(f" ΔE_ST | {ex_gap:14.8f} | {c1_gap:16.8f} | {s1}{abs(e1_gap_µ):19.3f} | {c2_gap:16.8f} | {s2}{abs(e2_gap_µ):19.3f}")

    med1, mx1, rmse1 = _stats(e1_µ)
    med2, mx2, rmse2 = _stats(e2_µ)
    print("-------|----------------|------------------|---------------------|------------------|---------------------")
    print(f" median|                |                  | {med1:19.3f} |                  | {med2:19.3f}")
    print(f" max   |                |                  | {mx1:19.3f} |                  | {mx2:19.3f}")
    print(f" RMSE  |                |                  | {rmse1:19.3f} |                  | {rmse2:19.3f}")

# =============================
# Ansatz: (1) UCCSD 등가 2q, (2) real_amplitudes
# =============================
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RXXGate, RYYGate, real_amplitudes

@dataclass
class AnsatzBuilder:
    """
    ansatz 생성기
    - mode='uccsd_lib_equiv_2q': 2e-2o UCCSD의 Z2-taper 등가 2-qubit 단일 파라미터 ansatz
      U(θ) = exp(-i θ/2 · (XX + YY))이며, |HF>=|10>를 초기 상태로 둔다.
    - mode='realamplitudes': Stage 1 결과 회로를 frozen하여 그 뒤에 real_amplitudes를 붙여 미세 보정
    """
    mode: str
    frozen_initial: Optional[QuantumCircuit] = None
    ra_reps: int = RA_REPS
    ra_entanglement: str = RA_ENTANGLEMENT

    num_parameters: Optional[int] = None
    eval_circuit: Optional[QuantumCircuit] = None
    draw_circuit_template: Optional[QuantumCircuit] = None

    def _build_uccsd_lib_equiv_2q(self) -> Tuple[QuantumCircuit, Parameter]:
        """
        2e-2o UCCSD의 테이퍼 등가(2-qubit) ansatz를 직접 구성.
        - 하드웨어 효율적 표현: RXX(θ)와 RYY(θ) 동일 파라미터를 직렬 배치
        - 초기상태: |10> (HF 점유)
        """
        q = QuantumCircuit(2)
        q.x(1)  # |10>
        theta = Parameter("theta_ucc")
        q.append(RXXGate(theta), [1, 0])
        q.append(RYYGate(theta), [1, 0])
        return q, theta

    def _build_ra(self) -> QuantumCircuit:
        """Qiskit 제공 real_amplitudes 함수형 빌더 사용(최종 단일회전 포함)."""
        return real_amplitudes(
            num_qubits=2,
            reps=self.ra_reps,
            entanglement=self.ra_entanglement,
            insert_barriers=False,
        )

    def create(self, num_qubits: int) -> QuantumCircuit:
        """
        mode에 따라 평가용 회로(eval_circuit)와 도식용 템플릿(draw_circuit_template)을 구성.
        - 'uccsd_lib_equiv_2q': 파라미터 1개
        - 'realamplitudes': frozen_initial 이후 RA 블록 추가
        """
        if self.mode == 'uccsd_lib_equiv_2q':
            qc, _ = self._build_uccsd_lib_equiv_2q()
            self.eval_circuit = qc
            self.draw_circuit_template = qc
            self.num_parameters = 1

        elif self.mode == 'realamplitudes':
            if self.frozen_initial is None:
                raise ValueError("'realamplitudes' 모드는 frozen_initial 회로가 필요합니다.")
            ra = self._build_ra()

            # 평가용: frozen_initial ∘ RA(decompose)
            qc = QuantumCircuit(num_qubits)
            qc.compose(self.frozen_initial, inplace=True)
            qc.barrier()
            qc.compose(ra.decompose(), inplace=True)
            self.eval_circuit = qc

            # 도식용: RA는 원형 그대로(가독성)
            draw = QuantumCircuit(num_qubits)
            draw.compose(self.frozen_initial, inplace=True)
            draw.barrier()
            draw.compose(self._build_ra(), inplace=True)
            self.draw_circuit_template = draw

            self.num_parameters = ra.num_parameters
        else:
            raise ValueError("Unknown ansatz mode.")

        return self.eval_circuit  # type: ignore[return-value]

# =============================
# VQD(Statevector 기반)
# =============================
class VQD:
    """
    Variational Quantum Deflation(VQD) 구현(정확한 Statevector 기대값/오버랩 사용)
    - 목적함수(ground):  <H> + β·Σ |<ψ(θ)|φ_ext>|^2
    - 목적함수(excited): <H> + β·[Σ_k |<ψ(θ)|ψ_k>|^2 + Σ |<ψ(θ)|φ_ext>|^2]
      (ψ_k: 이전에 찾은 고유상태 근사, φ_ext: 외부 참조 상태)
    """
    def __init__(
            self,
            h_op,
            num_qubits: int,
            beta: float = PENALTY_BETA,
            seed: int = GLOBAL_SEED,
            ansatz: Optional[AnsatzBuilder] = None,
            init_mode: str = 'random',                       # {'random', 'zeros'}
            external_reference_states: Optional[List[QuantumCircuit]] = None,
    ):
        self.H = h_op
        self.num_qubits = num_qubits
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.ansatz = ansatz or AnsatzBuilder(mode='uccsd_lib_equiv_2q')
        self.circuit = self.ansatz.create(self.num_qubits)
        self.init_mode = init_mode
        self.ext_refs = external_reference_states or []
        self.energies: List[float] = []
        self.params_list: List[np.ndarray] = []

    def _init_theta(self, size: int) -> np.ndarray:
        """
        초기 파라미터 생성.
        - 'zeros': 0 근처의 작은 난수로 시작(미세 보정용)
        - 'random': [-π, π] 균등분포
        """
        if self.init_mode == 'zeros':
            return INIT_EPS * self.rng.standard_normal(size)
        return self.rng.uniform(-np.pi, np.pi, size=size)

    def _circ_bound(self, params: np.ndarray) -> QuantumCircuit:
        """현재 ansatz에 주어진 파라미터를 바인딩한 회로 반환."""
        return self.circuit.assign_parameters(params)

    def _energy(self, params: np.ndarray) -> float:
        """라벨/계수로 표현된 H에 대해 |ψ(θ)>의 기대값 ⟨H⟩를 정확 계산."""
        sv = Statevector(self._circ_bound(params))
        expv = 0.0 + 0.0j
        for lbl, c in zip(self.H.labels, self.H.coeffs):
            P = label_to_matrix(lbl)
            expv += c * (sv.data.conj().T @ (P @ sv.data))
        return float(np.real(expv))

    def _overlap2_params(self, a: np.ndarray, b: np.ndarray) -> float:
        """두 파라미터 a,b에 대한 |⟨ψ(a)|ψ(b)⟩|^2."""
        sa = Statevector(self._circ_bound(a))
        sb = Statevector(self._circ_bound(b))
        return float(abs(complex(sa.inner(sb))) ** 2)

    def _overlap2_ext(self, params: np.ndarray, ref: QuantumCircuit) -> float:
        """현재 상태와 외부 참조 회로 ref의 중첩 제곱 |⟨ψ(θ)|φ_ext⟩|^2."""
        s = Statevector(self._circ_bound(params))
        r = Statevector(ref)
        return float(abs(complex(s.inner(r))) ** 2)

    def _obj_ground(self, params: np.ndarray) -> float:
        """바닥상태 목적함수."""
        pen = 0.0
        for ref in self.ext_refs:
            pen += self._overlap2_ext(params, ref)
        return self._energy(params) + self.beta * pen

    def _obj_excited(self, params: np.ndarray) -> float:
        """여기상태 목적함수(이전에 찾은 상태들과 직교화)."""
        pen = 0.0
        for prev in self.params_list:
            pen += self._overlap2_params(params, prev)
        for ref in self.ext_refs:
            pen += self._overlap2_ext(params, ref)
        return self._energy(params) + self.beta * pen

    def _minimize(self, fun, x0: np.ndarray):
        """SciPy minimize 래퍼."""
        if OPTIMIZATION_METHOD.upper() == 'COBYLA':
            options = {'maxiter': MAXIMUM_ITERATION, 'rhobeg': COBYLA_RHOBEG, 'disp': False}
        else:
            options = {'maxiter': MAXIMUM_ITERATION, 'disp': False}
        return minimize(fun, x0, method=OPTIMIZATION_METHOD, options=options)

    def _run_single(self, objective, label: str) -> Tuple[float, np.ndarray]:
        """
        동일 상태를 여러 번 초기화해 최적의 결과를 선택.
        - 실패(res.success=False) 시 해당 시도는 무시하고 다음 시도로 진행.
        """
        best_e = np.inf
        best_x: Optional[np.ndarray] = None
        for _ in range(ATTEMPT_COUNT):
            x0 = self._init_theta(self.ansatz.num_parameters or 0)
            res = self._minimize(objective, x0)
            if not res.success:
                continue
            e = self._energy(res.x)
            if e < best_e:
                best_e, best_x = e, np.array(res.x, dtype=float)
        if best_x is None:
            raise RuntimeError(f"Failed to optimize {label} state.")
        return best_e, best_x

    def find_ground_state(self) -> bool:
        """바닥상태 탐색 및 기록."""
        e, x = self._run_single(self._obj_ground, "ground")
        self.energies.append(e)
        self.params_list.append(x)
        print(f"  - State 0: {e:.9f} Ha")
        return True

    def find_excited_state(self) -> bool:
        """다음 여기상태 탐색 및 기록."""
        idx = len(self.energies)
        e, x = self._run_single(self._obj_excited, f"excited {idx}")
        self.energies.append(e)
        self.params_list.append(x)
        print(f"  - State {idx}: {e:.9f} Ha")
        return True

    def draw_circuit(self, state_idx: int = 0, title: Optional[str] = None) -> Optional[plt.Figure]:
        """
        특정 상태의 파라미터를 바인딩한 회로를 MPL Figure로 렌더링.
        - DRAW_CIRCUITS=False면 호출하지 않는 것을 권장(상위 유틸에서 체크)
        - SHOW_CIRCUIT_FIGURES=False여도 Figure는 생성/반환되며, 화면에는 표시하지 않음
        """
        if state_idx >= len(self.params_list):
            return None
        circ = self.ansatz.draw_circuit_template.assign_parameters(self.params_list[state_idx])
        if title is None:
            title = f"State {state_idx} (E={self.energies[state_idx]:.6f} Ha)"
        fig = circ.draw(
            output='mpl',
            fold=FOLD,
            style={'dpi': 600, 'fontsize': FONTSIZE, 'subfontsize': FONTSIZE - 1},
        )
        fig.set_size_inches(*FIGSIZE)
        fig.suptitle(title, fontsize=FONTSIZE + 4, fontweight='bold')
        _register_fig(fig)
        return fig

    def run(self, num_states: int = 3, verbose: bool = True) -> List[float]:
        """VQD로 연속된 고유상태(num_states개)를 순차 탐색."""
        if verbose and num_states > 1:
            print(f"- Compute {num_states} states")
        self.find_ground_state()
        for _ in range(1, num_states):
            self.find_excited_state()
        return self.energies

# =============================
# 회로 도식 유틸
# =============================
def draw_quantum_circuits(vqd: VQD, tag: str = "") -> None:
    """
    현재까지 탐색된 모든 상태에 대해 회로 그림 생성만 수행.
    - 실제 화면 출력은 show_all_figures_blocking()에서 일괄 처리
    - DRAW_CIRCUITS=False면 아무 것도 하지 않음
    """
    if not DRAW_CIRCUITS:
        return
    print("\n[Circuits] prepare figures" + (f" {tag}" if tag else ""))
    for i in range(len(vqd.energies)):
        vqd.draw_circuit(i)

# =============================
# 메인 파이프라인
# =============================
def main():
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("PSPCz VQD×2: Stage 1 (UCCSD library, tapered-equivalent 2q) → Stage 2 (Frozen + real_amplitudes)")
    print(f"- ε-jitter: {INIT_EPS} | RA reps: {RA_REPS}, entanglement={RA_ENTANGLEMENT} | Optimizer: {OPTIMIZATION_METHOD}")
    print("=" * 70)

    # Hamiltonian 및 정확 해 스펙트럼(행렬 대각화)
    H_op, coeffs = build_h_pspcz()
    print("[PSPCz] Hamiltonian (6-31G(d), HOMO–LUMO 2-qubit)")
    print(format_hamiltonian_equation(coeffs))

    evals, _ = np.linalg.eigh(H_op.to_matrix())
    exact_total = np.array(sorted(np.real_if_close(evals)))
    print("\n[Exact Eigenvalues] (first 3, total energy)")
    for i, e in enumerate(exact_total[:3]):
        print(f"  - State {i}: {e:.8f} Ha")

    # ----- Stage 1: UCCSD(lib 등가 2q) VQD -----
    print("\n[Stage 1] UCCSD(lib, 2q equivalent) (VQD)")
    ansatz_s1 = AnsatzBuilder(mode='uccsd_lib_equiv_2q')
    vqd1 = VQD(
        h_op=H_op,
        num_qubits=2,
        beta=PENALTY_BETA,
        seed=GLOBAL_SEED,
        ansatz=ansatz_s1,
        init_mode='random',
    )
    e1 = vqd1.run(num_states=NUM_STATES, verbose=True)

    # Stage 1 결과 테이블
    print_table_stage("Stage 1 Results (Exact vs Calc)", exact_total[:NUM_STATES], e1)

    # Stage 1 회로 그림
    draw_quantum_circuits(vqd1, tag='[Stage 1]')
    show_all_figures_blocking()

    # ----- Stage 2: 상태별 real_amplitudes 미세 보정 -----
    print("\n[Stage 2] Per-state refine: Frozen Initial + real_amplitudes (ε-jitter)")
    frozen_stage1 = [
        vqd1.circuit.assign_parameters(vqd1.params_list[j])
        for j in range(NUM_STATES)
    ]
    e2: List[float] = []
    ref_pool: List[QuantumCircuit] = []

    for j in range(NUM_STATES):
        print(f"  * refine from State {j}")
        ansatz_s2 = AnsatzBuilder(
            mode='realamplitudes',
            frozen_initial=frozen_stage1[j],
            ra_reps=RA_REPS,
            ra_entanglement=RA_ENTANGLEMENT,
        )
        vqd2 = VQD(
            h_op=H_op,
            num_qubits=2,
            beta=PENALTY_BETA,
            seed=GLOBAL_SEED,
            ansatz=ansatz_s2,
            init_mode='zeros',                  # zeros 근처에서 시작(미세 보정)
            external_reference_states=ref_pool, # 이미 보정된 앞선 상태들과의 직교화 강화
        )
        _ = vqd2.run(num_states=1, verbose=False)
        e2.append(vqd2.energies[0])

        # Stage 2 각 상태의 회로 그림
        draw_quantum_circuits(vqd2, tag=f'[Stage 2, from State {j}]')

        refined_j = vqd2.circuit.assign_parameters(vqd2.params_list[0])
        ref_pool.append(refined_j)

    # Stage 2 결과 테이블
    print_table_stage("Stage 2 Results (Exact vs Calc)", exact_total[:NUM_STATES], e2)
    show_all_figures_blocking()

    # 최종 리포트(인덱스 0/1/2를 각각 S0/T1/S1로 해석)
    E0, T1, S1 = e2[0], e2[1], e2[2]
    dEST = S1 - T1
    print("\n[Final Report — PSPCz @ 6-31G(d), HOMO–LUMO 2q, VQD×2]")
    print(f" S0 = {E0:.9f} Ha")
    print(f" T1 = {T1:.9f} Ha")
    print(f" S1 = {S1:.9f} Ha")
    print(f" ΔE_ST = {dEST:.9f} Ha  ({dEST*27.211386245981:.6f} eV)")

    # 최종 비교 테이블(Stage1/Stage2 + ΔE_ST + 통계)
    print_table_combined("Final Comparison (Exact, Stage1, Stage2)", exact_total[:NUM_STATES], e1, e2)

    print("\nDone.")

if __name__ == "__main__":
    main()
