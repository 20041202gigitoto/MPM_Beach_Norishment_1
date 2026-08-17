"""
mpm_beach_wave.py (Numba版)
============================
mpm_beach.py (Numba版) をベースに、簡易振動流モデルによる波浪外力を
追加した版。mpm_scratch/mpm_beach_wave.py の Numba 高速化版であり、
物理モデル・パラメータ・出力は元のコードと同一で、1ステップ(substep)の
計算を numpy のバッチ演算ではなく Numba (@njit) の粒子ごとの明示的
forループに書き換えている点のみが異なる(詳細は mpm_beach.py 冒頭の
コメントを参照)。

このコードでできること
-----------------------
- 沖合の海底(平坦なローカル基盤)上に設置された、水没マウンド断面
  (沖側スロープ+天端+岸側スロープ、天端は静水面下)を MPM 粒子群として生成する。
- 重力(水中の浮力を考慮した水中単位体積重量)を外力として、マウンドが
  自重で沈下・法面が崩れて安定形状(水中安息角)に近づく様子を計算する。
- 静水面(SWL)以下かつ、その時点(変形後)のマウンド表面から一定の厚み
  (活動層, WAVE_ACTIVE_LAYER)以内にある格子ノードに、線形波理論(Airy波)
  による水平往復流速(orbital velocity)を「目標流速」として与える簡易振動流
  モデルにより、波浪による周期的なせん断外力を砂粒子に作用させる(外部波浪
  モデルとのカップリングなしの近似)。表面から深く埋もれた内部の粒子には
  直接作用させない(実際の波が海底面近傍にしか直接影響しないことに対応)。
  これにより、自重変形だけでは生じない、波の往復作用によるマウンドの
  緩やかな変形・移動・侵食を再現する。
- 静水面(SWL)を基準線として図示し、マウンドが常に水没した状態にあることを
  可視化する。
- 計算結果(粒子位置)をスナップショット保存し、初期断面と最終断面を
  比較するプロットを出力する。

このコードで「できないこと」(意図的に含めていない要素)
-----------------------------------------------------
- 実際の流体(水)の運動方程式は解いていない。波浪外力は「線形波理論に
  基づく目標流速に格子速度を緩和的に近づける」簡易抗力モデルであり、
  水そのものの質量・圧力・乱流・砕波・戻り流れ(undertow)は表現しない。
- 波の空間的な位相差(kx)は無視している。また対称な線形波のみを与えており、
  浅水変形に伴う波形の非対称性(スキューネス)は含まない。
- 侵食・堆積量の定量評価(体積収支等)は含まない。
- 複数材料、海底勾配は含まない。単一材料・局所的に平坦な海底のみ。

数値手法
--------
- MLS-MPM (Hu et al. 2018) を陽解法・2次 B-スプライン形状関数・APIC で実装。
- 砂の構成則: Hencky弾性 + Drucker-Prager 完全塑性(Klar et al. 2016)。
- substep 本体は Numba (@njit) でJITコンパイルされた粒子ごとの明示的
  forループとして実装している(numpyバッチ演算版からの変更点)。
- 依存ライブラリは numpy, matplotlib, numba。

実行方法
--------
    python mpm_beach_wave.py
    python mpm_beach_wave.py --t-total 12.0


VSCode でグラフをインライン表示する方法
----------------------------------------
- 実行結果の断面図・時系列図は常に PNG として output/ (または各ファイル
  固有の出力先) に保存されるため、VSCode のエクスプローラーでPNGファイルを
  クリックすれば画像プレビューとして閲覧できる。
- それに加えて、本ファイルは `# %%` でセル分割されているため、VSCode の
  Python拡張機能を使えば「Run Cell」または「Run Current File in Interactive
  Window」でJupyter形式のインタラクティブウィンドウとして実行できる。
  この方法で実行すると、`plt.show()` が別ウィンドウを開く代わりに、
  インタラクティブウィンドウ内にグラフがそのままインライン表示される。
"""


# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False



# %%
# ============================================================================
# 1. パラメータ設定 (ここを書き換えて条件を変える)
# ============================================================================

# ---- 背景格子 ----
DX = 0.02          # 格子間隔 [m]
LX = 2.6           # 計算領域の x方向長さ(岸沖方向)[m]
LY = 1.0           # 計算領域の y方向長さ(鉛直方向)[m]
BOUND = 3          # 境界条件を課す格子層の厚さ(ノード数)

# ---- 静水面(SWL: Still Water Level) ----
WATER_LEVEL = 0.55   # 静水面の y座標 [m]

# ---- 仮想の沖合養浜マウンドの初期断面形状 ----
X_TOE_NEARSHORE = 0.4   # 岸側(浅い側)法尻の x座標 [m]
X_CREST_START = 1.0     # 天端(マウンド上面)開始の x座標 [m]
X_CREST_END = 1.6       # 天端終了の x座標 [m]
X_TOE_OFFSHORE = 2.1    # 沖側(深い側)法尻の x座標 [m]
MOUND_HEIGHT = 0.35     # マウンド高さ(海底からの比高)[m]
PARTICLES_PER_CELL = 2  # 1格子1辺あたりの粒子分割数(実際は この2乗個/セル)

SUBMERGENCE = WATER_LEVEL - MOUND_HEIGHT

# ---- 砂の材料物性(水中砂: 沖合養浜マウンドは常時水没している) ----
SAND_DENSITY_SAT = 1950.0   # 飽和砂の密度(間隙が海水で満たされた状態)[kg/m3]
WATER_DENSITY = 1025.0      # 海水の密度 [kg/m3]
RHO0 = SAND_DENSITY_SAT - WATER_DENSITY
E_MOD = 1.0e6        # ヤング率 [Pa]
NU = 0.3             # ポアソン比 [-]
FRICTION_DEG = 32.0  # 内部摩擦角 [deg] (Drucker-Prager)
GRAVITY = 9.81       # 重力加速度 [m/s2]

# ---- 波浪外力(簡易振動流モデル) ----
WAVE_HEIGHT = 0.12       # 代表波高 H [m]
WAVE_PERIOD = 1.6        # 代表周期 T [s]
WAVE_DRAG_TIME = 0.05    # 格子速度を目標流速へ緩和させる時定数 tau [s]
WAVE_RAMP_CYCLES = 1.5   # 助走区間 [周期数]
WAVE_ACTIVE_LAYER = 0.06 # 波浪外力を作用させる表層の厚み(活動層厚)[m]

# ---- 数値計算設定 ----
CFL = 0.3            # クーラン数(安定限界に対する余裕係数)
DAMPING = 0.999      # 格子速度への簡易減衰係数(1.0で減衰なし、微小な数値粘性)
T_TOTAL = 6.0        # 総計算時間 [s] (WAVE_PERIOD=1.6sで約3.75周期分)
FRAME_DT = 0.02      # スナップショットを保存する時間間隔 [s]

# ---- 出力 ----
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_wave")



# %%
# ============================================================================
# 2. 初期粒子配置(沖合養浜マウンド断面の生成)
# ============================================================================

def mound_surface_height(x):
    """岸沖距離 x [m] における水没マウンド表面の高さ y [m] を返す(台形断面)。"""
    y = np.zeros_like(x)

    rise = (x >= X_TOE_NEARSHORE) & (x < X_CREST_START)
    y[rise] = MOUND_HEIGHT * (x[rise] - X_TOE_NEARSHORE) / (X_CREST_START - X_TOE_NEARSHORE)

    flat = (x >= X_CREST_START) & (x <= X_CREST_END)
    y[flat] = MOUND_HEIGHT

    fall = (x > X_CREST_END) & (x <= X_TOE_OFFSHORE)
    y[fall] = MOUND_HEIGHT * (X_TOE_OFFSHORE - x[fall]) / (X_TOE_OFFSHORE - X_CREST_END)

    return y


def make_mound_particles():
    """台形マウンド断面の内部を埋めるように粒子を格子状(ppc x ppc/セル)に配置する。"""
    cell_x = np.arange(0.0, LX, DX)
    cell_y = np.arange(0.0, LY, DX)
    sub = (np.arange(PARTICLES_PER_CELL) + 0.5) / PARTICLES_PER_CELL * DX

    px = (cell_x[:, None] + sub[None, :]).ravel()
    py = (cell_y[:, None] + sub[None, :]).ravel()
    PX, PY = np.meshgrid(px, py, indexing="ij")
    PX = PX.ravel()
    PY = PY.ravel()

    surf = mound_surface_height(PX)
    inside = (PY > 1e-9) & (PY < surf)
    return PX[inside].copy(), PY[inside].copy()



# %%
# ============================================================================
# 3. MPM ソルバー本体 (MLS-MPM + Drucker-Prager 砂塑性 + 波浪外力)
# ============================================================================

def solve_wave_number(omega, depth, g=GRAVITY, tol=1e-10, max_iter=100):
    """線形波の分散関係 omega^2 = g k tanh(k depth) を Newton法で解き、波数 k を返す。"""
    k = omega ** 2 / g  # 深海波近似を初期値とする
    for _ in range(max_iter):
        th = np.tanh(k * depth)
        f = g * k * th - omega ** 2
        fp = g * th + g * k * depth * (1.0 - th ** 2)
        dk = f / fp
        k -= dk
        if abs(dk) < tol:
            break
    return k


class MPMState:
    """粒子・格子の状態と物性値をまとめて保持するだけの単純な入れ物。"""

    def __init__(self):
        px, py = make_mound_particles()
        n = len(px)

        self.n_particles = n
        self.x = np.stack([px, py], axis=1)          # 粒子位置 (N,2)
        self.v = np.zeros((n, 2))                     # 粒子速度 (N,2)
        self.C = np.zeros((n, 2, 2))                  # APIC アフィン速度行列 (N,2,2)
        self.F = np.tile(np.eye(2), (n, 1, 1))         # 弾性変形勾配 (N,2,2)

        p_vol = (DX / PARTICLES_PER_CELL) ** 2
        self.vol0 = np.full(n, p_vol)
        self.mass = self.vol0 * RHO0

        self.x0 = self.x.copy()  # 初期位置(比較プロット用に保持)

        # --- Lame定数 ---
        self.mu0 = E_MOD / (2.0 * (1.0 + NU))
        self.lambda0 = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

        # --- Drucker-Prager の摩擦パラメータ ---
        phi = np.radians(FRICTION_DEG)
        self.alpha_dp = np.sqrt(2.0 / 3.0) * (2.0 * np.sin(phi)) / (3.0 - np.sin(phi))

        # --- 背景格子 ---
        self.nx = int(round(LX / DX)) + 1
        self.ny = int(round(LY / DX)) + 1
        self.n_nodes = self.nx * self.ny
        self.inv_dx = 1.0 / DX
        self.d_inv = 4.0 * self.inv_dx * self.inv_dx  # 2次Bスプラインの逆慣性係数

        i_idx = np.repeat(np.arange(self.nx), self.ny)
        j_idx = np.tile(np.arange(self.ny), self.nx)
        self.bottom_mask = j_idx < BOUND
        self.side_mask = (i_idx < BOUND) | (i_idx > self.nx - 1 - BOUND)

        # --- 波浪外力(簡易振動流モデル)用の準備 ---
        self.node_y = j_idx.astype(np.float64) * DX
        self.wave_mask = (self.node_y < WATER_LEVEL) & ~self.bottom_mask
        self.wave_omega = 2.0 * np.pi / WAVE_PERIOD
        self.wave_k = solve_wave_number(self.wave_omega, WATER_LEVEL)
        self.wave_sinh_kd = np.sinh(self.wave_k * WATER_LEVEL)
        self.wave_active_layer_cells = max(1, int(round(WAVE_ACTIVE_LAYER / DX)))

        # --- 時間刻み(CFL条件) ---
        wave_speed = np.sqrt((self.lambda0 + 2.0 * self.mu0) / RHO0)
        dt_cfl = CFL * DX / wave_speed
        self.n_substeps = max(1, int(np.ceil(FRAME_DT / dt_cfl)))
        self.dt = FRAME_DT / self.n_substeps

        wave_length = 2.0 * np.pi / self.wave_k

        print(f"[MPM] 粒子数 = {n}")
        print(f"[MPM] 格子ノード数 = {self.nx} x {self.ny} = {self.n_nodes}")
        print(f"[MPM] dt = {self.dt:.3e} s, {self.n_substeps} substeps/frame")
        print(f"[MPM] マウンド天端の被覆水深(水没深さ) = {SUBMERGENCE:.3f} m "
              f"(静水面 y={WATER_LEVEL:.2f} m, 天端 y={MOUND_HEIGHT:.2f} m)")
        if SUBMERGENCE <= 0.0:
            print("[MPM] 警告: マウンド天端が静水面より高く、水没条件を満たしていません。")
        print(f"[MPM] 波浪外力: H={WAVE_HEIGHT:.2f} m, T={WAVE_PERIOD:.2f} s, "
              f"波長 L={wave_length:.2f} m (水深 d={WATER_LEVEL:.2f} m), "
              f"抗力時定数 tau={WAVE_DRAG_TIME:.3f} s, "
              f"活動層厚={WAVE_ACTIVE_LAYER:.3f} m ({self.wave_active_layer_cells} セル)")


# ----------------------------------------------------------------------------
# Numba高速化カーネル: 粒子ごとの明示的forループでMLS-MPMの1ステップ全体
# (F更新+Drucker-Prager塑性リターンマッピング -> P2G -> 格子更新+波浪外力
# -> G2P) を実行する。詳細は mpm_beach.py の同名カーネルのコメントを参照。
# 波浪外力ブロックのみ、その時点(変形後)の表面から活動層厚以内の格子
# ノードにのみ、線形波理論の水平往復流速へ向けた緩和項を加える点が
# mpm_beach.py のカーネルとの違い。
# ----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _substep_kernel(x, v, C, F, mass, vol0, mu0, lambda0, alpha_dp,
                     dt, inv_dx, dx, d_inv, nx, ny, gravity, damping,
                     bottom_mask, side_mask, lx, ly,
                     wave_mask, wave_k, wave_sinh_kd, wave_omega, t,
                     wave_height, wave_period, ramp, drag_coef,
                     wave_active_layer_cells):
    n = x.shape[0]
    n_nodes = nx * ny

    grid_m = np.zeros(n_nodes)
    grid_vx = np.zeros(n_nodes)
    grid_vy = np.zeros(n_nodes)

    F_elastic = np.empty((n, 2, 2))

    base_i = np.empty(n, dtype=np.int64)
    base_j = np.empty(n, dtype=np.int64)
    fx_arr = np.empty(n)
    fy_arr = np.empty(n)

    # ------------------------------------------------------------------
    # (a) F更新 + Drucker-Prager リターンマッピング, (b) P2G
    # ------------------------------------------------------------------
    for p in range(n):
        c00 = C[p, 0, 0]
        c01 = C[p, 0, 1]
        c10 = C[p, 1, 0]
        c11 = C[p, 1, 1]

        m00 = 1.0 + dt * c00
        m01 = dt * c01
        m10 = dt * c10
        m11 = 1.0 + dt * c11

        f00 = F[p, 0, 0]
        f01 = F[p, 0, 1]
        f10 = F[p, 1, 0]
        f11 = F[p, 1, 1]

        a = m00 * f00 + m01 * f10
        b = m00 * f01 + m01 * f11
        c = m10 * f00 + m11 * f10
        d = m10 * f01 + m11 * f11

        mm11 = a * a + c * c
        mm12 = a * b + c * d
        mm22 = b * b + d * d

        phi = 0.5 * np.arctan2(2.0 * mm12, mm11 - mm22)
        cp = np.cos(phi)
        sp = np.sin(phi)

        mean = 0.5 * (mm11 + mm22)
        disc = np.sqrt(max(((mm11 - mm22) * 0.5) ** 2 + mm12 ** 2, 0.0))
        sigma1 = np.sqrt(max(mean + disc, 0.0))
        sigma2 = np.sqrt(max(mean - disc, 0.0))

        v1x, v1y = cp, sp
        v2x, v2y = -sp, cp

        u1x = a * v1x + b * v1y
        u1y = c * v1x + d * v1y
        u2x = a * v2x + b * v2y
        u2y = c * v2x + d * v2y

        eps_tol = 1e-12
        n1 = np.hypot(u1x, u1y)
        if n1 > eps_tol:
            u1x /= n1
            u1y /= n1
        else:
            u1x = 1.0
            u1y = 0.0
        n2 = np.hypot(u2x, u2y)
        if n2 > eps_tol:
            u2x /= n2
            u2y /= n2
        else:
            u2x = -u1y
            u2y = u1x

        sigma1 = max(sigma1, 1e-6)
        sigma2 = max(sigma2, 1e-6)

        eps1 = np.log(sigma1)
        eps2 = np.log(sigma2)
        tr_eps = eps1 + eps2
        eps_hat0 = eps1 - 0.5 * tr_eps
        eps_hat1 = eps2 - 0.5 * tr_eps
        eps_hat_norm = np.sqrt(eps_hat0 * eps_hat0 + eps_hat1 * eps_hat1)

        delta_gamma = eps_hat_norm + (2.0 * lambda0 + 2.0 * mu0) / (2.0 * mu0) * tr_eps * alpha_dp

        new_eps0 = eps1
        new_eps1 = eps2
        if eps_hat_norm > 1e-12 and delta_gamma > 0.0 and tr_eps <= 0.0:
            scale = delta_gamma / eps_hat_norm
            new_eps0 = eps1 - scale * eps_hat0
            new_eps1 = eps2 - scale * eps_hat1
        if tr_eps > 0.0:
            new_eps0 = 0.0
            new_eps1 = 0.0

        new_sigma0 = np.exp(new_eps0)
        new_sigma1 = np.exp(new_eps1)

        ud00 = u1x * new_sigma0
        ud10 = u1y * new_sigma0
        ud01 = u2x * new_sigma1
        ud11 = u2y * new_sigma1

        F_elastic[p, 0, 0] = ud00 * v1x + ud01 * v2x
        F_elastic[p, 0, 1] = ud00 * v1y + ud01 * v2y
        F_elastic[p, 1, 0] = ud10 * v1x + ud11 * v2x
        F_elastic[p, 1, 1] = ud10 * v1y + ud11 * v2y

        tr_new = new_eps0 + new_eps1
        ps0 = 2.0 * mu0 * new_eps0 + lambda0 * tr_new
        ps1 = 2.0 * mu0 * new_eps1 + lambda0 * tr_new

        ud2_00 = u1x * ps0
        ud2_10 = u1y * ps0
        ud2_01 = u2x * ps1
        ud2_11 = u2y * ps1

        tau00 = ud2_00 * u1x + ud2_01 * u2x
        tau01 = ud2_00 * u1y + ud2_01 * u2y
        tau10 = ud2_10 * u1x + ud2_11 * u2x
        tau11 = ud2_10 * u1y + ud2_11 * u2y

        stress_coef = -dt * d_inv * vol0[p]
        aff00 = stress_coef * tau00 + mass[p] * c00
        aff01 = stress_coef * tau01 + mass[p] * c01
        aff10 = stress_coef * tau10 + mass[p] * c10
        aff11 = stress_coef * tau11 + mass[p] * c11

        px_cell = x[p, 0] * inv_dx - 0.5
        py_cell = x[p, 1] * inv_dx - 0.5
        bi = int(np.floor(px_cell))
        bj = int(np.floor(py_cell))
        base_i[p] = bi
        base_j[p] = bj
        fxi = x[p, 0] * inv_dx - bi
        fyj = x[p, 1] * inv_dx - bj
        fx_arr[p] = fxi
        fy_arr[p] = fyj

        wx0 = 0.5 * (1.5 - fxi) ** 2
        wx1 = 0.75 - (fxi - 1.0) ** 2
        wx2 = 0.5 * (fxi - 0.5) ** 2
        wy0 = 0.5 * (1.5 - fyj) ** 2
        wy1 = 0.75 - (fyj - 1.0) ** 2
        wy2 = 0.5 * (fyj - 0.5) ** 2

        wxa = (wx0, wx1, wx2)
        wya = (wy0, wy1, wy2)

        vxp = v[p, 0]
        vyp = v[p, 1]
        mp = mass[p]

        for oi in range(3):
            wxo = wxa[oi]
            dxo = (oi - fxi) * dx
            for oj in range(3):
                wyo = wya[oj]
                dyo = (oj - fyj) * dx
                weight = wxo * wyo
                node_idx = (bi + oi) * ny + (bj + oj)

                affdx = aff00 * dxo + aff01 * dyo
                affdy = aff10 * dxo + aff11 * dyo

                grid_m[node_idx] += weight * mp
                grid_vx[node_idx] += weight * (mp * vxp + affdx)
                grid_vy[node_idx] += weight * (mp * vyp + affdy)

    # ------------------------------------------------------------------
    # (c) 格子上の更新: 質量で正規化し、重力を与え、減衰させる
    # ------------------------------------------------------------------
    has_mass = np.zeros(n_nodes, dtype=np.bool_)
    for idx in range(n_nodes):
        if grid_m[idx] > 1e-12:
            has_mass[idx] = True
            grid_vx[idx] /= grid_m[idx]
            grid_vy[idx] /= grid_m[idx]
            grid_vy[idx] -= dt * gravity
        grid_vx[idx] *= damping
        grid_vy[idx] *= damping

    # ------------------------------------------------------------------
    # (c') 波浪外力(簡易振動流モデル、表面から活動層厚以内のみに作用)
    # ------------------------------------------------------------------
    if ramp > 0.0:
        j_surface = np.full(nx, -1, dtype=np.int64)
        for i in range(nx):
            base_idx = i * ny
            top = -1
            for j in range(ny):
                if grid_m[base_idx + j] > 1e-12:
                    top = j
            j_surface[i] = top

        cos_wt = np.cos(wave_omega * t)
        amp = ramp * np.pi * wave_height / wave_period

        for i in range(nx):
            js = j_surface[i]
            base_idx = i * ny
            for j in range(ny):
                idx = base_idx + j
                if not wave_mask[idx] or not has_mass[idx]:
                    continue
                depth_cells = js - j
                if depth_cells < 0 or depth_cells >= wave_active_layer_cells:
                    continue
                node_y = j * dx
                u_target = amp * np.cosh(wave_k * node_y) / wave_sinh_kd * cos_wt
                grid_vx[idx] += drag_coef * (u_target - grid_vx[idx])

    # ------------------------------------------------------------------
    # (c'') 境界条件
    # ------------------------------------------------------------------
    for idx in range(n_nodes):
        if bottom_mask[idx]:
            grid_vx[idx] = 0.0
            grid_vy[idx] = 0.0
        if side_mask[idx]:
            grid_vx[idx] = 0.0

    # ------------------------------------------------------------------
    # (d) Grid-to-Particle (G2P)
    # ------------------------------------------------------------------
    new_x = np.empty((n, 2))
    new_v = np.empty((n, 2))
    new_C = np.empty((n, 2, 2))

    margin = dx
    for p in range(n):
        bi = base_i[p]
        bj = base_j[p]
        fxi = fx_arr[p]
        fyj = fy_arr[p]

        wx0 = 0.5 * (1.5 - fxi) ** 2
        wx1 = 0.75 - (fxi - 1.0) ** 2
        wx2 = 0.5 * (fxi - 0.5) ** 2
        wy0 = 0.5 * (1.5 - fyj) ** 2
        wy1 = 0.75 - (fyj - 1.0) ** 2
        wy2 = 0.5 * (fyj - 0.5) ** 2

        wxa = (wx0, wx1, wx2)
        wya = (wy0, wy1, wy2)

        nvx = 0.0
        nvy = 0.0
        nc00 = 0.0
        nc01 = 0.0
        nc10 = 0.0
        nc11 = 0.0

        for oi in range(3):
            wxo = wxa[oi]
            dxo = (oi - fxi) * dx
            for oj in range(3):
                wyo = wya[oj]
                dyo = (oj - fyj) * dx
                weight = wxo * wyo
                node_idx = (bi + oi) * ny + (bj + oj)

                gvx = grid_vx[node_idx]
                gvy = grid_vy[node_idx]

                nvx += weight * gvx
                nvy += weight * gvy

                wC = d_inv * weight
                nc00 += wC * gvx * dxo
                nc01 += wC * gvx * dyo
                nc10 += wC * gvy * dxo
                nc11 += wC * gvy * dyo

        xn0 = x[p, 0] + dt * nvx
        xn1 = x[p, 1] + dt * nvy

        if xn0 < margin:
            xn0 = margin
        elif xn0 > lx - margin:
            xn0 = lx - margin
        if xn1 < margin:
            xn1 = margin
        elif xn1 > ly - margin:
            xn1 = ly - margin

        new_x[p, 0] = xn0
        new_x[p, 1] = xn1
        new_v[p, 0] = nvx
        new_v[p, 1] = nvy
        new_C[p, 0, 0] = nc00
        new_C[p, 0, 1] = nc01
        new_C[p, 1, 0] = nc10
        new_C[p, 1, 1] = nc11

    return new_x, new_v, new_C, F_elastic


def substep(s: MPMState, t: float):
    """MLS-MPM の1ステップ (P2G -> 格子更新+波浪外力 -> G2P) を実行し、状態を更新する。

    t: 現在のシミュレーション時刻 [s] (波浪外力の位相計算に用いる)。
    実体は Numba でJITコンパイルされた `_substep_kernel` への薄いラッパー。
    """
    ramp = min(t / (WAVE_RAMP_CYCLES * WAVE_PERIOD), 1.0)
    drag_coef = min(s.dt / WAVE_DRAG_TIME, 1.0)

    new_x, new_v, new_C, new_F = _substep_kernel(
        s.x, s.v, s.C, s.F, s.mass, s.vol0,
        s.mu0, s.lambda0, s.alpha_dp,
        s.dt, s.inv_dx, DX, s.d_inv, s.nx, s.ny, GRAVITY, DAMPING,
        s.bottom_mask, s.side_mask, LX, LY,
        s.wave_mask, s.wave_k, s.wave_sinh_kd, s.wave_omega, t,
        WAVE_HEIGHT, WAVE_PERIOD, ramp, drag_coef,
        s.wave_active_layer_cells,
    )
    s.x = new_x
    s.v = new_v
    s.C = new_C
    s.F = new_F



# %%
# ============================================================================
# 4. 後処理(断面形状の可視化)
# ============================================================================

def top_envelope(x, y, x_max=LX, bin_width=None):
    """粒子群から x方向にビニングし、各binの最高高さ(=マウンド表面)を抽出する。"""
    if bin_width is None:
        bin_width = 4 * DX
    bins = np.arange(0.0, x_max + bin_width, bin_width)
    idx = np.digitize(x, bins) - 1
    n_bins = len(bins) - 1

    top_y = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if np.any(mask):
            top_y[b] = y[mask].max()

    centers = 0.5 * (bins[:-1] + bins[1:])
    valid = ~np.isnan(top_y)
    return centers[valid], top_y[valid]


def plot_results(snapshots, s: MPMState, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- (a) 粒子位置スナップショットの重ね描き ---
    fig, ax = plt.subplots(figsize=(9, 4))
    n_snap = len(snapshots)
    for k, (t, x) in enumerate(snapshots):
        color = plt.cm.viridis(k / max(n_snap - 1, 1))
        alpha = 0.25 if 0 < k < n_snap - 1 else 0.9
        size = 2 if 0 < k < n_snap - 1 else 4
        ax.scatter(x[:, 0], x[:, 1], s=size, color=color, alpha=alpha,
                   label=f"t={t:.2f}s" if k in (0, n_snap - 1) else None)
    ax.axhline(WATER_LEVEL, color="tab:blue", linestyle=":", linewidth=1.5,
               label=f"静水面 SWL (y={WATER_LEVEL:.2f}m)")
    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_title("沖合養浜マウンドの自重変形+波浪外力による変形 (MPM, Numba)")
    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "particles_overlay.png"), dpi=150)

    # --- (b) 初期断面 vs 最終断面(表面プロファイル比較) ---
    t0, x0 = snapshots[0]
    t1, x1 = snapshots[-1]
    xe0, ye0 = top_envelope(x0[:, 0], x0[:, 1])
    xe1, ye1 = top_envelope(x1[:, 0], x1[:, 1])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xe0, ye0, "--", color="gray", label=f"初期断面 (t={t0:.2f}s)")
    ax.plot(xe1, ye1, "-", color="tab:orange", label=f"最終断面 (t={t1:.2f}s)")
    ax.axhline(WATER_LEVEL, color="tab:blue", linestyle=":", linewidth=1.5,
               label=f"静水面 SWL (y={WATER_LEVEL:.2f}m)")
    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_title("沖合養浜マウンド断面形状の変化(自重沈下・法面変形+波浪外力)")
    ax.set_xlim(0, LX)
    ax.set_ylim(0, max(WATER_LEVEL, MOUND_HEIGHT) * 1.3)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_comparison.png"), dpi=150)
    plt.show()

    print(f"[MPM] 図を保存しました: {out_dir}")



# %%
# ============================================================================
# 5. メイン実行
# ============================================================================

def run():
    state = MPMState()

    n_frames = int(round(T_TOTAL / FRAME_DT))
    snapshots = [(0.0, state.x.copy())]

    t = 0.0
    for frame in range(n_frames):
        for _ in range(state.n_substeps):
            substep(state, t)
            t += state.dt
        snapshots.append((t, state.x.copy()))

        if frame % max(1, n_frames // 10) == 0 or frame == n_frames - 1:
            print(f"[MPM] frame {frame + 1}/{n_frames}  t={t:.3f}s  "
                  f"max|v|={np.linalg.norm(state.v, axis=1).max():.3f} m/s")

        np.savez(
            os.path.join(OUTPUT_DIR, f"snapshot_{frame:04d}.npz"),
            t=t, x=state.x, v=state.v,
        )

    plot_results(snapshots, state, OUTPUT_DIR)
    return state, snapshots



def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="MPM沖合養浜マウンド自重変形解析(Numba版)。--t-total で総計算時間を延長できる。"
    )
    parser.add_argument(
        "--t-total", type=float, default=None,
        help=f"総計算時間 [s] (省略時はコード内の既定値 T_TOTAL={T_TOTAL}s を使用)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="出力先ディレクトリ(省略時は output_wave/。長時間実行では過去の出力を"
             "上書きしないよう別ディレクトリを指定することを推奨)",
    )
    # VSCode の Interactive Window / Jupyter 経由で実行すると sys.argv に
    # カーネル起動用の引数(-f <connection_file> 等)が混入するため、
    # 未知の引数は無視する parse_known_args() を使う。
    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.t_total is not None:
        T_TOTAL = args.t_total
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    state, snapshots = run()


# ============================================================================
# 拡張のヒント (研究を進める際の追加要素の例)
# ============================================================================
# mpm_scratch/mpm_beach_wave.py 末尾の「拡張のヒント」を参照(物理モデルの
# 拡張ポイントはNumba化による変更の影響を受けない)。
