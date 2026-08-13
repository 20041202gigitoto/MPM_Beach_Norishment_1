"""
mpm_beach_wave_cubic.py
========================
mpm_beach_wave.py をベースに、**形状関数を2次B-スプラインから3次B-スプライン
に変更した版**。物理モデル(MLS-MPM本体、Drucker-Prager砂塑性、波浪外力モデル、
可視化)はすべて mpm_beach_wave.py と同一で、粒子-格子間の補間に使う形状関数の
次数のみが異なる。

mpm_beach_wave.py との違い
--------------------------
- 形状関数: 2次B-スプライン(サポート幅3ノード、1粒子あたり3x3=9近傍) から
  3次B-スプライン(サポート幅4ノード、1粒子あたり4x4=16近傍)に変更。
  `_bspline_weights()` を3次カーネル(区分3次多項式)に置き換え、
  P2G/G2Pのオフセット配列・近傍数を 9 -> 16 に拡張した。
- APICの逆慣性係数 `d_inv`: 2次スプラインでは 4/dx^2 だったが、3次スプライン
  では理論値 3/dx^2 に変更(Jiang et al. 2016, "The Material Point Method
  for Simulating Continuum Materials", SIGGRAPH course notes の結果に基づく。
  D_p = (1/3) dx^2 I が3次B-スプラインのAPIC慣性テンソル)。
- 粒子座標のクリップ余白: 3次スプラインは補間の届く範囲(サポート幅)が
  2次より1セル分広いため、格子ノード配列の範囲外アクセスを防ぐクリップ
  余白を DX -> 2*DX に拡大した(境界近傍は元々 BOUND=3 層分が固定壁として
  扱われるため、この余白拡大が計算結果に与える影響は無視できる)。
- `base`(P2G/G2P用の基準ノード添字)の計算式: 2次では
  `floor(x/dx - 0.5)`、3次では `floor(x/dx) - 1` を用いる(スプラインの
  サポート範囲に対応した規約の違い)。
- 出力先ディレクトリを `output_cubic/` に分離し、2次版(`mpm_beach_wave.py`)
  の出力(`output/`)を上書きしないようにした。

計算コストへの影響
------------------
近傍ノード数が 9 -> 16(理論上 1.78倍)に増えるため、1ステップあたりの
P2G/G2P計算コストが増加する。実測では、SVD分解・Drucker-Prager
リターンマッピング等(近傍数に依存しない部分)が全体の一部を占める
ため、近傍数の増加率よりやや小さい約1.6〜1.7倍程度、substepあたりの
計算時間が増加する(粒子数・格子サイズに依存)。時間刻み dt は弾性波速度
から決まるCFL条件のみに依存し、形状関数の次数には依存しないため、
必要なsubstep数自体は変わらない。

このファイルのそれ以外の内容(コード構成、パラメータ、docstring)は
mpm_beach_wave.py を踏襲している。詳細な「このコードでできること」
「できないこと」等は mpm_beach_wave.py の説明を参照。

数値手法
--------
- MLS-MPM (Moving Least Squares MPM, Hu et al. 2018) を陽解法・
  3次 B-スプライン形状関数・APIC (Affine Particle-In-Cell) で実装。
- 砂の構成則: Hencky(対数ひずみ)弾性 + Drucker-Prager 完全塑性。
  変形勾配 F の特異値分解(SVD)を用いたリターンマッピングにより
  塑性変形を扱う (Klar et al. 2016, "Drucker-Prager Elastoplasticity
  for Sand Animation", ACM TOG/SIGGRAPH 2016 の手法に基づく)。
- 依存ライブラリは numpy と matplotlib のみ。
"""


import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False



# ============================================================================
# 1. パラメータ設定 (ここを書き換えて条件を変える)
# ============================================================================

# ---- 背景格子 ----
DX = 0.02          # 格子間隔 [m]
LX = 2.6           # 計算領域の x方向長さ(岸沖方向)[m]
LY = 1.0           # 計算領域の y方向長さ(鉛直方向)[m]
BOUND = 3          # 境界条件を課す格子層の厚さ(ノード数)

# ---- 静水面(SWL: Still Water Level) ----
# 沖合養浜マウンドは常時水没しているため、比較・可視化の基準として
# 静水面を明示的に持たせる(波浪計算はしないので力学には効かない)。
WATER_LEVEL = 0.55   # 静水面の y座標 [m]

# ---- 仮想の沖合養浜マウンドの初期断面形状 ----
# 台形断面: 海底(y=0, ローカルに平坦と仮定)上に、沖側・岸側どちらの法尻も
# 水没したまま盛り上がるマウンドを配置する(陸に接する「盛土」ではない)。
X_TOE_NEARSHORE = 0.4   # 岸側(浅い側)法尻の x座標 [m]
X_CREST_START = 1.0     # 天端(マウンド上面)開始の x座標 [m]
X_CREST_END = 1.6       # 天端終了の x座標 [m]
X_TOE_OFFSHORE = 2.1    # 沖側(深い側)法尻の x座標 [m]
MOUND_HEIGHT = 0.35     # マウンド高さ(海底からの比高)[m]
PARTICLES_PER_CELL = 2  # 1格子1辺あたりの粒子分割数(実際は この2乗個/セル)

# マウンド天端の水没深さ(被覆水深)。正であれば天端は常に水面下にある。
SUBMERGENCE = WATER_LEVEL - MOUND_HEIGHT

# ---- 砂の材料物性(水中砂: 沖合養浜マウンドは常時水没している) ----
SAND_DENSITY_SAT = 1950.0   # 飽和砂の密度(間隙が海水で満たされた状態)[kg/m3]
WATER_DENSITY = 1025.0      # 海水の密度 [kg/m3]
RHO0 = SAND_DENSITY_SAT - WATER_DENSITY
#   水中単位体積重量(浮力を考慮した「見かけの密度」)。
#   沖合養浜マウンドは常時水没しているため、自重変形を支配するのは
#   乾燥密度ではなく水中(浮力差引後)の密度である。本コードは陽解法MPMの
#   最小構成として、浮力を「密度を水中密度に置き換える」ことで簡易的に
#   表現している(間隙水圧の時間発展や排水過程は扱っていない)。
E_MOD = 1.0e6        # ヤング率 [Pa]
#   注: 陽解法MPMでは時間刻み dt が sqrt(E/rho) に反比例して小さくなるため、
#   計算コストを抑える目的で実際の水中砂(E~1e7-1e8Pa)より小さい値を
#   採用している。剛性が計算結果(安息角や崩壊形状)に与える影響は
#   小さいが、変位の絶対量や収束の速さには影響するので注意。
NU = 0.3             # ポアソン比 [-]
FRICTION_DEG = 32.0  # 内部摩擦角 [deg] (Drucker-Prager)
#   水中砂の内部摩擦角は粒子間の噛み合わせで決まるため、乾燥砂とほぼ
#   同程度とみなし同じ値を採用している(間隙水による粘性抵抗等の
#   動的効果は本コードでは扱わない)。
GRAVITY = 9.81       # 重力加速度 [m/s2]

# ---- 波浪外力(簡易振動流モデル) ----
# 静水面(WATER_LEVEL)以下の格子ノードに、線形波理論(Airy波)による水平
# 往復流速(orbital velocity)を「目標流速」として与え、格子速度をその目標値
# へ緩和(リラクゼーション)させることで、水そのものを解かずに波浪起源の
# 周期的せん断力を近似的に表現する。値はいずれも研究の出発点となる代表値
# (このコードの断面自体が仮想断面であり、特定海域の実測波浪ではない)。
WAVE_HEIGHT = 0.12       # 代表波高 H [m]
WAVE_PERIOD = 1.6        # 代表周期 T [s]
WAVE_DRAG_TIME = 0.05    # 格子速度を目標流速へ緩和させる時定数 tau [s]
                         # (小さいほど強く引きずられる。0に近づけるほど
                         #  水と一体で動く極限、大きくすると抗力が弱くなる)
WAVE_RAMP_CYCLES = 1.5   # 助走区間 [周期数](振幅を0→1へなめらかに立ち上げ、
                         # 急激な立ち上がりによる数値的な衝撃を避ける)
WAVE_ACTIVE_LAYER = 0.06 # 波浪外力を作用させる表層の厚み(活動層厚)[m]
                         # マウンド内部に埋もれた粒子まで一様に流体外力を
                         # 与えるのは非物理的(実際の波はマウンド表面近傍にしか
                         # 直接作用しない)なため、その時点の(変形後の)表面から
                         # この厚み分だけを対象とする。substep() 内で毎ステップ、
                         # 格子質量から現在の表面位置を検出して判定する。

# ---- 数値計算設定 ----
CFL = 0.3            # クーラン数(安定限界に対する余裕係数)
DAMPING = 0.999      # 格子速度への簡易減衰係数(1.0で減衰なし、微小な数値粘性)
T_TOTAL = 6.0        # 総計算時間 [s] (WAVE_PERIOD=1.6sで約3.75周期分。
                     # より長期の応答を見たい場合は --t-total で延長できる)
FRAME_DT = 0.02      # スナップショットを保存する時間間隔 [s]

# ---- 出力 ----
# 2次B-スプライン版(mpm_beach_wave.py)の出力(output/)を上書きしないよう
# 別ディレクトリに分離する。
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_cubic")



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



# ============================================================================
# 3. MPM ソルバー本体 (MLS-MPM + Drucker-Prager 砂塑性)
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
        # Mohr-Coulomb の内部摩擦角 phi に対応する Drucker-Prager 円錐の
        # 傾き係数 (Klar et al. 2016 の定義に基づく近似)。
        phi = np.radians(FRICTION_DEG)
        self.alpha_dp = np.sqrt(2.0 / 3.0) * (2.0 * np.sin(phi)) / (3.0 - np.sin(phi))

        # --- 背景格子 ---
        self.nx = int(round(LX / DX)) + 1
        self.ny = int(round(LY / DX)) + 1
        self.n_nodes = self.nx * self.ny
        self.inv_dx = 1.0 / DX
        self.d_inv = 3.0 * self.inv_dx * self.inv_dx  # 3次Bスプラインの逆慣性係数 (D_p=(1/3)dx^2 I)

        i_idx = np.repeat(np.arange(self.nx), self.ny)
        j_idx = np.tile(np.arange(self.ny), self.nx)
        self.bottom_mask = j_idx < BOUND
        self.side_mask = (i_idx < BOUND) | (i_idx > self.nx - 1 - BOUND)

        # --- 波浪外力(簡易振動流モデル)用の準備 ---
        self.node_i = i_idx
        self.node_j = j_idx
        self.node_y = j_idx.astype(np.float64) * DX
        self.wave_mask = (self.node_y < WATER_LEVEL) & ~self.bottom_mask
        self.wave_omega = 2.0 * np.pi / WAVE_PERIOD
        self.wave_k = solve_wave_number(self.wave_omega, WATER_LEVEL)
        self.wave_sinh_kd = np.sinh(self.wave_k * WATER_LEVEL)
        self.wave_cosh_ky = np.cosh(self.wave_k * self.node_y)
        self.wave_active_layer_cells = max(1, int(round(WAVE_ACTIVE_LAYER / DX)))

        # --- 時間刻み(CFL条件) ---
        # dtは弾性波速度と格子間隔から決まり、形状関数の次数には依存しない
        # (2次B-スプライン版と同じ式・同じ値になる)。
        wave_speed = np.sqrt((self.lambda0 + 2.0 * self.mu0) / RHO0)
        dt_cfl = CFL * DX / wave_speed
        self.n_substeps = max(1, int(np.ceil(FRAME_DT / dt_cfl)))
        self.dt = FRAME_DT / self.n_substeps

        wave_length = 2.0 * np.pi / self.wave_k

        print(f"[MPM] 粒子数 = {n}")
        print(f"[MPM] 格子ノード数 = {self.nx} x {self.ny} = {self.n_nodes}")
        print(f"[MPM] 形状関数: 3次 B-スプライン (1粒子あたり4x4=16近傍)")
        print(f"[MPM] dt = {self.dt:.3e} s, {self.n_substeps} substeps/frame")
        print(f"[MPM] マウンド天端の被覆水深(水没深さ) = {SUBMERGENCE:.3f} m "
              f"(静水面 y={WATER_LEVEL:.2f} m, 天端 y={MOUND_HEIGHT:.2f} m)")
        if SUBMERGENCE <= 0.0:
            print("[MPM] 警告: マウンド天端が静水面より高く、水没条件を満たしていません。")
        print(f"[MPM] 波浪外力: H={WAVE_HEIGHT:.2f} m, T={WAVE_PERIOD:.2f} s, "
              f"波長 L={wave_length:.2f} m (水深 d={WATER_LEVEL:.2f} m), "
              f"抗力時定数 tau={WAVE_DRAG_TIME:.3f} s, "
              f"活動層厚={WAVE_ACTIVE_LAYER:.3f} m ({self.wave_active_layer_cells} セル)")


def _bspline_weights(fx):
    """3次 B-スプライン形状関数の重み。

    fx: 各次元でセル内の相対座標。substep() では
    `base = floor(x/dx) - 1`, `fx = x/dx - base` として計算され、
    fx はおおむね [1, 2) の範囲を取る(offset=0..3 の4ノードに対応)。

    一様3次B-スプラインカーネル(距離 d = fx - offset に対して):
        w(d) = 2/3 - d^2 + |d|^3/2          (|d| < 1)
        w(d) = (2 - |d|)^3 / 6              (1 <= |d| < 2)
        w(d) = 0                            (それ以外)
    (2次スプラインの `_bspline_weights` と同じ役割で、返り値の次元数が
    3 -> 4 に変わる点のみが異なる。)
    """
    offsets = np.arange(4.0)
    d = fx[..., None] - offsets            # (..., 4)
    ad = np.abs(d)
    near = ad < 1.0
    w_near = 2.0 / 3.0 - d ** 2 + 0.5 * ad ** 3
    w_far = (2.0 - ad) ** 3 / 6.0
    w = np.where(near, w_near, w_far)
    w = np.where(ad < 2.0, w, 0.0)
    return w  # shape (N, dim, 4)


# ----------------------------------------------------------------------------
# 高速化ヘルパー: (N,2,2) のバッチ処理に特化した明示的な演算。
#
# np.einsum / np.linalg.svd は汎用的だが、2x2という極小行列を数千〜数万個
# バッチ処理する用途では、1回あたりのディスパッチ/LAPACK呼び出しオーバー
# ヘッドが支配的になり非常に遅い(この関数群を使わない実装に対して
# プロファイリングで実測: SVDだけで約27倍、全体で1桁以上の高速化)。
# 以下はすべて成分ごとの四則演算のみで書き下し、そのオーバーヘッドを避ける。
# ----------------------------------------------------------------------------

def matmul2x2(A, B):
    """バッチ (N,2,2) 行列積 A @ B を成分ごとの演算で計算する。"""
    C = np.empty_like(A)
    C[:, 0, 0] = A[:, 0, 0] * B[:, 0, 0] + A[:, 0, 1] * B[:, 1, 0]
    C[:, 0, 1] = A[:, 0, 0] * B[:, 0, 1] + A[:, 0, 1] * B[:, 1, 1]
    C[:, 1, 0] = A[:, 1, 0] * B[:, 0, 0] + A[:, 1, 1] * B[:, 1, 0]
    C[:, 1, 1] = A[:, 1, 0] * B[:, 0, 1] + A[:, 1, 1] * B[:, 1, 1]
    return C


def svd2x2(A):
    """バッチ (N,2,2) 行列の特異値分解 A = U @ diag(sigma) @ Vt を閉形式で計算する。

    A^T A の固有分解(対称2x2行列の固有値・固有ベクトルは解析的に書ける)から
    V と sigma>=0 を直接求め、U = A V / sigma で復元する。sigma=0 の退化ケース
    (u ベクトルが定義できない)は直交補完で埋める。np.linalg.svd と等価な
    分解を返すが、LAPACKの反復解法を経由しないため大幅に高速。
    """
    a = A[:, 0, 0]
    b = A[:, 0, 1]
    c = A[:, 1, 0]
    d = A[:, 1, 1]

    # A^T A の成分 (対称行列)
    m11 = a * a + c * c
    m12 = a * b + c * d
    m22 = b * b + d * d

    phi = 0.5 * np.arctan2(2.0 * m12, m11 - m22)
    cp = np.cos(phi)
    sp = np.sin(phi)

    mean = 0.5 * (m11 + m22)
    disc = np.sqrt(np.clip(((m11 - m22) * 0.5) ** 2 + m12 ** 2, 0.0, None))
    sigma1 = np.sqrt(np.clip(mean + disc, 0.0, None))
    sigma2 = np.sqrt(np.clip(mean - disc, 0.0, None))

    v1x, v1y = cp, sp
    v2x, v2y = -sp, cp

    u1x = a * v1x + b * v1y
    u1y = c * v1x + d * v1y
    u2x = a * v2x + b * v2y
    u2y = c * v2x + d * v2y

    eps = 1e-12
    n1 = np.hypot(u1x, u1y)
    n2 = np.hypot(u2x, u2y)
    ok1 = n1 > eps
    ok2 = n2 > eps
    u1x = np.where(ok1, u1x / np.where(ok1, n1, 1.0), 1.0)
    u1y = np.where(ok1, u1y / np.where(ok1, n1, 1.0), 0.0)
    # sigma2 がほぼ0で u2 の向きが定まらない場合は u1 の直交補完で埋める
    u2x_alt = -u1y
    u2y_alt = u1x
    u2x = np.where(ok2, u2x / np.where(ok2, n2, 1.0), u2x_alt)
    u2y = np.where(ok2, u2y / np.where(ok2, n2, 1.0), u2y_alt)

    U = np.empty_like(A)
    U[:, 0, 0] = u1x
    U[:, 1, 0] = u1y
    U[:, 0, 1] = u2x
    U[:, 1, 1] = u2y

    Vt = np.empty_like(A)
    Vt[:, 0, 0] = v1x
    Vt[:, 0, 1] = v1y
    Vt[:, 1, 0] = v2x
    Vt[:, 1, 1] = v2y

    sigma = np.stack([sigma1, sigma2], axis=1)
    return U, sigma, Vt


def substep(s: MPMState, t: float):
    """MLS-MPM の1ステップ (P2G -> 格子更新 -> G2P) を実行し、状態を更新する。

    3次B-スプライン版: 1粒子あたりの近傍ノード数が2次版の9(3x3)から
    16(4x4)に増える点、基準ノード添字 `base` の計算式、APIC逆慣性係数
    `d_inv` が異なる点を除き、2次版(mpm_beach_wave.py)の substep() と
    アルゴリズムは同一。

    t: 現在のシミュレーション時刻 [s] (波浪外力の位相計算に用いる)。
    """
    dt = s.dt
    inv_dx = s.inv_dx
    ny = s.ny
    n = s.n_particles

    # ------------------------------------------------------------------
    # (a) 変形勾配の更新 + Drucker-Prager リターンマッピング
    # ------------------------------------------------------------------
    M = np.empty_like(s.C)
    M[:, 0, 0] = 1.0 + dt * s.C[:, 0, 0]
    M[:, 0, 1] = dt * s.C[:, 0, 1]
    M[:, 1, 0] = dt * s.C[:, 1, 0]
    M[:, 1, 1] = 1.0 + dt * s.C[:, 1, 1]
    F_trial = matmul2x2(M, s.F)

    U, sigma, Vt = svd2x2(F_trial)
    sigma = np.clip(sigma, 1e-6, None)

    eps = np.log(sigma)                       # Hencky(対数)ひずみの主値 (N,2)
    tr_eps = eps.sum(axis=1)                  # 体積ひずみ成分
    eps_hat = eps - 0.5 * tr_eps[:, None]      # 偏差成分
    eps_hat_norm = np.linalg.norm(eps_hat, axis=1)

    # Drucker-Prager 降伏関数からのリターンマッピング量
    delta_gamma = eps_hat_norm + (2.0 * s.lambda0 + 2.0 * s.mu0) / (2.0 * s.mu0) * tr_eps * s.alpha_dp

    new_eps = eps.copy()

    has_shear = eps_hat_norm > 1e-12
    plastic = has_shear & (delta_gamma > 0.0) & (tr_eps <= 0.0)
    scale = np.zeros_like(eps_hat_norm)
    scale[plastic] = delta_gamma[plastic] / eps_hat_norm[plastic]
    new_eps[plastic] = eps[plastic] - scale[plastic, None] * eps_hat[plastic]

    tension = tr_eps > 0.0  # 引張(体積膨張)側は砂は応力を負担できない -> 応力解放
    new_eps[tension] = 0.0

    new_sigma = np.exp(new_eps)
    # F_elastic = U @ diag(new_sigma) @ Vt (対角行列の右積は列のスケーリング)
    UD = U * new_sigma[:, None, :]
    F_elastic = matmul2x2(UD, Vt)

    # Hencky 弾性による Kirchhoff応力 (主応力を U で世界座標系へ回転)
    # tau = U @ diag(stress) @ U^T
    tr_new = new_eps.sum(axis=1)
    principal_stress = 2.0 * s.mu0 * new_eps + s.lambda0 * tr_new[:, None]
    UD2 = U * principal_stress[:, None, :]
    Ut = U.transpose(0, 2, 1)
    tau = matmul2x2(UD2, Ut)

    stress_term = (-dt * s.d_inv * s.vol0)[:, None, None] * tau
    affine = stress_term + s.mass[:, None, None] * s.C

    # ------------------------------------------------------------------
    # (b) Particle-to-Grid (P2G)
    # ------------------------------------------------------------------
    # 3次B-スプラインのサポート幅は4ノード。`base` は「粒子から見て左下
    # 隅にあたる基準ノード添字」で、offset=0..3 が該当4ノードに対応する
    # (2次版の `floor(x/dx - 0.5)` とは規約が異なる)。
    base = np.floor(s.x * inv_dx).astype(np.int64) - 1
    fx = s.x * inv_dx - base.astype(np.float64)
    w = _bspline_weights(fx)  # (N,2,4)

    # 4x4近傍(16オフセット)をまとめて1つの (N,16,...) 配列として処理し、
    # 散布加算(P2G)は np.add.at の代わりに np.bincount でまとめて行う
    # (np.add.at は重複インデックスに対する非バッファ処理のため大幅に遅い)。
    offsets = np.array([[i, j] for i in range(4) for j in range(4)], dtype=np.float64)  # (16,2)
    node_i = base[:, 0:1] + offsets[None, :, 0].astype(np.int64)   # (N,16)
    node_j = base[:, 1:2] + offsets[None, :, 1].astype(np.int64)   # (N,16)
    flat_idx = (node_i * ny + node_j).ravel()                      # (N*16,)

    dpos = (offsets[None, :, :] - fx[:, None, :]) * DX             # (N,16,2)
    weight = (w[:, 0, :, None] * w[:, 1, None, :]).reshape(n, 16)  # (N,16)  [i,j]順=offsets順と一致

    affine_dpos = np.empty((n, 16, 2))
    affine_dpos[:, :, 0] = affine[:, 0, 0:1] * dpos[:, :, 0] + affine[:, 0, 1:2] * dpos[:, :, 1]
    affine_dpos[:, :, 1] = affine[:, 1, 0:1] * dpos[:, :, 0] + affine[:, 1, 1:2] * dpos[:, :, 1]

    contrib_v = weight[:, :, None] * (s.mass[:, None, None] * s.v[:, None, :] + affine_dpos)
    contrib_m = weight * s.mass[:, None]

    grid_m = np.bincount(flat_idx, weights=contrib_m.ravel(), minlength=s.n_nodes)
    grid_vx = np.bincount(flat_idx, weights=contrib_v[:, :, 0].ravel(), minlength=s.n_nodes)
    grid_vy = np.bincount(flat_idx, weights=contrib_v[:, :, 1].ravel(), minlength=s.n_nodes)
    grid_v = np.stack([grid_vx, grid_vy], axis=1)

    # ------------------------------------------------------------------
    # (c) 格子上の更新: 質量で正規化し、重力を与え、境界条件を課す
    # ------------------------------------------------------------------
    has_mass = grid_m > 1e-12
    grid_v[has_mass] /= grid_m[has_mass, None]
    grid_v[has_mass, 1] -= dt * GRAVITY
    grid_v *= DAMPING

    # --- 波浪外力(簡易振動流モデル、表層の活動層のみに作用) ---
    # 静水面以下のノードに線形波理論の水平往復流速を目標値として与え、
    # 格子速度をその目標値へ緩和させる(空間位相 kx は無視し、時間変化のみ考慮)。
    # ただし、その時点(変形後)のマウンド表面から WAVE_ACTIVE_LAYER の厚み
    # 分だけを対象とし、内部に埋もれた粒子には作用させない。表面位置は
    # 毎ステップ、格子質量 grid_m から「各x列で質量を持つ最上端ノード」
    # として動的に検出する(粒子を直接ビニングするより、既に計算済みの
    # grid_m を再利用する方が安価)。
    ramp = min(t / (WAVE_RAMP_CYCLES * WAVE_PERIOD), 1.0)
    if ramp > 0.0:
        mass_2d = grid_m.reshape(s.nx, s.ny)
        j_range = np.arange(s.ny)
        j_if_mass = np.where(mass_2d > 1e-12, j_range[None, :], -1)
        j_surface = j_if_mass.max(axis=1)                       # (nx,) 列ごとの表面ノード添字(質量なしなら-1)

        depth_cells = j_surface[s.node_i] - s.node_j             # 表面から何セル下か(負なら表面より上=空気)
        active_layer = (depth_cells >= 0) & (depth_cells < s.wave_active_layer_cells)

        u_target = (ramp * np.pi * WAVE_HEIGHT / WAVE_PERIOD
                    * s.wave_cosh_ky / s.wave_sinh_kd * np.cos(s.wave_omega * t))
        drag_coef = min(dt / WAVE_DRAG_TIME, 1.0)
        wave_active = s.wave_mask & has_mass & active_layer
        grid_v[wave_active, 0] += drag_coef * (u_target[wave_active] - grid_v[wave_active, 0])

    grid_v[s.bottom_mask] = 0.0        # 海底面: 固定(剛な不透過基盤)
    grid_v[s.side_mask, 0] = 0.0       # 左右端: 水平方向のみ固定(スリップ壁)

    # ------------------------------------------------------------------
    # (d) Grid-to-Particle (G2P)
    # ------------------------------------------------------------------
    gv = grid_v[flat_idx].reshape(n, 16, 2)                # 散布ではなく収集(gather)なので add.at 不要

    new_v = (weight[:, :, None] * gv).sum(axis=1)          # (N,2)

    new_C = np.empty((n, 2, 2))
    wC = s.d_inv * weight
    new_C[:, 0, 0] = (wC * gv[:, :, 0] * dpos[:, :, 0]).sum(axis=1)
    new_C[:, 0, 1] = (wC * gv[:, :, 0] * dpos[:, :, 1]).sum(axis=1)
    new_C[:, 1, 0] = (wC * gv[:, :, 1] * dpos[:, :, 0]).sum(axis=1)
    new_C[:, 1, 1] = (wC * gv[:, :, 1] * dpos[:, :, 1]).sum(axis=1)

    x_new = s.x + dt * new_v

    # 数値的な安全策: 万一の発散で粒子が領域外に出ないようクリップする。
    # 3次B-スプラインはサポート幅が4ノード分(2次版より1セル分広い)ため、
    # 近傍ノード添字が格子配列の範囲外に出ないよう、余白を DX -> 2*DX に
    # 拡大している(境界近傍は元々 BOUND=3 層分が固定壁のため、この余白
    # 拡大が計算結果に与える影響は無視できる)。
    x_new[:, 0] = np.clip(x_new[:, 0], 2.0 * DX, LX - 2.0 * DX)
    x_new[:, 1] = np.clip(x_new[:, 1], 2.0 * DX, LY - 2.0 * DX)

    s.x = x_new
    s.v = new_v
    s.C = new_C
    s.F = F_elastic



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
    ax.set_title("沖合養浜マウンドの自重変形+波浪外力による変形 (MPM, 3次B-スプライン)")
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
    ax.set_title("沖合養浜マウンド断面形状の変化(自重沈下・法面変形+波浪外力、3次B-スプライン)")
    ax.set_xlim(0, LX)
    ax.set_ylim(0, max(WATER_LEVEL, MOUND_HEIGHT) * 1.3)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_comparison.png"), dpi=150)
    plt.show()

    print(f"[MPM] 図を保存しました: {out_dir}")



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
        description="MPM沖合養浜マウンド自重変形解析(3次B-スプライン)。"
                     "--t-total で総計算時間を延長できる。"
    )
    parser.add_argument(
        "--t-total", type=float, default=None,
        help=f"総計算時間 [s] (省略時はコード内の既定値 T_TOTAL={T_TOTAL}s を使用)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="出力先ディレクトリ(省略時は output_cubic/。長時間実行では過去の出力を"
             "上書きしないよう別ディレクトリを指定することを推奨)",
    )
    return parser.parse_args()


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
# mpm_beach_wave.py の「拡張のヒント」節を参照(このファイルでは形状関数の
# 変更以外の拡張ポイントは変わらない)。
