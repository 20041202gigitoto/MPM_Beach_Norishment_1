"""
mpm_beach.py
============
Material Point Method (MPM) のみを用いた、仮想の**沖合養浜マウンド断面**の
自重下変形解析 — 最小構成の基礎コード。

沖合養浜とは
------------
養浜のうち、前浜(汀線付近)に直接土砂を盛って渚を広げる「前浜養浜」とは異なり、
**沖合の海底上に土砂マウンド(人工海底山・サンドバー)を造成し、常時水没させた
状態で設置する**手法を「沖合養浜」と呼ぶ。マウンドは静水面(SWL)より十分深い
位置に天端があり、波によって徐々に土砂が岸側へ移動する(あるいは波浪減衰・
底質供給源として機能する)ことを狙いとする。

このコードでできること
-----------------------
- 沖合の海底(平坦なローカル基盤)上に設置された、水没マウンド断面
  (沖側スロープ+天端+岸側スロープ、天端は静水面下)を MPM 粒子群として生成する。
- 重力(水中の浮力を考慮した水中単位体積重量)のみを外力として、マウンドが
  自重で沈下・法面が崩れて安定形状(水中安息角)に近づく様子を計算する。
- 静水面(SWL)を基準線として図示し、マウンドが常に水没した状態にあることを
  可視化する。
- 計算結果(粒子位置)をスナップショット保存し、初期断面と最終断面を
  比較するプロットを出力する。

このコードで「できないこと」(意図的に含めていない要素)
-----------------------------------------------------
- 波浪外力・潮位変動・水とのカップリング(波による移動・侵食)は含まない
  (自重変形のみ。沖合養浜の本質である「波による土砂移動」は再現しない)。
- 侵食・堆積量の定量評価(体積収支等)は含まない。
- 複数材料(海底の在来砂と養浜材の区別)、海底勾配(沖に向かう水深変化)は
  含まない。単一材料・局所的に平坦な海底のみ。
これらは本コードを土台として、必要に応じて拡張していくことを想定している
(ファイル末尾の「拡張のヒント」を参照)。

数値手法
--------
- MLS-MPM (Moving Least Squares MPM, Hu et al. 2018) を陽解法・
  2次 B-スプライン形状関数・APIC (Affine Particle-In-Cell) で実装。
- 砂の構成則: Hencky(対数ひずみ)弾性 + Drucker-Prager 完全塑性。
  変形勾配 F の特異値分解(SVD)を用いたリターンマッピングにより
  塑性変形を扱う (Klar et al. 2016, "Drucker-Prager Elastoplasticity
  for Sand Animation", ACM TOG/SIGGRAPH 2016 の手法に基づく)。
- 依存ライブラリは numpy と matplotlib のみ。

参考: Stomakhin et al. (2013) MLS-MPM/APIC の枠組みは
Hu et al. (2018) "A Moving Least Squares Material Point Method with
Displacement Discontinuity and Two-Way Rigid Body Coupling" のアルゴリズムに
概ね準拠(教育目的で公開されている軽量MPM実装 (mpm99/mpm128 系) と
同様の定式化)。

VSCode での実行方法
--------------------
- Python拡張機能がインストールされていれば、このファイルは `# %%` の
  セル区切りごとに「Run Cell」でインタラクティブウィンドウ実行できる
  (matplotlibの図がウィンドウ内にインライン表示される)。
- 単純に「Run Python File」(通常のスクリプト実行)でも、グラフウィンドウが
  ポップアップ表示され、かつ output/ に PNG が保存される。
- 推奨インタプリタ: mpm_scratch/.venv (numpy, matplotlib のみを含む
  専用の軽量venv。 .vscode/settings.json で自動的に指定される)。
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

# ---- 数値計算設定 ----
CFL = 0.3            # クーラン数(安定限界に対する余裕係数)
DAMPING = 0.999      # 格子速度への簡易減衰係数(1.0で減衰なし、微小な数値粘性)
T_TOTAL = 1.2        # 総計算時間 [s]
FRAME_DT = 0.02      # スナップショットを保存する時間間隔 [s]

# ---- 出力 ----
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")



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
        self.d_inv = 4.0 * self.inv_dx * self.inv_dx  # 2次Bスプラインの逆慣性係数

        i_idx = np.repeat(np.arange(self.nx), self.ny)
        j_idx = np.tile(np.arange(self.ny), self.nx)
        self.bottom_mask = j_idx < BOUND
        self.side_mask = (i_idx < BOUND) | (i_idx > self.nx - 1 - BOUND)

        # --- 時間刻み(CFL条件) ---
        wave_speed = np.sqrt((self.lambda0 + 2.0 * self.mu0) / RHO0)
        dt_cfl = CFL * DX / wave_speed
        self.n_substeps = max(1, int(np.ceil(FRAME_DT / dt_cfl)))
        self.dt = FRAME_DT / self.n_substeps

        print(f"[MPM] 粒子数 = {n}")
        print(f"[MPM] 格子ノード数 = {self.nx} x {self.ny} = {self.n_nodes}")
        print(f"[MPM] dt = {self.dt:.3e} s, {self.n_substeps} substeps/frame")
        print(f"[MPM] マウンド天端の被覆水深(水没深さ) = {SUBMERGENCE:.3f} m "
              f"(静水面 y={WATER_LEVEL:.2f} m, 天端 y={MOUND_HEIGHT:.2f} m)")
        if SUBMERGENCE <= 0.0:
            print("[MPM] 警告: マウンド天端が静水面より高く、水没条件を満たしていません。")


def _bspline_weights(fx):
    """2次 B-スプライン形状関数の重み。fx: 各次元でセル内の相対座標(範囲[0.5,1.5))"""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return np.stack([w0, w1, w2], axis=-1)  # shape (N, dim, 3)


def substep(s: MPMState):
    """MLS-MPM の1ステップ (P2G -> 格子更新 -> G2P) を実行し、状態を更新する。"""
    dt = s.dt
    inv_dx = s.inv_dx
    ny = s.ny

    grid_m = np.zeros(s.n_nodes)
    grid_v = np.zeros((s.n_nodes, 2))

    # ------------------------------------------------------------------
    # (a) 変形勾配の更新 + Drucker-Prager リターンマッピング
    # ------------------------------------------------------------------
    identity = np.eye(2)
    F_trial = np.einsum("nij,njk->nik", identity[None, :, :] + dt * s.C, s.F)

    U, sigma, Vt = np.linalg.svd(F_trial)
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
    Sigma_F = np.zeros_like(s.F)
    Sigma_F[:, 0, 0] = new_sigma[:, 0]
    Sigma_F[:, 1, 1] = new_sigma[:, 1]
    F_elastic = np.einsum("nij,njk,nlk->nil", U, Sigma_F, Vt)  # U @ diag(new_sigma) @ V^T

    # Hencky 弾性による Kirchhoff応力 (主応力を U で世界座標系へ回転)
    tr_new = new_eps.sum(axis=1)
    principal_stress = 2.0 * s.mu0 * new_eps + s.lambda0 * tr_new[:, None]
    Sigma_stress = np.zeros_like(s.F)
    Sigma_stress[:, 0, 0] = principal_stress[:, 0]
    Sigma_stress[:, 1, 1] = principal_stress[:, 1]
    tau = np.einsum("nij,njk,nlk->nil", U, Sigma_stress, U)  # U @ diag(stress) @ U^T

    stress_term = (-dt * s.d_inv * s.vol0)[:, None, None] * tau
    affine = stress_term + s.mass[:, None, None] * s.C

    # ------------------------------------------------------------------
    # (b) Particle-to-Grid (P2G)
    # ------------------------------------------------------------------
    base = np.floor(s.x * inv_dx - 0.5).astype(np.int64)
    fx = s.x * inv_dx - base.astype(np.float64)
    w = _bspline_weights(fx)  # (N,2,3)

    for i in range(3):
        for j in range(3):
            node_i = base[:, 0] + i
            node_j = base[:, 1] + j
            flat_idx = node_i * ny + node_j
            dpos = (np.array([i, j], dtype=np.float64)[None, :] - fx) * DX
            weight = w[:, 0, i] * w[:, 1, j]

            contrib_v = weight[:, None] * (
                s.mass[:, None] * s.v + np.einsum("nij,nj->ni", affine, dpos)
            )
            np.add.at(grid_v, flat_idx, contrib_v)
            np.add.at(grid_m, flat_idx, weight * s.mass)

    # ------------------------------------------------------------------
    # (c) 格子上の更新: 質量で正規化し、重力を与え、境界条件を課す
    # ------------------------------------------------------------------
    has_mass = grid_m > 1e-12
    grid_v[has_mass] /= grid_m[has_mass, None]
    grid_v[has_mass, 1] -= dt * GRAVITY
    grid_v *= DAMPING

    grid_v[s.bottom_mask] = 0.0        # 海底面: 固定(剛な不透過基盤)
    grid_v[s.side_mask, 0] = 0.0       # 左右端: 水平方向のみ固定(スリップ壁)

    # ------------------------------------------------------------------
    # (d) Grid-to-Particle (G2P)
    # ------------------------------------------------------------------
    new_v = np.zeros_like(s.v)
    new_C = np.zeros_like(s.C)

    for i in range(3):
        for j in range(3):
            node_i = base[:, 0] + i
            node_j = base[:, 1] + j
            flat_idx = node_i * ny + node_j
            dpos = (np.array([i, j], dtype=np.float64)[None, :] - fx) * DX
            weight = w[:, 0, i] * w[:, 1, j]

            gv = grid_v[flat_idx]
            new_v += weight[:, None] * gv
            new_C += s.d_inv * weight[:, None, None] * np.einsum("ni,nj->nij", gv, dpos)

    x_new = s.x + dt * new_v

    # 数値的な安全策: 万一の発散で粒子が領域外に出ないようクリップする
    x_new[:, 0] = np.clip(x_new[:, 0], DX, LX - DX)
    x_new[:, 1] = np.clip(x_new[:, 1], DX, LY - DX)

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
    ax.set_title("沖合養浜マウンドの自重変形 (MPM)")
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
    ax.set_title("沖合養浜マウンド断面形状の変化(自重沈下・法面変形)")
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
            substep(state)
        t += FRAME_DT
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



if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    state, snapshots = run()


# ============================================================================
# 拡張のヒント (研究を進める際の追加要素の例)
# ============================================================================
# - 波浪外力: 沖合養浜の本質は波によるマウンドの変形・移動なので、静水面
#   (WATER_LEVEL)以下の格子ノードに、時間変化する水平方向の底面せん断力・
#   波圧を追加するのが最初の一歩(例: substep() の grid_v 更新部分に外力項を
#   追加)。本格的にはXBeach等の外部モデルの出力(水位・波高・底面流速等)を
#   読み込み、時間ステップごとに力へ変換して与える一方向カップリングへ
#   拡張できる。
# - 海底勾配: 現状は局所的に平坦な海底(y=0)を仮定しているが、
#   mound_surface_height() のベースラインを x の関数(例: 沖に向かって
#   深くなる勾配)に置き換えれば、実際の海浜横断形状の上にマウンドを
#   設置した解析ができる。
# - 侵食・堆積の定量評価: フレームごとに top_envelope() を計算して保存し、
#   初期からの高さ変化 dz(x) や断面積変化(侵食量・堆積量)を時系列で追跡する。
# - 複数材料: 粒子ごとに material_id 配列を持たせ、密度・ヤング率・摩擦角を
#   材料IDごとに切り替える(例: 在来海底砂 と 養浜材(礫)を区別する)。
# - 粘着力(cohesion): Drucker-Prager 降伏関数に定数項を加え、
#   delta_gamma の閾値をオフセットすることでシルト分を含む養浜材等の
#   粘着力を表現できる。
# - 間隙水圧・排水過程: 現状は「密度を水中密度に置き換える」簡易的な浮力
#   モデルのみ。過剰間隙水圧の発生・消散(圧密)を扱う二相系(砂粒子+間隙水)
#   に拡張すれば、地震時液状化等も評価できる。
# - 実測断面への置き換え: mound_surface_height() を、実測地形の
#   点列を線形補間する関数に差し替えれば、任意の実海岸・実マウンド形状へ
#   適用できる。
# - 高速化: 粒子数が多い場合は Taichi 等でこのアルゴリズムをGPU化すると
#   大規模・長時間の計算が現実的になる(本コードのロジックは
#   Taichiへの移植を前提に、1ステップの処理を明示的に分解して書いている)。
