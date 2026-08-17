"""
mpm_beach_wave_slope_erosion.py (Numba版)
============================================
mpm_beach_wave_slope.py (Numba版) の計算ロジック(MLS-MPM + Drucker-Prager
砂塑性 + 簡易波浪外力、Numba (@njit) でJITコンパイルされた substep)を
そのまま再利用しつつ、侵食・堆積の過程を時系列で追跡できるようにした版
(mpm_scratch/mpm_beach_wave_slope_erosion.py の Numba 高速化版)。

mpm_scratch版との違い
----------------------
- MPMソルバー本体(MPMState, substep, 粒子生成, bed_elevation など)の
  インポート元が `mpm_beach_wave_slope`(このディレクトリ内の Numba版)に
  なっている点のみが異なる。侵食・堆積の追跡ロジック自体(surface_profile,
  area_changes, run, 可視化)は mpm_scratch版と完全に同一。

mpm_beach_wave_slope.py (Numba版) との違い
---------------------------------------------
- 毎フレーム、マウンド表面の高さプロファイル(top_envelope)を固定のx位置
  ビンに対して記録し、初期断面からの高さ変化 dz(x, t) = surface(x, t) -
  surface(x, 0) を時系列で保存する(堆積: dz>0, 侵食: dz<0)。
- dz(x, t) を x 方向に積分して、堆積量・侵食量・正味の断面積変化を
  フレームごとの時系列として記録する。
- IMAGE_SAVE_INTERVAL_FRAMES フレームごと(既定100フレーム)に、初期断面
  との差分(堆積・侵食域を塗り分け)を示す断面プロファイル画像を保存する。
- 計算終了後に、(a) 表面プロファイルの時間発展の重ね描き、
  (b) dz(x, t) のヒートマップ(侵食・堆積分布の時空間発展)、
  (c) 堆積量・侵食量・正味変化量の時系列グラフ、および生データ(.npz)を出力する。

実行方法
--------
    python mpm_beach_wave_slope_erosion.py
    python mpm_beach_wave_slope_erosion.py --t-total 12.0 --image-interval 100


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
import argparse

import numpy as np
import matplotlib.pyplot as plt

import mpm_beach_wave_slope as base

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# %%
# ============================================================================
# 1. このファイル独自のパラメータ
# ============================================================================

TRACK_INTERVAL_FRAMES = 1          # 表面プロファイル(dz)を記録するフレーム間隔
IMAGE_SAVE_INTERVAL_FRAMES = 100   # top_envelope(差分)画像を保存するフレーム間隔
ENVELOPE_BIN_WIDTH = 4 * base.DX   # ビニング幅(base.top_envelope の既定値と同じ)

OUTPUT_DIR = os.path.join(base.OUTPUT_DIR, "erosion_tracking")


# %%
# ============================================================================
# 2. 表面プロファイル抽出(フレーム間で位置が対応する固定ビン版)
# ============================================================================

def surface_profile(x, y, x_max=base.LX, bin_width=ENVELOPE_BIN_WIDTH):
    """粒子群からx方向にビニングし、各binの表面高さ(マウンド砂の最高点)を返す。

    base.top_envelope() は粒子が存在しないbinを詰めて(valid配列で)返すため、
    フレームごとにビンの本数・位置が変わり得て時系列比較に使いにくい。
    この関数はビン配列を固定することで、どのフレームでも同じインデックス=
    同じx位置を指すようにしている(dz(x, t) をフレーム間で単純な引き算で
    比較できるようにするため)。

    粒子が存在しないbinは「そのx位置の砂の厚みが0(海底がそのまま露出)」を
    意味するため、np.nanではなく bed_elevation(x) を高さとして返す。
    """
    bins = np.arange(0.0, x_max + bin_width, bin_width)
    idx = np.digitize(x, bins) - 1
    n_bins = len(bins) - 1

    centers = 0.5 * (bins[:-1] + bins[1:])
    top_y = base.bed_elevation(centers)
    for b in range(n_bins):
        mask = idx == b
        if np.any(mask):
            top_y[b] = max(top_y[b], y[mask].max())

    return centers, top_y


def area_changes(dz, bin_width=ENVELOPE_BIN_WIDTH):
    """dz(x) から堆積量・侵食量・正味変化量(断面積換算, m^2)を求める。

    堆積量・侵食量はいずれも正の値として返す(侵食量は dz<0 側の絶対値)。
    """
    deposition = np.sum(np.clip(dz, 0.0, None)) * bin_width
    erosion = -np.sum(np.clip(dz, None, 0.0)) * bin_width
    net = deposition - erosion
    return deposition, erosion, net


# %%
# ============================================================================
# 3. 実行ループ(base.substep を流用し、毎フレーム表面プロファイルを記録)
# ============================================================================

def run(t_total=None, output_dir=None):
    t_total = base.T_TOTAL if t_total is None else t_total
    out_dir = OUTPUT_DIR if output_dir is None else output_dir
    frames_dir = os.path.join(out_dir, "envelope_frames")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    state = base.MPMState()
    n_frames = int(round(t_total / base.FRAME_DT))

    x_centers, y0_env = surface_profile(state.x[:, 0], state.x[:, 1])

    times = [0.0]
    envelopes = [y0_env]
    dz_series = [np.zeros_like(y0_env)]
    dep_series = [0.0]
    ero_series = [0.0]
    net_series = [0.0]

    _save_envelope_image(x_centers, y0_env, y0_env, 0.0, frames_dir, frame_index=0)

    t = 0.0
    for frame in range(n_frames):
        for _ in range(state.n_substeps):
            base.substep(state, t)
            t += state.dt

        if (frame + 1) % max(1, n_frames // 10) == 0 or frame == n_frames - 1:
            print(f"[erosion] frame {frame + 1}/{n_frames}  t={t:.3f}s  "
                  f"max|v|={np.linalg.norm(state.v, axis=1).max():.3f} m/s")

        need_track = (frame + 1) % TRACK_INTERVAL_FRAMES == 0 or frame == n_frames - 1
        need_image = (frame + 1) % IMAGE_SAVE_INTERVAL_FRAMES == 0 or frame == n_frames - 1

        if need_track or need_image:
            _, y_env = surface_profile(state.x[:, 0], state.x[:, 1])
            dz = y_env - y0_env

            if need_track:
                dep, ero, net = area_changes(dz)
                times.append(t)
                envelopes.append(y_env)
                dz_series.append(dz)
                dep_series.append(dep)
                ero_series.append(ero)
                net_series.append(net)

            if need_image:
                _save_envelope_image(x_centers, y0_env, y_env, t, frames_dir, frame_index=frame + 1)

    times = np.array(times)
    envelopes = np.stack(envelopes, axis=0)
    dz_series = np.stack(dz_series, axis=0)
    dep_series = np.array(dep_series)
    ero_series = np.array(ero_series)
    net_series = np.array(net_series)

    np.savez(
        os.path.join(out_dir, "erosion_deposition_timeseries.npz"),
        t=times, x=x_centers, envelope=envelopes, dz=dz_series,
        deposition_area=dep_series, erosion_area=ero_series, net_area=net_series,
    )

    _plot_profile_evolution(times, x_centers, envelopes, out_dir)
    _plot_summary(times, x_centers, dz_series, dep_series, ero_series, net_series, out_dir)

    print(f"[erosion] 時系列データ・図を保存しました: {out_dir}")
    return state, times, x_centers, envelopes, dz_series


# %%
# ============================================================================
# 4. 可視化
# ============================================================================

def _save_envelope_image(x_centers, y0_env, y_env, t, out_dir, frame_index):
    """初期断面との差分(堆積・侵食域)を塗り分けた断面プロファイル画像を保存する。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_centers, y0_env, "--", color="gray", label="初期断面 (t=0.00s)")
    ax.plot(x_centers, y_env, "-", color="tab:orange", label=f"現在断面 (t={t:.2f}s)")

    dz = y_env - y0_env
    ax.fill_between(x_centers, y0_env, y_env, where=(dz >= 0),
                     color="tab:green", alpha=0.3, interpolate=True, label="堆積")
    ax.fill_between(x_centers, y0_env, y_env, where=(dz < 0),
                     color="tab:red", alpha=0.3, interpolate=True, label="侵食")

    bed_x = np.linspace(0.0, base.LX, 200)
    ax.plot(bed_x, base.bed_elevation(bed_x), color="tab:brown", linewidth=1.5, label="海底")
    ax.axhline(base.WATER_LEVEL, color="tab:blue", linestyle=":", linewidth=1.5,
               label=f"静水面 SWL (y={base.WATER_LEVEL:.2f}m)")

    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_title(f"侵食・堆積分布 (frame={frame_index}, t={t:.2f}s)")
    ax.set_xlim(0, base.LX)
    ax.set_ylim(0, base.LY)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"envelope_frame_{frame_index:04d}.png"), dpi=150)
    plt.close(fig)


def _plot_profile_evolution(times, x_centers, envelopes, out_dir):
    """表面プロファイルの時間発展を重ね描きする(古い時刻ほど薄い紫、新しいほど黄)。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    n = len(times)
    for k in range(n):
        color = plt.cm.viridis(k / max(n - 1, 1))
        alpha = 0.25 if 0 < k < n - 1 else 0.9
        lw = 1 if 0 < k < n - 1 else 2
        ax.plot(x_centers, envelopes[k], color=color, alpha=alpha, linewidth=lw,
                label=f"t={times[k]:.2f}s" if k in (0, n - 1) else None)

    bed_x = np.linspace(0.0, base.LX, 200)
    ax.plot(bed_x, base.bed_elevation(bed_x), color="tab:brown", linewidth=1.5, label="海底")
    ax.axhline(base.WATER_LEVEL, color="tab:blue", linestyle=":", linewidth=1.5,
               label=f"静水面 SWL (y={base.WATER_LEVEL:.2f}m)")

    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_title("マウンド表面プロファイルの時間発展")
    ax.set_xlim(0, base.LX)
    ax.set_ylim(0, base.LY)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_evolution.png"), dpi=150)


def _plot_summary(times, x_centers, dz_series, dep_series, ero_series, net_series, out_dir):
    # (a) dz(x, t) ヒートマップ: 侵食・堆積分布の時空間発展
    fig, ax = plt.subplots(figsize=(9, 4))
    dz_masked = np.ma.masked_invalid(dz_series)
    vmax = np.abs(dz_masked).max() if dz_masked.count() > 0 else 1.0
    mesh = ax.pcolormesh(x_centers, times, dz_masked, shading="nearest",
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="高さ変化 dz [m] (+堆積 / -侵食)")
    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("時刻 t [s]")
    ax.set_title("侵食・堆積分布の時間発展 dz(x, t)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "erosion_deposition_heatmap.png"), dpi=150)

    # (b) 堆積量・侵食量・正味変化量の時系列
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, dep_series, color="tab:green", label="堆積量(断面積換算)")
    ax.plot(times, -ero_series, color="tab:red", label="侵食量(断面積換算, 負値表示)")
    ax.plot(times, net_series, color="black", linewidth=1.5, label="正味変化量(堆積-侵食)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("時刻 t [s]")
    ax.set_ylabel("断面積変化 [m^2] (奥行き1mあたり)")
    ax.set_title("侵食・堆積量の時系列")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "erosion_deposition_volume_timeseries.png"), dpi=150)

    plt.show()


# %%
# ============================================================================
# 5. メイン実行
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MPM沖合養浜マウンド解析(海底勾配あり、Numba版)+ 侵食・堆積の時系列追跡。"
    )
    parser.add_argument("--t-total", type=float, default=None,
                         help=f"総計算時間 [s] (省略時は既定値 T_TOTAL={base.T_TOTAL}s)")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="出力先ディレクトリ(省略時は output/erosion_tracking/)")
    parser.add_argument("--track-interval", type=int, default=TRACK_INTERVAL_FRAMES,
                         help="表面プロファイル(dz)を記録するフレーム間隔")
    parser.add_argument("--image-interval", type=int, default=IMAGE_SAVE_INTERVAL_FRAMES,
                         help="断面差分(堆積・侵食)画像を保存するフレーム間隔")
    # VSCode の Interactive Window / Jupyter 経由で実行すると sys.argv に
    # カーネル起動用の引数(-f <connection_file> 等)が混入するため、
    # 未知の引数は無視する parse_known_args() を使う。
    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    TRACK_INTERVAL_FRAMES = args.track_interval
    IMAGE_SAVE_INTERVAL_FRAMES = args.image_interval

    run(t_total=args.t_total, output_dir=args.output_dir)
