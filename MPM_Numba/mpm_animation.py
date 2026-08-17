"""
mpm_animation.py (Numba版 共通ユーティリティ)
================================================
mpm_beach*.py 系スクリプトの run() が計算ループ中に集めるスナップショット列
snapshots = [(t, x), ...] (t: 時刻 [s], x: 粒子座標配列 (n_particles, 2)) から、
計算開始~終了までの粒子の動きを1本のGIFアニメーションとして書き出す共通
ヘルパー。各メインスクリプトの plot_results() から呼び出され、既存の静止画
(particles_overlay.png / profile_comparison.png) と同じタイミング(メイン
計算ループ終了直後、同一の run() 呼び出しの中)でGIFも生成する。新たな
計算のやり直しは不要で、既に持っているsnapshotsをそのまま流用する。

呼び出し元スクリプトは(VSCodeインタラクティブウィンドウでのインライン表示
のため)pyplotの対話的バックエンド(macosx等)を使うが、そのままアニメー
ション描画に使うと、Numbaのスレッドプールでの並列JITコンパイル後にGUI
バックエンドでフレームを連続描画した際、まれに(サイレントな)ネイティブ
クラッシュが発生することを確認した。これを避けるため、本モジュールは
pyplotを経由せず Figure + FigureCanvasAgg を直接使い、呼び出し元の
バックエンド設定に関係なく常にオフスクリーン(Agg)でレンダリングする。
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FuncAnimation, PillowWriter


def save_particle_gif(
    snapshots,
    out_path,
    lx,
    ly,
    water_level,
    bed_x=None,
    bed_y=None,
    title="粒子運動アニメーション",
    fps=25,
    dpi=100,
    max_frames=300,
):
    """粒子位置スナップショットからGIFアニメーションを生成して保存する。

    Parameters
    ----------
    snapshots : list[tuple[float, np.ndarray]]
        (時刻 t [s], 粒子座標 x (n_particles, 2)) のリスト。計算ループ中に
        フレームごとに記録されたものをそのまま渡す想定(t=0の初期状態を
        含む)。
    out_path : str
        出力するGIFファイルのパス。
    lx, ly : float
        描画範囲(x方向・y方向の計算領域サイズ)[m]。
    water_level : float
        静水面の高さ [m](水平点線として描画)。
    bed_x, bed_y : np.ndarray, optional
        海底地形を表す線を描画する場合の x, y 座標配列(勾配ありの版のみ
        指定する。省略時は海底線を描画しない)。
    title : str
        グラフタイトル(時刻表示が末尾に追記される)。
    fps : int
        GIFの再生フレームレート。
    dpi : int
        出力解像度。
    max_frames : int
        GIFに含める最大フレーム数。snapshotsがこれより多い場合は等間隔で
        間引く(フレーム数が多すぎるとファイルサイズ・生成時間が過大になる
        ため)。先頭・末尾フレームは必ず含める。
    """
    n_total = len(snapshots)
    if n_total == 0:
        raise ValueError("snapshots が空です")

    if n_total > max_frames:
        idx = np.linspace(0, n_total - 1, max_frames).round().astype(int)
        idx = sorted(set(idx.tolist()))
    else:
        idx = list(range(n_total))
    frames = [snapshots[i] for i in idx]

    fig = Figure(figsize=(9, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    t0, x0 = frames[0]
    scat = ax.scatter(x0[:, 0], x0[:, 1], s=3, color="tab:orange")

    if bed_x is not None and bed_y is not None:
        ax.plot(bed_x, bed_y, color="tab:brown", linewidth=1.5, label="海底")

    ax.axhline(water_level, color="tab:blue", linestyle=":", linewidth=1.5,
               label=f"静水面 SWL (y={water_level:.2f}m)")

    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title(f"{title} (t={t0:.2f}s)")
    fig.tight_layout()

    def update(frame_idx):
        t, x = frames[frame_idx]
        scat.set_offsets(x)
        ax.set_title(f"{title} (t={t:.2f}s)")
        return scat,

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)

    print(f"[MPM] GIFアニメーションを保存しました: {out_path} "
          f"({len(frames)}フレーム, 元スナップショット数={n_total})")
