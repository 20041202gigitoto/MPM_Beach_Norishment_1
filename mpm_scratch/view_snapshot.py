"""
view_snapshot.py
=================
mpm_beach.py が output/ に保存する snapshot_XXXX.npz の中身を閲覧するための
補助スクリプト。npz はNumPyのバイナリ形式なのでテキストエディタでは開けない
ため、このスクリプトで (1) 内容をテキスト表示、(2) 断面図として可視化、
(3) CSVへ変換、のいずれかを行う。

使い方
------
VSCode の「Run Python File」でそのまま実行すると、デフォルトの
snapshot_0000.npz を表示する。特定のファイルや全部を見たい場合は、
下の "設定" セルの TARGET / FRAME_INDEX を書き換えるか、コマンドラインから

    python view_snapshot.py output/snapshot_0059.npz
    python view_snapshot.py output/snapshot_0059.npz --csv
    python view_snapshot.py output/snapshot_0059.npz --plot-only

のように引数で指定する。
"""

# %%
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# %%
def load_snapshot(path):
    """1つの snapshot_XXXX.npz を読み込み、(t, x, v) を返す。"""
    with np.load(path) as d:
        t = float(d["t"])
        x = d["x"]
        v = d["v"]
    return t, x, v


def summarize(path):
    """内容をテキストで表示する(件数・座標範囲・速度範囲など)。"""
    t, x, v = load_snapshot(path)
    speed = np.linalg.norm(v, axis=1)

    print(f"--- {os.path.basename(path)} ---")
    print(f"  時刻 t          = {t:.4f} s")
    print(f"  粒子数           = {x.shape[0]}")
    print(f"  x範囲(岸沖距離)  = [{x[:, 0].min():.4f}, {x[:, 0].max():.4f}] m")
    print(f"  y範囲(高さ)      = [{x[:, 1].min():.4f}, {x[:, 1].max():.4f}] m")
    print(f"  最大速度 |v|max  = {speed.max():.4f} m/s")
    print(f"  平均速度 |v|mean = {speed.mean():.4f} m/s")
    print(f"  先頭5粒子の位置 x[:5] =\n{x[:5]}")
    print(f"  先頭5粒子の速度 v[:5] =\n{v[:5]}")


def plot_snapshot(path, out_path=None, show=True):
    """1スナップショットの粒子位置を、速度の大きさで色分けして可視化する。"""
    t, x, v = load_snapshot(path)
    speed = np.linalg.norm(v, axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(x[:, 0], x[:, 1], c=speed, cmap="viridis", s=4)
    fig.colorbar(sc, ax=ax, label="速度の大きさ |v| [m/s]")
    ax.set_xlabel("岸沖距離 x [m]")
    ax.set_ylabel("高さ y [m]")
    ax.set_title(f"{os.path.basename(path)}  (t={t:.3f}s, N={x.shape[0]})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[view_snapshot] 図を保存しました: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def export_csv(path, out_path=None):
    """x, v を1粒子1行のCSV(t, x, y, vx, vy)へ変換する(Excel等で閲覧用)。"""
    t, x, v = load_snapshot(path)
    if out_path is None:
        out_path = os.path.splitext(path)[0] + ".csv"

    header = "t,x,y,vx,vy"
    data = np.column_stack([np.full(x.shape[0], t), x[:, 0], x[:, 1], v[:, 0], v[:, 1]])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[view_snapshot] CSVを保存しました: {out_path}")
    return out_path


def resolve_target(target):
    """フレーム番号(int)・ファイル名・フルパスのいずれでも受け付ける。"""
    if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
        return os.path.join(OUTPUT_DIR, f"snapshot_{int(target):04d}.npz")
    if os.path.dirname(target) == "":
        return os.path.join(OUTPUT_DIR, target)
    return target


# %%
# ============================================================================
# ここを書き換えて、VSCode の Run Cell でインタラクティブに閲覧できる
# ============================================================================
TARGET = "snapshot_0000.npz"   # 例: "snapshot_0059.npz" や 59(フレーム番号)でも可


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mpm_beach.py の snapshot npz を閲覧する")
    parser.add_argument("target", nargs="?", default=TARGET,
                         help="npzファイルのパス、ファイル名、またはフレーム番号")
    parser.add_argument("--csv", action="store_true", help="CSVへ変換して保存する")
    parser.add_argument("--plot-only", action="store_true", help="図の保存のみ行い、ウィンドウ表示はしない")
    parser.add_argument("--no-plot", action="store_true", help="図を作らずテキスト表示のみ行う")
    parser.add_argument("--list", action="store_true", help="output/ 内の全npzファイル一覧を表示して終了")
    args = parser.parse_args()

    if args.list:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "snapshot_*.npz")))
        print(f"[view_snapshot] {len(files)} 個のスナップショットが見つかりました:")
        for f in files:
            print(" ", os.path.basename(f))
        sys.exit(0)

    path = resolve_target(args.target)
    if not os.path.exists(path):
        print(f"[view_snapshot] ファイルが見つかりません: {path}")
        print("  --list オプションで output/ 内の一覧を確認できます。")
        sys.exit(1)

    summarize(path)

    if args.csv:
        export_csv(path)

    if not args.no_plot:
        default_png = os.path.splitext(path)[0] + "_view.png"
        plot_snapshot(path, out_path=default_png, show=not args.plot_only)
