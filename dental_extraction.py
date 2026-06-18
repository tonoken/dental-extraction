"""
dental_extraction.py
--------------------
歯列STLファイルをクリックで抜歯するツール

使い方:
    python dental_extraction.py <stl_file> [--teeth N]
    python dental_extraction.py               # サンプルデータで起動

操作:
    左クリック       : 歯を選択（ハイライト）
    右クリック       : 選択中の歯を抜歯（削除）
    U キー          : 最後の抜歯を元に戻す（Undo）
    S キー          : 抜歯後のSTLを保存
    Q / ESC         : 終了
"""

import sys
import numpy as np
import trimesh
from collections import deque
from pathlib import Path

try:
    from vedo import Mesh, Plotter, Text2D
except ImportError:
    print("vedo が必要です: pip install vedo")
    sys.exit(1)


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def load_stl(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("STLファイルの読み込みに失敗しました")
    return mesh


def faces_to_vedo_mesh(mesh: trimesh.Trimesh, face_ids: np.ndarray, color="ivory") -> Mesh:
    sub = mesh.submesh([face_ids], append=True)
    vm = Mesh([sub.vertices, sub.faces])
    vm.color(color).lighting("plastic")
    return vm


def is_single_shell(mesh: trimesh.Trimesh) -> bool:
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=50)
    return len(components) < 2


# ──────────────────────────────────────────────
# Geodesic Voronoi セグメンテーション（1シェル用）
# ──────────────────────────────────────────────

def segment_geodesic_voronoi(mesh: trimesh.Trimesh, n_teeth: int = 14) -> list[np.ndarray]:
    """
    1シェル歯列スキャン用セグメンテーション。
    1. XY重心をKMeansでn_teeth本にクラスタリング
    2. 各クラスタ内でZ最小のフェイスをシード（歯先はZ最小側）
    3. マルチソースBFSでGeodesic Voronoi分割
    """
    from sklearn.cluster import KMeans

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    face_z = face_centroids[:, 2]
    adj_pairs = mesh.face_adjacency

    # Step 1: XY KMeans
    print(f"KMeansで {n_teeth} 本に分類中...")
    km = KMeans(n_clusters=n_teeth, random_state=42, n_init=10)
    km.fit(face_centroids[:, :2])

    # Step 2: 各クラスタのZ最小点をシード（歯先はZ最小側）
    seeds = []
    for i in range(n_teeth):
        cluster = np.where(km.labels_ == i)[0]
        if len(cluster) == 0:
            continue
        seeds.append(cluster[face_z[cluster].argmin()])
    seeds = np.array(seeds)
    print(f"シード: {len(seeds)} 個（Z最低点=歯先）")

    # Step 3: 隣接リスト構築 → マルチソースBFS
    print("Geodesic Voronoi分割中...")
    adj_list: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in adj_pairs:
        adj_list[a].append(b)
        adj_list[b].append(a)

    labels = -np.ones(len(mesh.faces), dtype=np.int32)
    q: deque = deque()
    for i, seed in enumerate(seeds):
        labels[seed] = i
        q.append((seed, i))

    while q:
        face, label = q.popleft()
        for nb in adj_list[face]:
            if labels[nb] == -1:
                labels[nb] = label
                q.append((nb, label))

    n_labels = len(seeds)
    segments = [np.where(labels == i)[0] for i in range(n_labels)]
    sizes = [len(s) for s in segments]
    print(f"セグメント完了: {n_labels} 本  (最小 {min(sizes):,} 面 / 最大 {max(sizes):,} 面)")
    return segments


# ──────────────────────────────────────────────
# マルチシェル用セグメンテーション
# ──────────────────────────────────────────────

def segment_multi_shell(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=50)
    print(f"検出されたコンポーネント数: {len(components)}")
    return [np.array(c) for c in components]


# ──────────────────────────────────────────────
# 共通 UI クラス
# ──────────────────────────────────────────────

class DentalExtractor:
    NORMAL_COLOR   = "ivory"
    SELECTED_COLOR = "tomato"

    def __init__(self, stl_path: str, tooth_faces: list[np.ndarray],
                 mesh: trimesh.Trimesh):
        self.stl_path = Path(stl_path)
        self.mesh = mesh
        self.tooth_faces = tooth_faces

        self.removed: list[int] = []
        self.selected_idx: int | None = None
        self.history: list[int] = []
        self._build_plotter()

    def _build_plotter(self):
        self.plt = Plotter(title="歯列 抜歯ツール", bg="black", size=(1100, 750))

        self.vedo_teeth: list[Mesh | None] = []
        for i, fids in enumerate(self.tooth_faces):
            vm = faces_to_vedo_mesh(self.mesh, fids, self.NORMAL_COLOR)
            vm.name = str(i)
            self.vedo_teeth.append(vm)

        self.info_text = Text2D(
            "左クリック: 選択  右クリック: 抜歯  U: Undo  S: 保存  Q: 終了",
            pos="bottom-center", s=0.6, c="white", bg="k", alpha=0.5
        )
        self.status_text = Text2D("", pos="top-left", s=0.65, c="yellow")

        self.plt.add([t for t in self.vedo_teeth if t] + [self.info_text, self.status_text])
        self.plt.add_callback("LeftButtonPress",  self._on_left_click)
        self.plt.add_callback("RightButtonPress", self._on_right_click)
        self.plt.add_callback("KeyPress",         self._on_key)

    def _on_left_click(self, event):
        if event.actor is None:
            return
        try:
            idx = int(event.actor.name)
        except (ValueError, AttributeError):
            return
        if idx in self.removed:
            return
        if self.selected_idx is not None and self.selected_idx not in self.removed:
            self.vedo_teeth[self.selected_idx].color(self.NORMAL_COLOR)
        self.selected_idx = idx
        self.vedo_teeth[idx].color(self.SELECTED_COLOR)
        self._set_status(f"歯 #{idx+1} を選択中（右クリックで抜歯）")
        self.plt.render()

    def _on_right_click(self, event):
        if self.selected_idx is None:
            self._set_status("先に歯を左クリックで選択してください")
            self.plt.render()
            return
        idx = self.selected_idx
        if idx in self.removed:
            return
        self.plt.remove(self.vedo_teeth[idx])
        self.removed.append(idx)
        self.history.append(idx)
        self.selected_idx = None
        self._set_status(f"歯 #{idx+1} を抜歯しました（U: 元に戻す）")
        self.plt.render()

    def _on_key(self, event):
        key = event.keypress.lower()
        if key == "u":
            self._undo()
        elif key == "s":
            self._save()
        elif key in ("q", "escape"):
            self.plt.close()

    def _undo(self):
        if not self.history:
            self._set_status("元に戻す履歴がありません")
            self.plt.render()
            return
        idx = self.history.pop()
        self.removed.remove(idx)
        vm = faces_to_vedo_mesh(self.mesh, self.tooth_faces[idx], self.NORMAL_COLOR)
        vm.name = str(idx)
        self.vedo_teeth[idx] = vm
        self.plt.add(vm)
        self._set_status(f"歯 #{idx+1} を元に戻しました")
        self.plt.render()

    def _save(self):
        remaining = [i for i in range(len(self.tooth_faces)) if i not in self.removed]
        if not remaining:
            self._set_status("保存できる歯がありません")
            self.plt.render()
            return
        all_face_ids = np.concatenate([self.tooth_faces[i] for i in remaining])
        result_mesh = self.mesh.submesh([all_face_ids], append=True)
        out_path = self.stl_path.parent / (self.stl_path.stem + "_extracted.stl")
        result_mesh.export(str(out_path))
        self._set_status(f"保存完了: {out_path.name}  (抜歯数: {len(self.removed)})")
        self.plt.render()
        print(f"\n保存: {out_path}")

    def _set_status(self, msg: str):
        self.status_text.text(msg)

    def run(self):
        print("\n[操作方法]")
        print("  左クリック  : 歯を選択（赤くハイライト）")
        print("  右クリック  : 選択中の歯を抜歯")
        print("  U キー      : 最後の抜歯を元に戻す")
        print("  S キー      : 結果をSTLに保存")
        print("  Q / ESC     : 終了\n")
        self.plt.show(zoom="tightest", interactive=True)


# ──────────────────────────────────────────────
# サンプルSTL生成（テスト用）
# ──────────────────────────────────────────────

def create_sample_dental_stl(path: str = None):
    """テスト用：円弧状に並んだ円柱（歯に見立て）のSTLを生成"""
    if path is None:
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "sample_teeth.stl")
    import trimesh.creation as tc

    arches = []
    n = 14
    for jaw in range(2):
        y_offset = 0 if jaw == 0 else 35
        for i in range(n):
            angle = np.pi * i / (n - 1)
            x = 30 * np.cos(angle)
            y = 15 * np.sin(angle) + y_offset
            height = np.random.uniform(8, 14)
            radius = np.random.uniform(2.5, 4)
            cyl = tc.cylinder(radius=radius, height=height, sections=16)
            mat = np.eye(4)
            mat[0, 3] = x
            mat[1, 3] = y
            cyl.apply_transform(mat)
            arches.append(cyl)

    combined = trimesh.util.concatenate(arches)
    combined.export(path)
    print(f"サンプルSTL生成: {path}")
    return path


# ──────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="歯列STL 抜歯ツール")
    parser.add_argument("stl", nargs="?", help="STLファイルパス（省略時はサンプルデータ）")
    parser.add_argument("--teeth", type=int, default=14,
                        help="歯の本数（1シェル時に使用。デフォルト: 14）")
    args = parser.parse_args()

    if args.stl is None:
        print("STLファイルが指定されていないため、サンプルデータを使用します。")
        stl_path = create_sample_dental_stl()
        mesh = load_stl(stl_path)
        tooth_faces = segment_multi_shell(mesh)
    else:
        stl_path = args.stl
        mesh = load_stl(stl_path)
        print(f"STL読み込み完了: {mesh.faces.shape[0]:,} 面, {mesh.vertices.shape[0]:,} 頂点")
        if is_single_shell(mesh):
            tooth_faces = segment_geodesic_voronoi(mesh, args.teeth)
        else:
            tooth_faces = segment_multi_shell(mesh)

    extractor = DentalExtractor(stl_path, tooth_faces, mesh)
    extractor.run()
