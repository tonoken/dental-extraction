"""
dental_extraction.py
--------------------
歯列STLファイルをクリックで抜歯するツール

使い方:
    python dental_extraction.py <stl_file>
    python dental_extraction.py               # サンプルデータで起動

操作:
    左クリック       : 歯を選択（ハイライト）
    右クリック       : 選択中の歯を抜歯（削除）
    U キー          : 最後の抜歯を元に戻す（Undo）
    S キー          : 抜歯後のSTLを保存
    Q / ESC         : 終了
"""

import sys
import copy
import numpy as np
import trimesh
from pathlib import Path

# vedo
try:
    from vedo import Mesh, Plotter, Text2D
    from vedo.colors import color_map
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


def segment_teeth(mesh: trimesh.Trimesh, n_teeth: int = 32) -> list[np.ndarray]:
    """
    簡易セグメンテーション：
    X軸（左右）・Y軸（前後）でクラスタリングして歯を分割する。
    本格的な歯列モデルにはカスタム調整が必要。
    """
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=50)
    if len(components) < 2:
        # 1ピースの場合はX座標でスライス分割
        return _slice_by_x(mesh, n_teeth)

    # 複数コンポーネントがあればそのまま歯として扱う
    print(f"検出されたコンポーネント数: {len(components)}")
    return [np.array(c) for c in components]


def _slice_by_x(mesh: trimesh.Trimesh, n: int) -> list[np.ndarray]:
    """X座標でn等分してフェイスをグループ化"""
    face_cx = mesh.vertices[mesh.faces].mean(axis=1)[:, 0]
    bins = np.linspace(face_cx.min(), face_cx.max(), n + 1)
    groups = []
    for i in range(n):
        mask = (face_cx >= bins[i]) & (face_cx < bins[i + 1])
        idx = np.where(mask)[0]
        if len(idx) > 0:
            groups.append(idx)
    return groups


def faces_to_vedo_mesh(mesh: trimesh.Trimesh, face_ids: np.ndarray, color="ivory") -> Mesh:
    """指定フェイス群からvedo Meshを生成"""
    sub = mesh.submesh([face_ids], append=True)
    vm = Mesh([sub.vertices, sub.faces])
    vm.color(color).lighting("plastic")
    return vm


# ──────────────────────────────────────────────
# メインアプリ
# ──────────────────────────────────────────────

class DentalExtractor:
    NORMAL_COLOR   = "ivory"
    SELECTED_COLOR = "tomato"
    REMOVED_COLOR  = "gray"   # Undo表示用（実際には非表示）

    def __init__(self, stl_path: str):
        self.stl_path = Path(stl_path)
        self.mesh = load_stl(stl_path)
        print(f"STL読み込み完了: {self.mesh.faces.shape[0]} 面, {self.mesh.vertices.shape[0]} 頂点")

        self.tooth_faces: list[np.ndarray] = segment_teeth(self.mesh)
        print(f"歯のセグメント数: {len(self.tooth_faces)}")

        self.removed: list[int] = []          # 削除済みインデックス
        self.selected_idx: int | None = None  # 選択中インデックス
        self.history: list[int] = []          # Undo用スタック

        self._build_plotter()

    # ── ビルド ──────────────────────────

    def _build_plotter(self):
        self.plt = Plotter(title="歯列 抜歯ツール", bg="black", size=(1100, 750))

        # 各歯をMeshとして登録
        self.vedo_teeth: list[Mesh | None] = []
        for i, fids in enumerate(self.tooth_faces):
            vm = faces_to_vedo_mesh(self.mesh, fids, self.NORMAL_COLOR)
            vm.name = str(i)
            self.vedo_teeth.append(vm)

        # UI テキスト
        self.info_text = Text2D(
            "左クリック: 選択  右クリック: 抜歯  U: Undo  S: 保存  Q: 終了",
            pos="bottom-center", s=0.6, c="white", bg="k", alpha=0.5
        )
        self.status_text = Text2D(
            "", pos="top-left", s=0.65, c="yellow"
        )

        actors = [t for t in self.vedo_teeth if t] + [self.info_text, self.status_text]
        self.plt.add(actors)
        self.plt.add_callback("LeftButtonPress",  self._on_left_click)
        self.plt.add_callback("RightButtonPress", self._on_right_click)
        self.plt.add_callback("KeyPress",         self._on_key)

    # ── コールバック ──────────────────────

    def _on_left_click(self, event):
        """左クリック：歯を選択"""
        if event.actor is None:
            return
        idx = self._actor_to_idx(event.actor)
        if idx is None or idx in self.removed:
            return

        # 前の選択を解除
        if self.selected_idx is not None and self.selected_idx not in self.removed:
            self.vedo_teeth[self.selected_idx].color(self.NORMAL_COLOR)

        self.selected_idx = idx
        self.vedo_teeth[idx].color(self.SELECTED_COLOR)
        self._set_status(f"歯 #{idx+1} を選択中（右クリックで抜歯）")
        self.plt.render()

    def _on_right_click(self, event):
        """右クリック：選択中の歯を抜歯"""
        if self.selected_idx is None:
            self._set_status("先に歯を左クリックで選択してください")
            self.plt.render()
            return

        idx = self.selected_idx
        if idx in self.removed:
            return

        # 抜歯処理
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

    # ── Undo ──────────────────────────────

    def _undo(self):
        if not self.history:
            self._set_status("元に戻す履歴がありません")
            self.plt.render()
            return

        idx = self.history.pop()
        self.removed.remove(idx)

        # Meshを再生成して追加
        vm = faces_to_vedo_mesh(self.mesh, self.tooth_faces[idx], self.NORMAL_COLOR)
        vm.name = str(idx)
        self.vedo_teeth[idx] = vm
        self.plt.add(vm)
        self._set_status(f"歯 #{idx+1} を元に戻しました")
        self.plt.render()

    # ── 保存 ──────────────────────────────

    def _save(self):
        remaining = [
            i for i in range(len(self.tooth_faces))
            if i not in self.removed
        ]
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

    # ── ヘルパー ──────────────────────────

    def _actor_to_idx(self, actor) -> int | None:
        try:
            return int(actor.name)
        except (ValueError, AttributeError):
            return None

    def _set_status(self, msg: str):
        self.status_text.text(msg)

    # ── 起動 ──────────────────────────────

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
    if path is None:
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "sample_teeth.stl")
    """
    テスト用：円弧状に並んだ円柱（歯に見立て）のSTLを生成
    """
    import trimesh.creation as tc

    arches = []
    n = 14  # 片顎 14本
    for jaw in range(2):  # 上顎・下顎
        y_offset = 0 if jaw == 0 else 35
        for i in range(n):
            angle = np.pi * i / (n - 1)  # 0〜180°
            x = 30 * np.cos(angle)
            y = 15 * np.sin(angle) + y_offset
            height = np.random.uniform(8, 14)
            radius = np.random.uniform(2.5, 4)
            cyl = tc.cylinder(radius=radius, height=height, sections=16)
            mat = np.eye(4)
            mat[0, 3] = x
            mat[1, 3] = y
            mat[2, 3] = 0
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
    if len(sys.argv) < 2:
        print("STLファイルが指定されていないため、サンプルデータを使用します。")
        stl_path = create_sample_dental_stl()
    else:
        stl_path = sys.argv[1]

    extractor = DentalExtractor(stl_path)
    extractor.run()
