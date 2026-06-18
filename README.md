# 歯列 抜歯ツール

STLファイルをビューアで開き、歯をクリックして対話的に抜歯（削除）できるツールです。

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# STLファイルを指定して起動
python dental_extraction.py <stl_file>

# サンプルデータで起動（引数なし）
python dental_extraction.py
```

## 操作方法

| 操作 | 動作 |
|------|------|
| 左クリック | 歯を選択（赤くハイライト） |
| 右クリック | 選択中の歯を抜歯（削除） |
| `U` キー | 最後の抜歯を元に戻す（Undo） |
| `S` キー | 抜歯後のSTLを保存 |
| `Q` / `ESC` | 終了 |

保存ファイルは元のSTLと同じフォルダに `<元のファイル名>_extracted.stl` として出力されます。
