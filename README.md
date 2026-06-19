# Bad Apple!! — 東方Project Dot Matrix Player

https://nahsuoij.github.io

Bad Apple!! 影絵アニメをドットマトリクスで再現したウェブプレイヤー。東方Projectキャラクターのカラーリング付き。

## Features

- **Dot Matrix Rendering** — 48×36 デフォルト、スライダーで精細度調整可（24×18 〜 96×72）
- **Touhou Character Colors** — キャラクター領域に東方公式カラー適用
  - 博麗霊夢 Reimu: PINK
  - 霧雨魔理沙 Marisa: YELLOW
  - 十六夜咲夜 Sakuya: SILVER
  - レミリア Remilia: RED
  - チルノ Cirno: ICE BLUE
  - パチュリー Patchouli: PURPLE
- **Karaoke Lyrics** — 進度バー上にリアルタイム歌詞表示、逐字ハイライト
- **Fullscreen Immersive** — クリックでフルスクリーン、コントロール自動非表示
- **Background Image Upload** — 任意の画像を背景に設定可（スケールフィット）
- **Danmaku** — 背景に東方テキスト弾幕を流す
- **Auto-play** — ロード後自動再生

## Files

| File | Description |
|------|-------------|
| `index.html` | メインページ（全機能統合） |
| `frames.js` | 2629フレーム、48×36、RLE圧縮 |
| `badapple.m4a` | 音声ファイル |
| `pako.min.js` | zlib 展開ライブラリ |

## Controls

| Key | Action |
|-----|--------|
| Space | 再生 / 一時停止 |
| M | ミュート切替 |
| F | フルスクリーン |
| ↑ / ↓ | 精細度調整 |

## License

東方Project © ZUN / 上海アリス幻樂団
