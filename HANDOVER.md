# 引継ぎドキュメント

whisp-carier v0.1.0 の開発経緯・設計判断・既知の問題をまとめたドキュメントです。  
フォークして改善する方への参考情報です。

---

## 開発経緯

- Faster-Whisper-XXL Pro（Purfview作）はRTX 5090対応だが有料（£50）かつソース非公開
- 同等機能をOSSで再実装し、RTX 5090 (sm_120) でネイティブ動作するCLIを作成
- Amatsukaze（rigaya改造版）からの呼び出しを主なユースケースとして開発

## アーキテクチャ

```
whisp_carier.py     ← エントリポイント。argparse、transcribe、出力処理
audio_filter.py     ← ffmpegラッパー + audio-separator（MDX/Roformer）
vad.py              ← 外部VADバックエンド（pyannote, auditok, webrtc）
whisp-carier.bat    ← PythonをフルパスでCLI起動するラッパー
```

### 依存関係の要点

| パッケージ | 役割 | 備考 |
|-----------|------|------|
| torch 2.8.0+cu128 | GPU推論 | sm_120対応の最低バージョン |
| faster-whisper 1.2.1 | 文字起こし本体 | 内蔵VADがsilero v6 |
| audio-separator 0.44.5 | ボーカル抽出 | MelBand-Roformer対応 |
| stable-ts | --realign機能 | 実験的、不安定 |

---

## 設計判断と理由

### condition_on_previous_text = False（デフォルト）

Whisperはデフォルトで前セグメントのテキストを次のプロンプトに渡す。  
これにより文脈が繋がるメリットがある反面、一度ハルシネーションに入るとループが止まらない。

テストの結果：
- `True` → 76セグメント（ハルシネーションで後半が壊れる）
- `False` → 200セグメント（正常動作）

精度よりも安定性を優先し `False` をデフォルトにした。

### hallucination_silence_threshold = 0（無効）

faster-whisperの内蔵機能。無音後のセグメントをハルシネーションとして破棄する。  
しかしテストでは正常なセグメントも巻き添えで消えた（212→104）。  
アニメ素材は台詞間の無音が長いため、この機能は使えない。

### 音声フィルター = 実験的

`--ff_loudnorm`, `--ff_lowhighpass`, `--ff_vocal_extract` はいずれも  
内蔵VADとの相性問題でセグメント数が激減する。  
フィルターなしが最も安定する。

### カスタムVAD = 実質不要

faster-whisper 1.2+ は内蔵で silero v6 (ONNX) を使用。  
`--vad_method` で silero_v4/v5/v6 を指定しても全て同じ内蔵モデルが動く。  
外部VAD（pyannote等）は `clip_timestamps` 経由で渡すが、  
Whisperの連続処理と独立するためハルシネーションが起きやすい。

---

## 既知の問題

### 1. ハルシネーション（部分的に残る）

`condition_on_previous_text=False` で大幅に改善したが完全ではない。  
BGM区間や無音区間で「ぬぬぬ...」「父上に命じられたか?」等の繰り返しが出ることがある。

現在の対策：
- 完全一致の重複検出（MAX_DUPES=2でスキップ）
- 単一文字ループ検出（10文字超、文字種2以下）

改善案：
- Purfviewの `--ignore_dupe_prompt` 相当の実装
- `--hallucinations_list` による既知パターンリスト
- セグメント内フレーズ繰り返し検出（実装したが誤爆が多く無効化した）

### 2. 音声フィルターが実用レベルに達していない

- `--ff_lowhighpass` → 後半の音声が消える
- `--ff_vocal_extract mb-roformer` → アニメ声まで除去される
- 併用すると0セグメントになる

原因：フィルター後の音声がVADの期待する特性と合わない。  
改善案：フィルター後にVADパラメータを自動調整する仕組み。

### 3. --realign が不安定

stable-tsの `align()` がSRTのテキストを音声全体に引き伸ばすバグがある。  
現在は `transcribe()` を再実行する形に変更したが、二重処理で無駄。  
実用性は低い。

### 4. PyInstallerでのexe化ができない

numpy 2.4.4 と PyInstaller の相性問題。  
`numpy._core._exceptions` モジュールが見つからないエラー。  
numpy 1.xに下げれば動く可能性があるが未検証。  
現状は `.bat` ラッパー方式で対応。

---

## テスト結果サマリ

| 素材 | オプション | セグメント | 時間 | 備考 |
|------|-----------|-----------|------|------|
| サイバーフォーミュラ（25分） | `--beam_size 10 --best_of 10` | 513 | 465s | 会話多い |
| ワールドイズダンシング（30分） | `--beam_size 10 --best_of 10` | 200 | 55s | ダンスシーン多い |
| テスト音声（7秒TTS） | デフォルト | 3 | 1.4s | 正常動作確認 |

---

## 今後やるなら

1. **ハルシネーション対策の改善** — 既知パターンリスト、n-gram重複検出
2. **音声フィルターとVADの連携改善** — フィルター後のVAD閾値自動調整
3. **exe化** — numpy 1.x系で再挑戦、またはNuitka等の別ビルドツール
4. **faster-whisper本体へのパッチ** — 内蔵VADモデルの差し替え機構
5. **バッチ推論の活用** — `--batched` で速度向上（品質トレードオフの検証）

---

## 開発環境再構築手順

```powershell
# Python 3.11 インストール（python.org から）
# CUDA Toolkit 12.8 インストール（developer.nvidia.com から）

pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

以上。
