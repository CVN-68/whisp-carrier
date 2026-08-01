# whisp-carier

RTX 5090 (Blackwell / sm_120) にネイティブ対応した faster-whisper CLIツール。  
Faster-Whisper-XXL の代替として、全てオープンソースのコンポーネントで構築。

## 経緯

Faster-Whisper-XXL Pro は RTX 5090 対応版が有料（£50寄付）かつソース非公開だったため、
同等機能を持つオープンソース版を自作しました。

## 特徴

- **RTX 5090 ネイティブ動作** — torch 2.8.0+cu128、互換モード落ちなし
- **Amatsukaze 対応** — faster-whisper-xxl.exe と同じCLIインターフェース
- **内蔵 VAD** — silero v6（faster-whisper 1.2+ に同梱）
- **ハルシネーション抑制** — ループ検出 + 無音区間の自動カット
- **ボーカル抽出** — MelBand-Roformer（最高品質）/ MDX Kim_Vocal_2
- **音声フィルター** — loudnorm、バンドパス、RNNoise、FFTノイズ除去、ノイズゲート等
- **出力形式** — SRT, VTT, JSON, TXT, TSV, LRC

## 必要環境

- Windows 10/11 (x64)
- Python 3.11
- NVIDIA RTX GPU + CUDA 12.8 以上のドライバ
- CUDA Toolkit 12.8
- ffmpeg（PATHに通っていること）

## インストール

```powershell
# PyTorch (CUDA 12.8版)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# その他の依存関係
pip install -r requirements.txt
```

## 使い方

```powershell
# 基本（日本語、large-v3モデル）
python whisp_carier.py "動画.mp4" -m large-v3 -l ja -o source -pp

# 高品質設定
python whisp_carier.py "動画.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp

# BGM/環境音除去付き（重いがノイズの多い素材に有効）
python whisp_carier.py "動画.mp4" -m large-v3 --ff_vocal_extract mb-roformer -o source -pp

# 音量正規化 + バンドパスフィルター
python whisp_carier.py "動画.mp4" -m large-v3 --ff_loudnorm --ff_lowhighpass -o source -pp
```

> **注意:** `--ff_loudnorm --ff_lowhighpass` と `--ff_vocal_extract mb-roformer` は併用しないでください。  
> バンドパス後にボーカル抽出すると音声が消えます。どちらか一方を選択してください。

## Amatsukaze との連携

1. Amatsukaze の「基本設定」で Whisper パスに以下を指定：
   ```
   C:\Users\<ユーザー名>\whisp-carier\whisp-carier.bat
   ```
2. 追加オプション例：

   | 用途 | 追加オプション |
   |------|---------------|
   | 基本（推奨） | `--beam_size 10 --best_of 10` |
   | 言語固定（自動検出が不安定な場合） | `--language ja --beam_size 10 --best_of 10` |
   | 日本語アニメ向け | `--language ja --standard_asia --beam_size 10 --best_of 10` |

   ※ モデルや出力形式は Amatsukaze が自動で指定するため、手動で `-m` や `-f` を追加する必要はありません。

   **音声フィルター系オプションについて:**  
   `--ff_loudnorm`, `--ff_lowhighpass`, `--ff_vocal_extract` 等の音声フィルターは実験的機能です。  
   テストの結果、フィルターを適用すると内蔵VADとの相性問題でセグメントが大幅に減少するケースが確認されています。  
   通常はフィルターなし（`--beam_size 10 --best_of 10` のみ）が最も安定します。

## ファイル構成

```
whisp_carier.py     — メインCLI（文字起こし・出力処理）
audio_filter.py     — ffmpegフィルター + ボーカル抽出（MDX/Roformer）
vad.py              — カスタムVADバックエンド（pyannote, auditok, webrtc）
whisp-carier.bat    — Amatsukaze互換ランチャー
requirements.txt    — Python依存関係
```

## 注意事項

- `--ff_vocal_extract mb-roformer` は初回実行時にモデル（約900MB）をダウンロードします
- アニメ等で声とBGMが近い周波数の場合、ボーカル抽出が声まで消すことがあります
- レース実況などノイズの多い素材では `--ff_loudnorm --ff_lowhighpass` の方が安定します
- `--realign` は現在実験的機能です（stable-tsとの連携に不安定な部分あり）

## テスト環境

| 項目 | スペック |
|------|---------|
| CPU | AMD Ryzen 9 5900XT |
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| RAM | 32GB |
| OS | Windows 11 (26200) |
| Python | 3.11.9 |
| PyTorch | 2.8.0+cu128 |
| CUDA Driver | 591.44 / CUDA 13.1 |
| CUDA Toolkit | 12.8.61 |
| faster-whisper | 1.2.1 |
| 連携ソフト | Amatsukaze 1.0.5.5 and 1.0.8.5 (rigaya改造版) |




## ステータス

**Active — 評価段階。** フィードバック歓迎。  
Amatsukaze との連携テストを進めています。

## ベースとなったプロジェクト

本プロジェクトは以下のオープンソースプロジェクトを基に構築されています。

| プロジェクト | 役割 | リンク |
|-------------|------|--------|
| OpenAI Whisper | 音声認識モデル本体 | https://github.com/openai/whisper |
| faster-whisper | CTranslate2ベースのWhisper推論エンジン | https://github.com/SYSTRAN/faster-whisper |
| PyTorch | GPU計算基盤（CUDA 12.8 / sm_120 対応） | https://pytorch.org/ |
| silero-vad | 音声区間検出モデル | https://github.com/snakers4/silero-vad |
| audio-separator | ボーカル抽出（MDX / Mel-Band-Roformer） | https://github.com/karaokenerds/python-audio-separator |
| stable-ts | タイムスタンプ再調整（実験的） | https://github.com/jianfch/stable-ts |
| CTranslate2 | 高速Transformer推論 | https://github.com/OpenNMT/CTranslate2 |
| ffmpeg | 音声前処理・フィルタリング | https://ffmpeg.org/ |

開発のきっかけ：[Faster-Whisper-XXL](https://github.com/Purfview/whisper-standalone-win)（Purfview作）のRTX 5090対応版が有料かつソース非公開だったため、同等機能をオープンソースのみで再実装したもの。

## オプション一覧

### モデル・デバイス

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--model` | `-m` | `large-v3` | Whisperモデル名。`large-v3`, `large-v3-turbo`, `medium`, `small` 等 |
| `--model_dir` | | なし | モデル保存先ディレクトリ。未指定時は自動ダウンロード |
| `--device` | `-d` | `auto` | 使用デバイス。`cuda` / `cpu` / `auto` |
| `--compute_type` | `-ct` | `default` | 量子化タイプ。`float16`, `int8`, `int8_float16`, `float32` 等 |

### 出力

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--output_dir` | `-o` | `default` | 出力先。`source`=入力と同じ場所、`.`=カレント |
| `--output_format` | `-f` | `srt` | 出力形式（複数可）。`srt`, `vtt`, `json`, `txt`, `tsv`, `lrc`, `all` |
| `--postfix` | | なし | 検出言語をファイル名末尾に付加 |

### 言語・タスク

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--language` | `-l` | なし（自動検出） | 言語コード。`ja`, `en`, `zh` 等 |
| `--task` | | `transcribe` | `transcribe`=書き起こし、`translate`=英語翻訳 |

### 品質・精度

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--beam_size` | `-bs` | `5` | ビームサーチ幅。推奨: `10` |
| `--best_of` | `-bo` | `5` | 候補数。推奨: `10` |
| `--patience` | `-p` | `2.0` | ビームサーチの忍耐度 |
| `--temperature` | | `0` | サンプリング温度。0=確定的 |
| `--repetition_penalty` | | `1.0` | 繰り返しペナルティ（1.0以上で抑制） |
| `--no_repeat_ngram_size` | | `0` | n-gram繰り返し禁止サイズ。0=無効 |
| `--condition_on_previous_text` | `-condition` | `False` | 前の出力を次のプロンプトに使う |
| `--initial_prompt` | `-prompt` | `auto` | 初回プロンプト。`None`で無効 |
| `--word_timestamps` | `-wt` | `True` | 単語レベルタイムスタンプ |

### ハルシネーション対策

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--hallucination_silence_threshold` | `-hst` | `0`（無効） | 無音後のセグメントをハルシネーションとして破棄する閾値（秒） |
| `--no_speech_threshold` | | `0.6` | no_speech確率がこの値以上なら無音と判断 |
| `--compression_ratio_threshold` | | `2.4` | 圧縮比がこの値以上ならデコード失敗 |
| `--logprob_threshold` | | `-1.0` | 平均対数確率がこの値以下ならデコード失敗 |

### VAD（音声区間検出）

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--vad_filter` | `-vad` | `True` | VAD有効化 |
| `--vad_threshold` | | `0.45` | 音声判定閾値（低いほど感度高） |
| `--vad_min_speech_duration_ms` | | `250` | 最短音声チャンク（ミリ秒） |
| `--vad_max_speech_duration_s` | | なし | 最長音声チャンク（秒） |
| `--vad_min_silence_duration_ms` | | `3000` | 無音待機時間（ミリ秒） |
| `--vad_speech_pad_ms` | | `900` | 前後パディング（ミリ秒） |

> **VADモデルについて:**  
> faster-whisper 1.2+ は内蔵で silero v6 (ONNX) を使用しています。  
> `--vad_method` で `silero_v4_fw` / `silero_v5_fw` / `silero_v6` / `silero_v6_fw` を指定しても、
> 実際に動くのはすべて同じ内蔵 silero v6 です。  
> 本当に別のVADエンジンを使いたい場合は `pyannote_v3`, `auditok`, `webrtc` を指定してください。
> ただし外部VADはWhisperと独立して動くため、ハルシネーションが起きやすくなる傾向があります。
> 通常は内蔵VAD（デフォルト）のまま使うことを推奨します。

### 音声フィルター

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--ff_track` | `1` | 音声トラック番号（1〜6） |
| `--ff_fc` | なし | フロントセンターのみ抽出 |
| `--ff_lc` | なし | 左チャンネルのみ抽出 |
| `--ff_invert` | なし | 左ch極性反転+モノラルミックス |
| `--ff_rnndn_sh` | なし | RNNoise SHモデル（攻撃的） |
| `--ff_rnndn_xiph` | なし | RNNoise Xiphモデル（穏やか） |
| `--ff_fftdn` | `0` | FFTノイズ除去（0=無効、12=標準、最大97） |
| `--ff_gate` | なし | ノイズゲート |
| `--ff_speechnorm` | なし | 音声部分を極端に増幅 |
| `--ff_loudnorm` | なし | EBU R128ラウドネス正規化 |
| `--ff_lowhighpass` | なし | 50Hz-7800Hzバンドパス |
| `--ff_tempo` | `1.0` | テンポ調整（0.5〜2.0） |
| `--ff_silence_suppress` | `0 3.0` | 無音抑制（閾値dB, 最小長秒） |

### ボーカル抽出

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--ff_vocal_extract` | なし | `mdx_kim2` または `mb-roformer` |
| `--mdx_chunk` | `15` | MDXチャンクサイズ（秒） |
| `--voc_device` | `cuda` | 抽出用デバイス |

### バッチ処理

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--batch_recursive` | `-br` | なし | ディレクトリ再帰処理 |
| `--batched` | | なし | バッチ推論（2〜8倍速、品質微低下） |
| `--batch_size` | | `8` | バッチ並列数 |
| `--skip` | | なし | 出力済みならスキップ |
| `--check_files` | | なし | 入力ファイル事前チェック |
| `--print_progress` | `-pp` | なし | 進捗表示（セグメントごとにリアルタイム） |

> **進捗表示について:**  
> `-pp` を指定しなくても、10セグメントごとに自動で進捗ログが出力されます。  
> Amatsukazeのログ画面で処理状況を確認できます。
| `--beep_off` | | なし | 完了ビープ無効 |

### 字幕フォーマット

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--standard` | なし | 標準プリセット（幅42字、2行、文単位） |
| `--standard_asia` | なし | アジア言語プリセット（幅16字、2行） |
| `--sentence` | なし | 文単位分割 |
| `--max_line_width` | `1000` | 1行最大文字数 |
| `--max_line_count` | `1` | 最大行数 |
| `--max_gap` | `3.0` | 文末判定の空白閾値（秒） |

### その他

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--realign` | | なし | タイムスタンプ再調整（実験的） |
| `--realign_device` | | なし | realign用デバイス |
| `--version` | | | バージョン表示 |
| `--checkcuda` | `-cc` | | CUDAデバイス数表示 |
| `--verbose` | `-v` | `False` | デバッグ出力 |

---

## Amatsukaze 記載例

追加オプション欄にそのままコピペしてください。

```
# 推奨（シンプル高品質）
--beam_size 10 --best_of 10

# 言語固定 + 高品質
--language ja --beam_size 10 --best_of 10

# 日本語アニメ向け
--language ja --standard_asia --beam_size 10 --best_of 10

# 字幕を標準フォーマットに
--standard --beam_size 10 --best_of 10

# VAD感度UP（小声の拾い漏れ対策）
--vad_threshold 0.3 --beam_size 10 --best_of 10

# ハルシネーション対策強化
--hallucination_silence_threshold 2 --repetition_penalty 1.2 --beam_size 10 --best_of 10
```

### 実験的オプション（非推奨）

以下は内蔵VADとの相性問題でセグメントが減少する場合があります。

```
# ノイズ多い素材（実験的）
--ff_loudnorm --ff_lowhighpass --beam_size 10 --best_of 10

# BGMが激しい素材（実験的・音声が消える可能性あり）
--ff_vocal_extract mb-roformer --beam_size 10 --best_of 10
```

## ライセンス

MIT
