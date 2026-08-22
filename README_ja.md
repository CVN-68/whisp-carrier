# whisp-carrier

RTX 5090 (Blackwell / sm_120) にネイティブ対応した faster-whisper CLIツール。  
Faster-Whisper-XXL の代替として、全てオープンソースのコンポーネントで構築。

## 経緯

Faster-Whisper-XXL Pro は RTX 5090 対応版が有料（£50寄付）かつソース非公開だったため、
同等機能を持つオープンソース版を自作しました。

## 特徴

- **RTX 5090 ネイティブ動作** — torch 2.8.0+cu128、互換モード落ちなし
- **Amatsukaze 対応** — faster-whisper-xxl.exe と同じCLIインターフェース
- **モデルエイリアス** — 日本語アニメ向けの `-m anime-whisper` 等。transformers 形式の
  Whisper ファインチューンは初回実行時に CTranslate2 へ自動変換
- **内蔵 VAD** — silero v6（faster-whisper 1.2+ に同梱）
- **ハルシネーション抑制** — ループ検出 + 無音区間の自動カット
- **ボーカル抽出** — MelBand-Roformer（最高品質）/ MDX Kim_Vocal_2
- **音声フィルター** — loudnorm、バンドパス、RNNoise、FFTノイズ除去、ノイズゲート等
- **字幕整形** — 文単位分割、行幅・行数指定、禁則処理、単語タイムスタンプによる再タイミング
- **設定ファイル** — YAMLでプロファイルを切り替え。Amatsukaze側の設定を触らずに変更できる
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
python whisp_carrier.py "動画.mp4" -m large-v3 -l ja -o source -pp

# 高品質設定
python whisp_carrier.py "動画.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp

# BGM/環境音除去付き（重いがノイズの多い素材に有効）
python whisp_carrier.py "動画.mp4" -m large-v3 --ff_vocal_extract mb-roformer -o source -pp

# 音量正規化 + バンドパスフィルター
python whisp_carrier.py "動画.mp4" -m large-v3 --ff_loudnorm --ff_lowhighpass -o source -pp
```

> **注意:** `--ff_loudnorm --ff_lowhighpass` と `--ff_vocal_extract mb-roformer` は併用しないでください。  
> バンドパス後にボーカル抽出すると音声が消えます。どちらか一方を選択してください。

## モデル

`--model` にはビルトインのモデルサイズ、エイリアス、ローカルディレクトリ、
Hugging Face のリポジトリIDを指定できます。エイリアス一覧は `--list_models` で表示されます。

```powershell
# 日本語アニメ・ノベルゲーム系のセリフ向け
python whisp_carrier.py "動画.mp4" -m anime-whisper --standard_asia -o source -pp

# transformers 形式で公開されている任意の Whisper ファインチューン
python whisp_carrier.py "動画.mp4" -m efwkjn/whisper-ja-anime-v0.3 -l ja -o source
```

| エイリアス | 実体 | ライセンス | 備考 |
|-----------|------|-----------|------|
| `anime-whisper` | [litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper) | MIT | kotoba-whisper-v2.0 を約5,300時間のアニメ調演技セリフでファインチューンしたモデル。学習外のノベルゲーム音声で CER 13.0（whisper-large-v3 は 16.5）とモデルカードが報告 |
| `kotoba-v2` | [kotoba-tech/kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | Apache-2.0 | 日本語汎用。large-v3 の蒸留モデルで anime-whisper のベース。CTranslate2 形式で公開されているため変換不要 |

エイリアスには「そのモデルが望むオプション」も紐づいています。
`anime-whisper` なら `--language ja` と `--no_repeat_ngram_size 5` を自動で選び、
初期プロンプトが指定されていれば警告します（このモデルはプロンプトを渡すと
ハルシネーションループに入ります）。

CLIやYAMLで明示指定した値は常に優先され、どう解決されたかは `[MODEL]` 行に出ます。

```
[MODEL] alias 'anime-whisper' -> litagin/anime-whisper (MIT)
[MODEL] using converted model: ...\_models\ct2-litagin-anime-whisper-float16
[MODEL]   language = 'ja'  (default for anime-whisper)
[MODEL]   no_repeat_ngram_size = 5  (default for anime-whisper)
```

明示指定した場合はこうなります。

```
[MODEL]   language = 'en' kept as given (anime-whisper would use 'ja')
[MODEL]   WARNING: initial_prompt='anime': anime-whisper degrades badly with an initial prompt ...
```

### 変換について

faster-whisper は CTranslate2 形式のモデルしか読めないため、transformers 形式の
モデルは初回実行時に `_models/ct2-<名前>-<量子化>/`（`--model_dir` 指定時はその配下）へ
変換されます。変換には `transformers` が必要で、重みは通常の Hugging Face キャッシュ
経由でダウンロードされます。2回目以降はキャッシュを読むだけです。
`--compute_type` を変えた後などにやり直したい場合は `--reconvert` を使います。

変換時に2点だけ自動で面倒を見ています。どちらも間違えるとエラーにならず静かに壊れるためです。

- **tokenizer.json** — 元リポジトリに無い場合は生成します。無いまま読ませると
  faster-whisper が whisper-tiny のトークナイザに黙ってフォールバックし、
  エラーも出さずに誤った文字列を出力します。
- **alignment heads** — 実際のデコーダ層数と突き合わせて検証します。蒸留モデルは
  教師モデルのヘッド一覧をそのまま引き継いでいるため、`--word_timestamps` が
  存在しないデコーダ層を参照し、Pythonのトレースバックも出さずにプロセスが即死します
  （anime-whisper が実際にこれに該当します。HANDOVER.md 参照）。

### 変換は exe 版では行えません

**exe 版とスクリプト版の使い分け**は次のとおりです。

| | exe 版 | スクリプト版 |
|---|---|---|
| 通常の文字起こし（`large-v3` 等） | ○ | ○ |
| CTranslate2 形式のモデル | ○ | ○ |
| 変換済みモデルの読み込み | ○ | ○ |
| transformers 形式モデルの変換（`-m anime-whisper`） | × | ○ |

exe には変換に必要な `transformers` を同梱していません。変換は初回だけの作業で、
そのために配布物へ数百MBと未検証の実行経路を持ち込む必要がないためです。
**anime-whisper を使いたい場合はスクリプト版で一度変換し、
できたディレクトリを exe に渡してください。**

```powershell
# 1回だけ。スクリプト版で変換する
python whisp_carrier.py test_speech.wav -m anime-whisper -o . -f srt

# 以降は exe でも使える
whisp-carrier.exe "動画.mp4" -m _models\ct2-litagin-anime-whisper-float16
```

exe に `-m anime-whisper` を直接渡した場合は、上と同じ手順を案内して終了します
（終了コード 2）。ビルトインサイズや CTranslate2 形式のモデルには影響しません。

## 終了コード

バッチ処理や Amatsukaze から呼ぶ場合に判定できるようにしてあります。

| コード | 意味 |
|-------|------|
| `0` | 全ファイル成功 |
| `1` | **1つ以上のファイルが失敗**、または入力が見つからない |
| `2` | 起動前のエラー（設定ファイル、モデル解決、VAD の初期化） |

複数ファイルを渡した場合、1本が失敗しても残りは処理を続けます。ただし
**最後に失敗した本数とファイル名を stderr に出し、終了コードは 1 になります。**
全件成功したときだけ `[whisp-carrier] All done.` を表示します。

音声フィルター（`--ff_*`）が失敗した場合、そのファイルは文字起こしされずに
失敗として数えられます。フィルターを指定したのに適用できなかった状態で
無加工の結果を返すことはしません。

## 設定ファイル（プロファイル）

Amatsukaze の「追加オプション」欄を毎回書き換えるのが面倒なので、
YAMLファイルで設定を切り替えられるようにしてあります。
**呼び出し側の設定は一切変更しません。**

### 使い方

`whisp-carrier.yaml.example` を `whisp-carrier.yaml` にリネームして、
`whisp_carrier.py`（exe版なら exe）と同じフォルダに置くだけです。
ファイル名が `.example` のままだと読み込まれません。

```yaml
override: true
active_profile: anime

# 共通設定（全プロファイルのベース）
beam_size: 10
best_of: 10

profiles:
  anime:
    language: ja
    standard_asia: true
  race:
    language: ja
    ff_loudnorm: true
    ff_lowhighpass: true
```

キー名は `--help` のオプション名から先頭の `--` を外したものです。

| 書き方 | 対応するCLI |
|--------|------------|
| `beam_size: 10` | `--beam_size 10` |
| `standard_asia: true` | `--standard_asia` |
| `output_format: [srt, vtt]` | `-f srt vtt` |
| `language: null` | 未指定（自動検出） |

精度検証では `active_profile` の1行を書き換えて比較します。

### override — CLIとどちらを優先するか

| 設定 | 挙動 |
|------|------|
| `override: true` | **設定ファイルを優先。** Amatsukazeが渡すオプションを上書きする |
| `override: false`（既定） | **CLI指定を優先。** 設定ファイルはCLIが指定しなかった項目だけを埋める |

どの値がどこから来たかは実行時に `[CONFIG]` 行で確認できます。

```
[CONFIG] C:\Users\...\whisp-carrier.yaml | profile=anime | override=on
[CONFIG]   beam_size = 10
[CONFIG]   language = 'ja'  (overrides CLI 'en')
[CONFIG]   max_line_width = 16
```

`override: false` のときにCLIが勝った項目も明示されます。

```
[CONFIG]   beam_size: kept the CLI value, ignoring config 10 (enable override to reverse this)
```

### 関連オプション

| オプション | 説明 |
|-----------|------|
| `--config PATH` | 別の場所の設定ファイルを使う |
| `--no_config` | 設定ファイルを一時的に無効化 |
| `--profile NAME` | `active_profile` を一時的に上書き |
| `--config_override` | ファイルを編集せずに override を有効化 |

> **注意:** 設定ファイルに書いたキー名が間違っている場合はエラーで停止します。  
> 黙って無視すると精度比較そのものが無効になるためです。


## Amatsukaze との連携

1. Amatsukaze の「基本設定」で Whisper パスに以下を指定：
   ```
   C:\Users\<ユーザー名>\whisp-carrier\whisp-carrier.bat
   ```
2. 追加オプション例：

   | 用途 | 追加オプション |
   |------|---------------|
   | 基本（推奨） | `--beam_size 10 --best_of 10` |
   | 言語固定（自動検出が不安定な場合） | `--language ja --beam_size 10 --best_of 10` |
   | 日本語アニメ向け | `--language ja --standard_asia --beam_size 10 --best_of 10` |

   ※ モデルや出力形式は Amatsukaze が自動で指定するため、手動で `-m` や `-f` を追加する必要はありません。

   **モデルを anime-whisper に変えたい場合:**
   Amatsukaze は `-m` を自分で渡してくるため、追加オプション欄にもう一度 `-m` を
   書くのは避けてください。設定ファイル側で `model: anime-whisper` と `override: true`
   を指定するのが確実です。初回のみ変換が走るので、Amatsukaze から呼ぶ前に
   コマンドラインで一度実行して変換を済ませておくのが安全です。

   ```powershell
   python whisp_carrier.py test_speech.wav -m anime-whisper -o . -f srt
   ```

   ※ この欄を毎回書き換えたくない場合は[設定ファイル](#設定ファイルプロファイル)を使ってください。
   `override: true` にすれば、この欄を空にしたままYAML側だけで設定を切り替えられます。
   精度検証で設定を何度も差し替えるときはこちらが楽です。

   **音声フィルター系オプションについて:**  
   `--ff_loudnorm`, `--ff_lowhighpass`, `--ff_vocal_extract` 等の音声フィルターは実験的機能です。  
   テストの結果、フィルターを適用すると内蔵VADとの相性問題でセグメントが大幅に減少するケースが確認されています。  
   通常はフィルターなし（`--beam_size 10 --best_of 10` のみ）が最も安定します。

## ファイル構成

```
whisp_carrier.py            — メインCLI（文字起こし・出力処理）
audio_filter.py             — ffmpegフィルター + ボーカル抽出（MDX/Roformer）
vad.py                      — カスタムVADバックエンド（pyannote, auditok, webrtc）
subtitle_format.py          — 字幕整形（文分割・折り返し・再タイミング）
whisp_models.py             — モデルエイリアスとCTranslate2変換
whisp_config.py             — YAML設定ファイル / プロファイル
whisp-carrier.bat           — Amatsukaze互換ランチャー
whisp-carrier.yaml.example  — 設定ファイルのサンプル
requirements.txt            — Python依存関係
```

## 注意事項

- `-m anime-whisper` は初回実行時にモデル（約3GB）をダウンロードして CTranslate2 に変換します。
  変換後は `_models/` 配下（float16 で約1.5GB）から読み込むため、2回目以降は数秒で起動します
- anime-whisper は日本語専用モデルです。英語音声を入れるとカタカナで書き起こされます
- anime-whisper の書き起こしは半角の `! ?` と半角数字を使い、文末の `。` はほぼ付きません。
  `--sentence` / `--standard_asia` の文末判定は半角記号と `…` にも対応しているため、そのまま動きます
- **`--ff_vocal_extract` は exe 版では使えません。** audio-separator を同梱していないためで、
  指定するとその旨を表示して止まります。スクリプト版を使ってください。
  同梱を試した際、パッケージ自体は入るものの実行時に scipy の拡張モジュールが読めず
  失敗し、しかも「scipy を再インストールしろ」という無関係な案内が出る状態だったため、
  同梱せず理由を説明する形にしています。なお本プロジェクトが公開している精度の数値は
  すべて `--ff_*` を使わずに測ったものなので、この制約は精度に影響しません
- `--ff_vocal_extract mb-roformer` は初回実行時にモデル（約900MB）をダウンロードします（スクリプト版）
- アニメ等で声とBGMが近い周波数の場合、ボーカル抽出が声まで消すことがあります
- レース実況などノイズの多い素材では `--ff_loudnorm --ff_lowhighpass` の方が安定します
- `--realign` は現在実験的機能です（stable-tsとの連携に不安定な部分あり）。
  モデルを2つ目としてロードするためVRAMと時間が二重にかかり、
  書き出したSRTを上書きするため字幕整形の行組みが失われます
- `--batched` は動作しますが、VADチャンクのまとめ方が変わるためセグメントが粗くなります
  （テスト音声で 3 → 1）。字幕用途では分割が荒くなる可能性があるため、速度と品質を
  測ってから採用してください

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
| Anime Whisper | 日本語アニメ調セリフ向けモデル（`-m anime-whisper`） | https://huggingface.co/litagin/anime-whisper |
| Kotoba-Whisper | 日本語蒸留Whisper。Anime Whisperのベース | https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 |
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
| `--model` | `-m` | `large-v3` | モデル名 / エイリアス / ローカルパス / HFリポジトリID。`large-v3`, `large-v3-turbo`, `anime-whisper` 等 |
| `--model_dir` | | なし | モデル保存先ディレクトリ。未指定時は自動ダウンロード。変換モデルの置き場所も兼ねる |
| `--list_models` | | | エイリアス一覧を表示して終了 |
| `--reconvert` | | なし | 変換済みキャッシュがあっても再変換する |
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
| `--initial_prompt` | `-prompt` | なし（プロンプトを渡さない） | 初回プロンプト。`None` / `auto` を指定した場合も無効扱い |
| `--word_timestamps` | `-wt` | `True` | 単語レベルタイムスタンプ |

### ハルシネーション対策

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--loop_filter` | | `True` | 同じ文字・語句の繰り返しだけになったセグメントを破棄する |
| `--hallucination_silence_threshold` | `-hst` | `0`（無効） | 無音後のセグメントをハルシネーションとして破棄する閾値（秒） |
| `--no_speech_threshold` | | `0.6` | no_speech確率がこの値以上なら無音と判断 |
| `--compression_ratio_threshold` | | `2.4` | 圧縮比がこの値以上ならデコード失敗 |
| `--logprob_threshold` | | `-1.0` | 平均対数確率がこの値以下ならデコード失敗 |

> **`--loop_filter` について:**
> 叫び声やBGM区間でWhisperが「ああああ…」「よーし、よーし、…」を延々と
> 出力することがあり、これが実測で精度に最も響いていました。判定は
> 「12文字以上で文字種2以下」「同じ文字が8連続以上」「1〜6文字の単位が
> 4回以上、かつ繰り返し部分が12文字以上」の3条件で、該当したセグメントを
> 丸ごと落とします。破棄したものは `[LOOP]` 行に理由付きで表示されます。
>
> 実測では9本のTVアニメ録画で全体CERが 24.3% → 22.0%、
> 30秒ブロック単位では 26.0% → 23.4% に改善し、取りこぼし（カバレッジ）は
> 変わりませんでした。「きゃあああああ」のような短い悲鳴や
> 「そそそそんなわけ」のような言い直しは残ります。
> 長い叫びをそのまま字幕に載せたい場合は `--loop_filter false` で無効化できます。

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

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
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

> **字幕整形の挙動:**
> - 日本語・中国語等は文字単位、英語等は単語単位で折り返します。
> - 行頭に来てはいけない文字（`、。）」…` 拗音促音など）が次行の先頭になる場合、
>   禁則処理として指定幅を1文字超えることを許して前行に残します。
>   `--max_line_width` は厳密な上限ではなく目安です。
> - 1つのセグメントが `--max_line_count` を超える場合は複数の字幕に分割し、
>   タイムスタンプは単語タイムスタンプから再計算します。
>   このため `--word_timestamps true`（既定）を推奨します。
>   無効の場合は文字数比で時間を按分した近似値になります。
> - `--max_gap` を超える無音は、句読点が無くても文の切れ目として扱います。
> - **`--realign` と併用しないでください。** `--realign` は書き出したSRTを
>   stable-ts の出力で上書きするため、整形した行組みが失われます。

### 設定ファイル

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | 自動検出 | YAML設定ファイルのパス。未指定時は同フォルダの `whisp-carrier.yaml` |
| `--no_config` | なし | 設定ファイルを無視 |
| `--profile` | なし | 適用するプロファイル名（`active_profile` を上書き） |
| `--config_override` | なし | 設定ファイルをCLIより優先（`override: true` と同等） |

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
