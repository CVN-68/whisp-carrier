# whisp-carrier

**日本語** | [English](README_en.md)

RTX 5090 (Blackwell / sm_120) にネイティブ対応した faster-whisper CLIツール。  
Faster-Whisper-XXL の代替として、全てオープンソースのコンポーネントで構築。

**このドキュメントは配布している exe 版のマニュアルです。**
Python から動かす開発版（無保証）の手順は [HANDOVER.md](HANDOVER.md) にあります。
exe でできないことは[exe版とスクリプト版の違い](#exe版とスクリプト版の違い)にまとめました。

## 経緯

Faster-Whisper-XXL Pro は RTX 5090 対応版が有料（£50寄付）かつソース非公開だったため、
同等機能を持つオープンソース版を自作しました。

## 特徴

- **RTX 5090 ネイティブ動作** — torch 2.8.0+cu128、互換モード落ちなし
- **Amatsukaze 対応** — faster-whisper-xxl.exe と同じCLIインターフェース
- **TEN VAD が既定**（Apache-2.0）— silero より台詞の取りこぼしが少なく、
  TVアニメ録画15本すべてで silero 以上でした（24分もの9本で全文CER 19.3% → 16.1%、
  子供向け4本で 31.1% → 21.9%）。silero 系も `--vad_method` で選べます
- **ハルシネーションループの抑制** — 同じ文字・語句の繰り返しだけになったセグメントを
  破棄します（既定で有効、`--loop_filter false` で無効化）
- **30秒超セグメントの修復** — Whisperの窓は30秒なので、それを超えるキューは
  復元処理の産物です。常時検査して単語タイムスタンプで分割します
- **モデルエイリアス** — 日本語アニメ向けの `-m anime-whisper` 等。transformers 形式の
  Whisper ファインチューンは初回実行時に CTranslate2 へ自動変換（スクリプト版のみ）
- **字幕整形** — 文単位分割、行幅・行数指定、禁則処理、単語タイムスタンプによる再タイミング
- **音声フィルター** — loudnorm、バンドパス、RNNoise、FFTノイズ除去、ノイズゲート等（実験的）
- **ボーカル抽出** — MelBand-Roformer（最高品質）/ MDX Kim_Vocal_2（スクリプト版のみ）
- **設定ファイル** — YAMLでプロファイルを切り替え。Amatsukaze側の設定を触らずに変更できる
- **出力形式** — SRT, VTT, JSON, TXT, TSV, LRC

## 導入

- Windows 10/11 (x64)
- NVIDIA RTX GPU + CUDA 12.8 以上のドライバ
- **Python も CUDA Toolkit も ffmpeg も不要です**（ffmpeg は同梱、LGPL版）

アーカイブを展開して、`whisp-carrier.exe` を好きな場所に置くだけです。
Amatsukaze から呼ぶ場合は[Amatsukaze との連携](#amatsukaze-との連携)へ。

同梱物は exe のほかに `LICENSE` / `LICENSE.ffmpeg.txt` /
`LICENSE.ten-vad.*.txt` / `THIRD-PARTY-NOTICES.md` /
`whisp-carrier.yaml.example` です。設定ファイルは必須ではありません
（[設定ファイル](#設定ファイルプロファイル)を使うときだけ `.example` を外してリネームします）。

## 使い方

```powershell
# 基本（日本語、large-v3モデル）
whisp-carrier.exe "動画.mp4" -m large-v3 -l ja -o source -pp

# 日本語の字幕に整形（16字2行）
whisp-carrier.exe "動画.mp4" -m large-v3 -l ja --standard_asia -o source

# ビーム幅を広げる（※実測では既定の 5 に対して 9本中8本で同等または悪化。
#   「上位設定」ではなく、単にそういう指定ができるという例）
whisp-carrier.exe "動画.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp

# 音量正規化 + バンドパスフィルター（実験的。既定は何も足さない）
whisp-carrier.exe "動画.mp4" -m large-v3 --ff_loudnorm --ff_lowhighpass -o source -pp
```

## モデル

`--model` にはビルトインのモデルサイズ、エイリアス、ローカルディレクトリ、
Hugging Face のリポジトリIDを指定できます。エイリアス一覧は `--list_models` で表示されます。

```powershell
# ビルトインのサイズ
whisp-carrier.exe "動画.mp4" -m large-v3 -l ja -o source

# 変換済みモデルのディレクトリ（下記のとおり変換自体は exe ではできません）
whisp-carrier.exe "動画.mp4" -m _models\ct2-litagin-anime-whisper-float16 -o source
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

### exe 版はモデル変換ができません

faster-whisper は CTranslate2 形式のモデルしか読めないため、transformers 形式の
モデル（`anime-whisper` や HF の Whisper ファインチューン）は一度変換が必要です。
**変換に必要な `transformers` は exe 版に同梱していません。**
初回だけの作業のために、配布物へ数百MBと未検証の実行経路を持ち込まない判断です。

**exe に `-m anime-whisper` を直接渡した場合は、手順を案内して終了します**
（終了コード 2）。ビルトインサイズ（`large-v3` 等）と CTranslate2 形式のモデル、
そして**変換済みディレクトリの読み込みは通常どおり動きます。**

変換の手順は [HANDOVER.md](HANDOVER.md) にあります（開発版で1回走らせて、
できた `_models/ct2-*` を exe に渡す形です）。

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

1. Amatsukaze の「基本設定」で Whisper パスに exe を指定：
   ```
   C:\Users\<ユーザー名>\whisp-carrier\whisp-carrier.exe
   ```
2. whisper-option（追加オプション）欄は **空のままを推奨します。**

   | 用途 | whisper-option 欄 |
   |------|------------------|
   | **基本（推奨）** | **空**（既定が測定した設定そのものです） |
   | 日本語の字幕に整形したい | `--standard_asia` |
   | 言語を固定したい（分割で出る短い断片の誤判定対策） | `--language ja` |

   **空でよい理由。** 既定が TEN VAD（閾値 0.75）・`beam_size 5` / `best_of 5`・
   `vad_segment_mode clip`・ループ抑制ありで、**これが本 README に載せた数値を
   出した条件そのもの**です。何も足さないほうが再現します。

   > **`--beam_size 10 --best_of 10` は書かないでください。**
   > 以前このドキュメントは「基本（推奨）」としてこれを載せていましたが、
   > 24分アニメ9本を同一素材で測ったところ **9本中8本で beam 5 と同等または悪化**しました
   > （全文CER 22.0% → 22.3%、時間は +2.8% しか変わらない）。
   > 既に欄に入っている場合は消してください。

   ※ モデルや出力形式は Amatsukaze が自動で指定するため、手動で `-m` や `-f` を追加する必要はありません。

   **モデルの選び方（whisper-model 欄）。**
   Amatsukaze の候補は `自動` / `未指定` / `small` / `medium` / `large-v1` /
   `large-v2` / `large-v3` / `large-v3-turbo` です。

   | 選択 | 渡されるもの | 備考 |
   |------|------------|------|
   | **`large-v3`** | `-m large-v3` | **推奨。本 README の数値はこれで測っています** |
   | `未指定` | `-m` を渡さない | 当実装の既定が `large-v3` なので結果は同じ。**設定ファイルの `model:` を効かせたいときはこちら**（`-m` と衝突しないので `override` が不要） |
   | **`自動`** | **`-m large-v3-turbo`** | **注意。** 下記参照 |
   | `large-v3-turbo` | `-m large-v3-turbo` | 速いが**未測定**。下記参照 |
   | `medium` 以下 | 同名を渡す | 日本語の字幕用途には非推奨 |

   > **`自動` を選ぶと large-v3-turbo で走ります**（実機で確認）。
   > turbo は large-v3 の encoder はそのままで、**decoder を 32層 → 4層に削ったモデル**です。
   > 速い代わりに、**本 README の数値は large-v3 のものなので当てになりません。**
   >
   > 当実装の強みはループ抑制・30秒超セグメントの修復・タイムスタンプの安定性で、
   > **どれも decoder 側の挙動**です。さらに `condition_on_previous_text` を
   > false にしているため文脈で立て直す余地もありません。turbo は未測定なので、
   > **`large-v3` を明示することを推奨します。**
   >
   > turbo や distil 系を渡した場合は実行時に `[MODEL] NOTE:` で注意が出ます。
   > 欄を触らずに打ち消したい場合は、設定ファイルに `model: large-v3` と
   > `override: true` を書いてください。

   **モデルを anime-whisper に変えたい場合:**
   whisper-model を `未指定` にして、設定ファイルに変換済みディレクトリのパスを
   `model:` として書くのがいちばん素直です（`large-v3` を選んだままなら
   `override: true` が必要）。**exe は変換ができないので、先に開発版で
   一度変換しておく必要があります**（[該当節](#exe-版はモデル変換ができません)）。
   **ただし anime-whisper は9本すべてで large-v3 に負けた**ので
   （全文CER 41.8% 対 24.3%、旧指標）、字幕用途では推奨しません。

   ※ この欄を毎回書き換えたくない場合は[設定ファイル](#設定ファイルプロファイル)を使ってください。
   `override: true` にすれば、この欄を空にしたままYAML側だけで設定を切り替えられます。
   精度検証で設定を何度も差し替えるときはこちらが楽です。
   **なお設定ファイルは必須ではありません。** 上の表のとおり欄を空にすれば既定で
   測定条件になるので、整形やプロファイル切り替えが要らなければ置かなくて構いません。

   **音声フィルター系オプションについて:**  
   `--ff_loudnorm` や `--ff_lowhighpass` 等の音声フィルターは実験的機能です。  
   フィルターを通すとセグメント数が大きく減るケースが確認されていて、
   実用レベルに達していません（原因の切り分けは HANDOVER.md の「既知の問題 2」）。  
   **本 README に載せた精度はすべてフィルターなしで測ったものです。通常は何も足さないでください。**

## 対象範囲（OP/ED・歌唱シーンは保証外）

**本編の台詞を字幕にすることを目標にしています。OP/ED や挿入歌・歌唱シーンは
対象範囲外です。ただし「出さない」ことは保証しません。** 実際、条件によっては
歌詞らしいテキストを拾って字幕に出します。

これは Whisper 系の文字起こしに共通する挙動で、本プロジェクト固有の欠陥では
ありません。同じ素材の同じ歌唱区間で出力文字数を数えた実測値です。

| 実装 | 歌唱区間に出した文字数 |
|------|----------------------|
| whisp-carrier（現在の既定 TEN VAD） | 2,274字 |
| Faster-Whisper-XXL r245.4 | 630字 |
| whisp-carrier（旧既定 silero） | 54字 |

旧既定の silero がほとんど拾わなかったのは性能ではなく**検出漏れ**で、
その漏れは本編の台詞も一緒に落としていました。TEN VAD に替えて台詞の取りこぼしが
減った副作用として、歌唱区間の出力も増えています。

**歌唱区間を落とすフィルタは入れていません。** 試しに落として採点すると精度が
悪化したためです（全文CER 15.5% → 16.1%）。歌に字幕を付ける番組もあり、
放送局の運用が一定でないため、機械的に落とすと正しい字幕まで消えます。

歌詞が不要な場合は、出力後に該当時間帯のキューを手で削るのが確実です。
判断材料としてログの `[VAD]` 行が使えます。

同じ理由で、**「ご視聴ありがとうございました」のような定型句が無音区間に
出ることがあります。** Whisper が学習データの高頻度句を無音に対して吐く挙動で、
これも Faster-Whisper-XXL に同様に見られます（同一素材9本で XXL 11件、本実装 43件）。
落とすフィルタは意図的に実装していません。**0.3〜0.6pt の精度向上のために、
作中のテレビ放送や配信を締める台詞を誤って消す機構を入れる取引が悪い**と
判断しました。詳細は HANDOVER.md の測定結果 #20 にあります。

## exe版とスクリプト版の違い

**exe 版が主経路で、サポート対象もこちらだけです。** 通常の文字起こし・VAD・
ループ抑制・30秒超の修復・字幕整形・設定ファイルはすべて exe で動きます。
**スクリプト版（開発版）は人柱版で動作を保証しません。**
必要になるのは次の3つを使うときだけです。手順は [HANDOVER.md](HANDOVER.md) にあります。

| 機能 | exe 版 | スクリプト版 | 同梱していない理由 |
|------|:------:|:-----------:|------------------|
| 通常の文字起こし（`large-v3` 等） | ○ | ○ | — |
| CTranslate2 形式のモデル・変換済みモデルの読み込み | ○ | ○ | — |
| TEN VAD / silero 系・ループ抑制・字幕整形・設定ファイル | ○ | ○ | — |
| Amatsukaze 連携 | ○（検証済み） | △（`whisp-carrier.bat` 経由。**現行版は未検証**） | — |
| **transformers 形式モデルの変換**（`-m anime-whisper` 等） | × | ○ | `transformers` 未同梱。変換は初回だけの作業なので、数百MBと未検証の実行経路を配布物に持ち込まない判断 |
| **`--ff_vocal_extract`**（ボーカル抽出） | × | ○ | `audio-separator` 未同梱。同梱自体は成功するが実行時に scipy の拡張モジュールが読めず、しかも「scipy を再インストールせよ」という無関係な案内が出る |
| **`--realign`**（タイムスタンプ再調整） | × | ○ | `stable-ts` 未同梱。そもそも推奨設定（`standard_asia`）では必ずスキップされる |
| `--vad_method pyannote_v3` / `pyannote_onnx_v3` | × | × | pytorch-lightning と speechbrain を引き込むうえ、実測で内蔵 silero より劣る |

**exe で上を指定した場合は、理由と回避策を表示して止まります**（黙って別の動作に
はなりません）。**そして精度には影響しません。** 本 README の数値はすべて
`--ff_*` なし・`large-v3` で測ったもので、この3つはどれもその経路に出てきません。

## 注意事項

- **初回実行時にモデルをダウンロードします**（`large-v3` で約3GB）。2回目以降は
  キャッシュから読むので数秒で起動します。`--model_dir` で置き場所を変えられます
- 変換済みの anime-whisper を渡す場合、**日本語専用モデル**なので英語音声は
  カタカナで書き起こされます。また半角の `! ?` と半角数字を使い、文末の `。` は
  ほぼ付きません（`--sentence` / `--standard_asia` の文末判定は半角記号と `…` にも
  対応しているため、整形はそのまま動きます）
- レース実況などノイズの多い素材では `--ff_loudnorm --ff_lowhighpass` が比較的安定します。
  ただし実験的機能で、**本 README の数値はフィルターなしのものです**
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


## 測定した精度

**TVアニメ録画15本を ARIB字幕（放送局が付けた字幕）を正解として採点した結果です。**
既定設定同士の比較で、音声フィルターは使っていません。歌唱区間は採点から除外しています。

| 素材群 | 本数 | 参照文字数 | whisp-carrier（既定） | Faster-Whisper-XXL r245.4 |
|--------|------|-----------|---------------------|--------------------------|
| 24分のTVアニメ | 9 | 35,036 | **16.1%** | 20.5% |
| 子供向け番組 | 4 | 16,704 | **21.9%** | 33.5% |
| 一挙放送（5時間22分） | 1 | 37,987 | **16.1%** | 未測定 |
| 一挙放送（4時間52分） | 1 | 50,255 | **14.1%** | 未測定 |

数値は全文CER（文字誤り率、低いほど良い）です。**単一の数字で語れる精度ではありません。**
番組の内容で大きく変わり、ファイル別では 8.7%〜30.5% に分布します。

| 層 | CER | 例 |
|----|-----|-----|
| 会話劇・職場もの | 8.7〜14.0% | 公女殿下の家庭教師 8.7% / LIAR GAME 9.7% / 桃源暗鬼 11.9% |
| アクション・日常 | 16〜17% | クレバテス 16.4% |
| 子供向け番組 | 23〜25% | 名探偵プリキュア 23.0% / おねがいアイプリ 23.6% / ぷにる 25.3% |
| 発話率の低い作品 | 30.5% | 死亡遊戯で飯を食う。#09（発話が尺の23%） |

**誤差の内訳は「間違い」より「取りこぼし」です。** 9本で参照の 86.6% を拾い、
出した文字の 90.0% が参照に一致しました。悪い側のファイルは精度は落ちておらず、
拾えていない量が多いだけです。

**構造面の比較。** 字幕として使うときに実務で効くのはこちらです。

| 項目 | whisp-carrier | Faster-Whisper-XXL r245.4 |
|------|--------------|--------------------------|
| 30秒を超えるセグメント（24分もの9本） | **0件** | 9件 / 合計602秒 |
| 長すぎるのに文字が少ないセグメント | 1件 | 15件 |
| ハルシネーションループの残存文字数（9本） | **35字** | 53字 |
| RTX 5090 で既定のまま動くか | **動く**（float16） | 動かない（`-ct float32` が必須で約2倍遅い） |
| 24分1本の処理時間 | 70〜100秒 | 175秒 |
| 5時間22分の処理時間 | 1214秒（実時間の6.3%） | 未測定 |

長尺での劣化はありません。一挙放送は5時間22分が 16.1%、4時間52分が **14.1%** で、
後者は全測定中の最良値でした。

### 比較条件（推奨設定同士になっています）

**Faster-Whisper-XXL 側は既定設定です。そしてそれが開発者の推奨です。**
[本家の README](https://github.com/Purfview/whisper-standalone-win/blob/main/README.md) には
「既定値は映画の書き起こし向けに調整済み」という趣旨の記述があり、
追加設定の推奨リストは公表されていません（使用例に出るのは `--sentence` /
`--standard` / `--batch_recursive` のみ）。※内容はライセンス配慮のため要約しています

**そして XXL の既定は「素」ではありません。** ハルシネーション対策が既定で3つ入っています。

| 設定 | Faster-Whisper-XXL r245.4 の既定 | whisp-carrier の既定 |
|------|--------------------------------|---------------------|
| VAD | 有効（silero） | 有効（**TEN VAD**） |
| `vad_min_silence_duration_ms` / `vad_speech_pad_ms` | 3000 / 900 | 3000 / 900（本家に合わせています） |
| `beam_size` / `best_of` | 5 / 5 | 5 / 5 |
| `patience` | 2.0 | 2.0 |
| `condition_on_previous_text` | **True** | **False**（実測で選択。下記） |
| 既知ハルシネーション一覧 | **既定で有効** | 相当機能なし |
| large-v3 用の擬似VAD閾値オフセット | **既定で有効** | 相当機能なし |
| プロンプト再投入・重複プロンプト抑制 | **既定で有効** | 相当機能なし |
| 音声フィルター（`--ff_*`） | すべて無効 | 使いません |
| 字幕整形プリセット | 無効 | 推奨は `standard_asia: true` |
| ループ抑制 | — | **既定で有効**（本家に相当機能なし） |

実際に渡したオプションはこれだけです。

```powershell
# Faster-Whisper-XXL r245.4
faster-whisper-xxl.exe <wav> -m large-v3 -ct float32 -f json --language ja --beep_off

# whisp-carrier（既定のまま。--no_config で設定ファイルの影響を除去）
python whisp_carrier.py <wav> -m large-v3 -f json --no_config --beep_off
```

**XXL に与えた差分は2つで、どちらも XXL 不利にはなっていません。**

- `-ct float32` — RTX 5090 では float16 が cuBLAS で落ちるため不可避です。同じ重みなので
  **精度は float32 のほうが有利側**です（代わりに約2倍遅くなります）
- `--language ja` — 言語を明示しています。なお当実装では言語を明示しても
  出力が1セグメントも変わりませんでした（自動判定が元々 ja 92〜100%）

**字幕整形を揃える必要はありません。** 整形が足すのは改行とキュー分割で、
採点時に**両陣営の出力から**空白を完全除去するため、全文CERは動きません。
つまり「両者を推奨の整形で揃える」を実行しても同じ数字になります。
一方 30秒ブロック単位のCERは整形で動くため、そちらは両者とも整形なしで揃えてあります。

**`condition_on_previous_text` を False にしている理由。** True を実測したところ、
発話率の低い作品でカバレッジが 59.7% → 21.3% に落ち、推論時間が5.9倍になり、
30分尺の出力が18分16秒で止まりました。本家と同じ挙動を「借りてくる」ことはできず、
ここは意図的に分岐しています。

**測っていない条件。** XXL 側に音声フィルター（`--ff_rnndn_sh` 等）や `--batched` を
当てた条件は測っていません。どちらも既定で無効かつ開発者の推奨にも入っておらず、
当実装はフィルターを使わない方針で決着しているため、
XXL だけフィルター有効で回すと非対称な比較になります。

測定環境・素材・指標の定義は [HANDOVER.md](HANDOVER.md) にあります。

## ステータス

**Active — 評価段階。** フィードバック歓迎。

Amatsukaze との連携テストは exe 版で完了しています。**測定素材と同じ回を実運用で流して、
VAD区間数・発話秒数・セグメント数・抑制したループまで一致**したので、
上の数値はこの経路でそのまま出てくるものです。**サポート対象は exe 版のみ**で、
スクリプト版（開発版）は[人柱版](#exe版とスクリプト版の違い)として置いています。

## ベースとなったプロジェクト

本プロジェクトは以下のオープンソースプロジェクトを基に構築されています。
**同梱物の完全な一覧とライセンス条文の所在は [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)** にあります
（配布アーカイブにも同梱）。

| プロジェクト | 役割 | ライセンス | リンク |
|-------------|------|-----------|--------|
| OpenAI Whisper | 音声認識モデル本体 | MIT | https://github.com/openai/whisper |
| faster-whisper | CTranslate2ベースのWhisper推論エンジン | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | 高速Transformer推論 | MIT | https://github.com/OpenNMT/CTranslate2 |
| **TEN VAD** | **既定の音声区間検出**（`--vad_method ten`）。DLL を同梱 | **Apache-2.0** | https://github.com/TEN-framework/ten-vad |
| silero-vad | 代替の音声区間検出（旧既定）。faster-whisper 内蔵の v6 ONNX も同じモデル | MIT | https://github.com/snakers4/silero-vad |
| onnxruntime | 上の ONNX モデルの実行 | MIT | https://onnxruntime.ai |
| PyTorch | GPU計算基盤（CUDA 12.8 / sm_120 対応） | BSD-3-Clause | https://pytorch.org/ |
| PyAV | 音声デコード（faster-whisper 経由） | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| ffmpeg | 音声前処理・フィルタリング（別プロセスとして実行） | **LGPL v3**（同梱ビルド） | https://ffmpeg.org/ |
| libsndfile | 音声の読み書き（`soundfile` 経由） | **LGPL-2.1-or-later** | https://github.com/libsndfile/libsndfile |
| PyInstaller | exe 化（ブートローダーが exe に入る） | GPL-2.0 + 凍結物の配布を認める例外 | https://github.com/pyinstaller/pyinstaller |
| PyYAML | 設定ファイルの読み込み | MIT | https://pyyaml.org/ |
| Anime Whisper | 日本語アニメ調セリフ向けモデル（`-m anime-whisper`） | MIT | https://huggingface.co/litagin/anime-whisper |
| Kotoba-Whisper | 日本語蒸留Whisper。Anime Whisperのベース | Apache-2.0 | https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 |
| audio-separator | ボーカル抽出（MDX / Mel-Band-Roformer）。**スクリプト版のみ** | MIT | https://github.com/karaokenerds/python-audio-separator |
| stable-ts | タイムスタンプ再調整（実験的）。**スクリプト版のみ** | MIT | https://github.com/jianfch/stable-ts |

CUDA / cuDNN（NVIDIA）と Intel OpenMP も `torch` と `ctranslate2` の wheel 経由で
同梱されています。再配布条件は THIRD-PARTY-NOTICES.md に記載しています。

開発のきっかけ：[Faster-Whisper-XXL](https://github.com/Purfview/whisper-standalone-win)（Purfview作）のRTX 5090対応版が有料かつソース非公開だったため、同等機能をオープンソースのみで再実装したもの。

## オプション一覧

exe 版とスクリプト版で共通です。**スクリプト版でしか動かないものには
「スクリプト版のみ」と付けてあります**（[違いの一覧](#exe版とスクリプト版の違い)）。

### モデル・デバイス

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--model` | `-m` | `large-v3` | モデル名 / エイリアス / ローカルパス / HFリポジトリID。`large-v3`, `large-v3-turbo` 等。transformers 形式（`anime-whisper` 等）の変換は**スクリプト版のみ** |
| `--model_dir` | | なし | モデル保存先ディレクトリ。未指定時は自動ダウンロード。変換モデルの置き場所も兼ねる |
| `--list_models` | | | エイリアス一覧を表示して終了 |
| `--reconvert` | | なし | 変換済みキャッシュがあっても再変換する（**スクリプト版のみ**） |
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
| `--beam_size` | `-bs` | `5` | ビームサーチ幅。**既定のままを推奨**（`10` は9本中8本で同等または悪化） |
| `--best_of` | `-bo` | `5` | 候補数。**既定のままを推奨**（同上） |
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
> 導入時の実測では9本のTVアニメ録画で全体CERが 24.3% → 22.0%、
> 30秒ブロック単位では 26.0% → 23.4% に改善し、取りこぼし（カバレッジ）は
> 変わりませんでした（**旧既定 silero・採点スクリプト修正前の数値**です。
> 現行の既定と指標での再測定はしていません）。現行既定での実効は
> ログの `[LOOP]` 行で確認でき、5時間22分の一挙放送で10セグメント1583字、
> 子供向け4本で6セグメント82字を落としています。
> 「きゃあああああ」のような短い悲鳴や
> 「そそそそんなわけ」のような言い直しは残ります。
> 長い叫びをそのまま字幕に載せたい場合は `--loop_filter false` で無効化できます。

### VAD（音声区間検出）

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--vad_filter` | `-vad` | `True` | VAD有効化 |
| `--vad_method` | | `ten` | VADバックエンド。既定は TEN VAD（Apache-2.0） |
| `--vad_threshold` | | バックエンド別 | 音声判定閾値（低いほど感度高）。未指定なら `ten` は `0.75`、silero 系は `0.45` |
| `--vad_segment_mode` | | `clip` | 検出区間をモデルへ渡す方法。`clip`=`clip_timestamps`、`collect`=無音を波形から切る |
| `--vad_min_speech_duration_ms` | | `250` | 最短音声チャンク（ミリ秒） |
| `--vad_max_speech_duration_s` | | なし | 最長音声チャンク（秒） |
| `--vad_min_silence_duration_ms` | | `3000` | 無音待機時間（ミリ秒） |
| `--vad_speech_pad_ms` | | `900` | 前後パディング（ミリ秒） |

> **VADモデルについて:**  
> **既定は TEN VAD（`--vad_method ten`、閾値 0.75）です。** silero より台詞の
> 取りこぼしが少なく、TVアニメ録画15本で全ファイル silero 以上でした
> （24分もの9本で全文CER 19.3% → 16.1%、子供向け4本で 31.1% → 21.9%）。
> 効き方は素材で大きく違い、会話劇では 0.1〜1.4pt、子供向けでは 4〜21pt 動きます。
>
> **閾値はバックエンド間で意味が違います。** 確率のスケールがモデル固有なので、
> silero 向けに調整した値を `ten` に渡すと必ず外れます。未指定のままにしてください。
>
> `--vad_method` で `silero_v4_fw` / `silero_v5_fw` / `silero_v6` / `silero_v6_fw` を
> 指定した場合は faster-whisper の内蔵VADが動きますが、これらはすべて同じ
> 内蔵 silero v6 に解決されます。`silero_v5` のように `_fw` が付かない名前は
> 本ツール側でVADを走らせてから区間を渡す経路で、**実測では内蔵VADより
> 全文CERで3pt良い**という結果でした。`pyannote_v3` / `auditok` / `webrtc` も
> 指定できますが、いずれも既定より劣ります。

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

### ボーカル抽出（**スクリプト版のみ**）

exe では指定すると理由を表示して止まります（[違いの一覧](#exe版とスクリプト版の違い)）。

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
| `--realign` | | なし | タイムスタンプ再調整（実験的、**スクリプト版のみ**）。字幕整形と併用すると整形側が失われるため、整形が有効なときはスキップされます |
| `--realign_device` | | なし | realign用デバイス |
| `--version` | | | バージョン表示 |
| `--checkcuda` | `-cc` | | CUDAデバイス数表示 |
| `--verbose` | `-v` | `False` | デバッグ出力 |

---

## Amatsukaze 記載例

**まず「空」を試してください。** 既定が本 README の数値を出した条件そのものです。

```
# 推奨
（空欄）

# 日本語の字幕に整形する（16字2行）
--standard_asia

# 言語を固定する（分割で出る短い断片の誤判定対策）
--language ja --standard_asia

# 字幕を標準フォーマットに（42字2行。英語などに）
--standard
```

> **`--beam_size 10 --best_of 10` は載せていません。**
> 以前このドキュメントは全ての例にこれを付けていましたが、
> 実測で **9本中8本が beam 5 と同等または悪化**しました
> （全文CER 22.0% → 22.3%、時間は +2.8%）。既に欄にある場合は消してください。

### 効くかどうか測ってから使うもの

以下は**実測で既定に負けた**か、素材によって逆効果になります。

```
# VAD感度UP（小声の拾い漏れ対策）
#   → TEN VAD では逆効果。0.45 まで下げると精度が 74.6% まで落ちます。
#      既定の 0.75 が最良でした（0.90 も4本合計で +1.3pt 悪化）
--vad_threshold 0.3

# ハルシネーション対策強化
#   → hallucination_silence_threshold はアニメ素材では正常セグメントも消えます
#      （台詞間の無音が長いため）。ループ抑制が既定で有効なので通常は不要
--repetition_penalty 1.2
```

### 実験的オプション（非推奨）

音声フィルターを通すとセグメント数が減る場合があり、実用レベルに達していません。
**本 README に載せた精度はすべてフィルターなしで測ったものです。**

```
# ノイズ多い素材（実験的）
--ff_loudnorm --ff_lowhighpass
```

## ライセンス

MIT
