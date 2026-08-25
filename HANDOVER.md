# 引継ぎドキュメント

whisp-carrier の設計判断・既知の問題・検証環境をまとめたドキュメント。
**フォークして改善する人が読むべきものはここに集めてある。**

**この文書は長寿命。** 設計が変わったときだけ書き換える。
現在地は [STATUS.md](STATUS.md)、測定と作業の記録は [MEASUREMENTS.md](MEASUREMENTS.md)。

| 見たいもの | 場所 |
|-----------|------|
| 今どこにいるか・次に何をするか | [STATUS.md](STATUS.md) |
| なぜそう作ったか・踏むと壊れる罠・eval の使い方 | **この文書** |
| 測定結果・作業の記録・完了した項目 | [MEASUREMENTS.md](MEASUREMENTS.md) |

**数値はここに置かない。** 現行の基準値は [STATUS.md](STATUS.md)、
測定の詳細は [MEASUREMENTS.md](MEASUREMENTS.md)、利用者向けは
[README.md](README.md) の「測定した精度」。**出どころを1箇所に保つため。**

| 節 | 内容 |
|----|------|
| [運用上の注意](#運用上の注意) | 踏むと測定や配布が静かに壊れるもの。**最初に読む** |
| [アーキテクチャ](#アーキテクチャ) | ファイル構成と依存関係。スクリプト版の実務もここ |
| [ドキュメント構成](#ドキュメント構成readme-は日本語がトップ開発版は-handover-に寄せた) | 3文書の役割分担。**ドキュメントを直す前に読む** |
| [再パックの発動条件](#再パックの発動条件ドキュメントだけの変更では固め直さない)・[リリースノートの書き方](#リリースノートの書き方2026-08-23-に決めた)・[版番号とタグの運用](#版番号とタグの運用スクリプト版だけの修正では版を上げない) | 配布の運用ルール |
| [なぜ 1.0 ではなく 0.9.0 なのか](#なぜ-10-ではなく-090-なのか) | 版番号が約束していること |
| [設計判断と理由](#設計判断と理由) | 既定値をそう決めた根拠 |
| [精度検証の環境（eval/）](#精度検証の環境eval) | 素材・指標・使い方 |
| [既知の問題](#既知の問題) | 未解決のもの |
| [今後やるなら](#今後やるなら) | 優先度の低い候補と、追わないと決めたもの |
| [開発環境再構築手順](#開発環境再構築手順) | pip・CUDA・隔離環境 |

---

### なぜ 1.0 ではなく 0.9.0 なのか

**0.1.0 として一度公開した後、0.9.0 に上げた**（`whisp_carrier.VERSION`）。
精度と構造の指標では本家を上回っていて配布物も揃っているが、
**1.0 が約束するのは「この CLI 表面を維持する」ことなので、まだ早い。**
理由は5つ。

- **現行 exe の動作確認が1台だけ。** RTX 5090 / Ryzen 9 5900XT / Windows 11 (26200) のみ。
  **旧スクリプト版は第三者が2機種で確認している**が別物（下記）。
  **exe を配る以上「うちの環境で動かない」が最初の報告になりうる**
- **Amatsukaze 連携の実証が n=1。** 同一素材で完全一致したのは強い証拠だが1本1回
- **CLI を変える余地を自分で残している。** 配線されていない
  `--vad_window_size_samples`、互換のため受け取って無視する
  `--hallucination_silence_th_temp` / `--model_preload`、そして
  [要望駆動で判断すると決めた機能](MEASUREMENTS.md#保留機能面の穴は要望駆動で判断する)（話者分離・
  フィルターの実用化・`--realign` の安定化）
- **未解決の既知の問題が残っている。** [音声フィルター](#2-音声フィルターが実用レベルに達していない)は
  再現自体が取れておらず、[`--realign`](#3---realign-が不安定) は不安定、`--batched` は品質未評価
- **採点指標が確定したのが直近。** [#19](MEASUREMENTS.md#19-採点指標に欠陥があった参照キューの表示時間を発話時間として使っていた) で
  欠陥が見つかって #1〜#18 が参考情報に落ちたばかりで、指標そのものが
  もう1サイクル分の検証を経ていない

**1.0 に上げる条件は「現行 exe が他人の環境で1回動くこと」。** それが v1.0 の中身そのもの。

#### 第三者環境での動作報告（2026-08 上旬・スクリプト版）

**Amatsukaze の作者（rigaya 氏）が検証し、Amatsukaze のドキュメントに掲載している。**
[該当箇所](https://github.com/rigaya/Amatsukaze#whisper%E3%81%AB%E3%82%88%E3%82%8B%E5%AD%97%E5%B9%95%E7%94%9F%E6%88%90)。

| 環境 | 結果 | 時期 |
|------|------|------|
| RTX 4080 + Amatsukaze 改造版（**bat 指定＝スクリプト版**） | 「精度含め問題なく動作」。Amatsukaze 側の拡張は不要との評価 | 2026-08-02 |
| RTX 2070 | 「問題なく動作」 | 2026-08-06 |

**現行 exe（0.9.1）についての第三者からの最初の報告は、不具合だった**（2026-08-24）。
`CUDA_PATH` が同梱 cuBLAS を隠していて起動後に停止する
（[記録](MEASUREMENTS.md#n-cuda_path-が同梱の-cublas-を隠していた2026-08-24-報告092-で修正)）。
0.9.2 で直したが、**「他人の環境で1回動く」はまだ満たしていない。**

**ただし検証されたのは 8月初旬のスクリプト版で、現在の配布物とは別物。**
その後に TEN VAD の既定化・ループ抑制・採点指標の修正・exe 配布が入っている。
**現行 exe（0.9.0）を第三者環境で通した記録は無い。**
なお **`.bat` 経由の連携が実際に動いた実例**でもある
（[現行版では未検証](#スクリプト版の実務exe版との差分)と書いているのはこのためで、
旧版では動いていた）。

#### 訂正して伝える必要がある説明（sm_120 フォールバック説）

**2026-08-03 に作者へ次の説明を送っており、それが Amatsukaze の
ドキュメントに反映されている。実測と合っていない。**

> XXL（無料版）でも RTX 5090 で動作自体はする。PyTorch の安定版が sm_120 カーネルを
> 含まないため sm_89 へサイレントフォールバックし、20〜40% の性能低下が起きる。
> エラーにならないのでユーザーが気づきにくい

**実測との食い違いが3点。**

| 当時の説明 | 実測 |
|-----------|------|
| 動作自体はする | **既定では落ちる。** `-ct float32` を明示して初めて動く（[#10](MEASUREMENTS.md#10-faster-whisper-xxl-との比較既定同士推奨設定では未実施)・[実運用ログ](MEASUREMENTS.md#実運用での裏付けamatsukaze-から素の-xxl-を呼ぶと落ちる2026-08-23)） |
| サイレントフォールバック | **エラーで停止する。** `RuntimeError: cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED` |
| 20〜40% の低下 | **約2倍遅い**（float32 は float16 の半分の速度） |

**機構の説明も怪しい。** XXL の推論は PyTorch ではなく **CTranslate2** なので、
PyTorch のカーネル有無は直接効かない。**同梱している CTranslate2 / cuBLAS ビルドの差**が
実態に近いと考えられるが、**そこは検証していないので断定しないこと。**
言えるのは「XXL r245.4 は既定で落ち、float32 なら動くが約2倍遅い。
whisp-carrier は同じ機で float16 のまま動く」まで。

**版を変えると波及する先。** `whisp_carrier.VERSION`（バナーに出るので**exe の再ビルドが必要**）、
アーカイブのファイル名、SHA-256、このドキュメントの見出しとリリースノート。
**アップロード後に変えるとリリースを作り直すことになるので、公開前に決めること。**

### 再パックの発動条件（ドキュメントだけの変更では固め直さない）

**このプロジェクトは測定値が動き続けるので、成果物をドキュメントの状態に
結びつけると測り直すたびに再パックが発生する。** 実際に1日で3回固め直した。
**そこで発動条件を決めた。**

| 変わったもの | 再パックするか |
|-------------|--------------|
| `whisp_carrier.py` などのコード（exe の挙動） | **する**（再ビルドも必要） |
| `VERSION`（バナーとファイル名に出る） | **する** |
| 同梱物の増減（`whisp_carrier.spec` の datas / binaries / コピー対象） | **する** |
| `LICENSE*` / `THIRD-PARTY-NOTICES.md` | **する。** 帰属と再配布条件に関わる |
| `whisp-carrier.yaml.example`（利用者が使う雛形） | **する** |
| `README.md` / `README_en.md` の本文だけ | **しない** |
| `HANDOVER.md` | **しない**（同梱していない） |

**理由は3つ。**

1. **分割アーカイブは中のファイルだけ差し替えられない**（7-Zip は multi-volume の
   更新をサポートしない）。「1ファイル直すために4.8GBを固め直す」しか手が無い
2. **利用者が精度を最初に見る場所はリリースノート**で、そこは何度でも編集できる。
   同梱 README が精度の細部で古くても、導入・Amatsukaze 設定・注意点として正しければ実害がない
3. **`dist/whisp-carrier/README.md` は次のビルドで spec が勝手に更新する**ので、
   コード変更のついでに自動的に追いつく

### リリースノートの書き方（2026-08-23 に決めた）

**リリースノートはそのリリース時点のスナップショットで、後から書き直さない。**
数字が動いたら次のリリースのノートに新しい数字を書く。

| そのリリースは | ノートの形 |
|---------------|----------|
| **数字を動かす**（既定の変更・モデル・素材追加・採点指標の修正など） | **フル。** 精度の表・既定の構成・Amatsukaze の設定・対象範囲まで書く |
| **数字を動かさない**（不具合修正・パッケージング・ドキュメント） | **差分。** 変更点とダウンロードだけ書き、フルのノートへリンクする |

0.9.1 は後者にした（[出力が 0.9.0 とバイト一致](#091-のビルド検証2026-08-236回目)しているので数字が動いていない）。
**リンク先は `../../releases/tag/v0.9.0` の相対形にする**とリポジトリ名の変更で切れない。

**差分ノートにも見出しの数字1つは残す。** GitHub はトップに最新リリースだけを出すので、
そこが数値ゼロだと「何のツールでどれくらいの精度か」が辿れなくなる。
1行の重複と引き換えに新規の読者を拾う判断。

**例外は1つ。exe を変えずに数字が動く場合**（[#19](MEASUREMENTS.md#19-採点指標に欠陥があった参照キューの表示時間を発話時間として使っていた) の採点コード修正がまさにこれで、
18本の測定が無効になったがリリースは出ていない）。**そのときはノートではなく
README の「測定した精度」を直す。** 数値の出どころをそこ1箇所に集約してあるのは元々そのため。

**ただしライセンス・帰属・同梱物の一覧の誤りは「ドキュメントだけ」に該当しない。**
[TEN VAD がクレジット表から抜けていた件と PyInstaller の記載漏れ](#参考プロジェクト表と-third-party-notices-の欠落を直した2026-08-23)は
この基準で再パックした。**判定に迷ったら「配布物の中身を正しく説明しているか」で切る。**

**現在の既知のずれ。** `0.9.0` のアーカイブ内 `README.md` は 44,842 バイトで、
ソースは 45,140 バイト。**差は [#22](MEASUREMENTS.md#22-anime-whisper-を現行パイプラインで再検証した4-の結論は維持) の
anime-whisper の精度引用だけ**で、しかも **anime-whisper は exe から使えない**
（変換にスクリプト版が必要）。上の基準では再パック不要と判断した。
`THIRD-PARTY-NOTICES.md` は一致している（15,087 バイト）。

### 版番号とタグの運用（スクリプト版だけの修正では版を上げない）

**版番号は配布物（exe）に紐づける。** タグを打つのは exe を作って公開したときだけ。
**スクリプト版だけの修正では版を上げず、タグも打たない。** main に置いて、
次に exe を作るときにまとめて版を上げる。

**理由が3つ。**

- **exe の作り直しは再ビルド約3分半＋再パック約8分**で、軽微な修正のたびに払う額ではない
- **スクリプト版は[人柱版](#スクリプト版の実務exe版との差分)で、利用者は `git pull` する層。**
  版番号で状態を指す必要が薄い
- **リリースページには `Source code (zip / tar.gz)` が自動で付く**ので、
  タグを打った時点のスクリプト版は**それだけで配布できている**

**スクリプト版のために別リリース（pre-release）は作らない。**
exe とスクリプト版は**同じコミットから作られる**ので、
「exe は正式、スクリプトは pre-release」という区別が成立しない。
実際に違うのは**サポート範囲**で、それは README と HANDOVER で宣言するもの。
リリースが2件並ぶと利用者が「どちらを落とすか」で迷うだけで、得るものがない。

**この方式の帰結。main の HEAD と配布 exe が乖離するのが常態になる。**
だから**「配布した exe がどのコミットの状態か」をここに書いておく必要がある。**

| 項目 | 状態（2026-08-23 時点） |
|------|----------------------|
| **ビルド済みだが未公開の exe** | **`0.9.1`、18:37:44 ビルド（`dist/whisp-carrier/`）。[J](MEASUREMENTS.md#j-同梱-ffmpeg-が使われていなかった2026-08-23) の修正と [K](MEASUREMENTS.md#k-torch-を外して配布サイズを削る2026-08-23) の torch 除外が入っている。未コミット・未パック・未リリース** |
| 公開済みの exe | `0.9.0`、13:44:55 ビルド。**`版番号変更`（`6ac58a5`）のコード状態** |
| アーカイブ内の docs（0.9.0） | 13:17〜13:18 のスナップショット（**[#22](MEASUREMENTS.md#22-anime-whisper-を現行パイプラインで再検証した4-の結論は維持) の反映前**） |
| main の HEAD | `d915eb3`（イシューテンプレート追加、版更新基準の明文化）。**0.9.1 の変更はまだ未コミット** |
| タグ | **ローカルには無い。** GitHub のリリース作成時にサーバ側で作られたので、`git fetch --tags` してから `git rev-parse v0.9.0` で確認する |

#### 0.9.1 のビルド検証（2026-08-23、6回目）

**環境変数なしで slim になる**ので、この表の数値は素のビルドのもの。

| 確認項目 | 結果 |
|---------|------|
| ビルド | `BUILD_EXIT=0`、`[spec] default (slim): dropping torch...`、`[spec] ffmpeg verified (known LGPL build)` |
| 同梱 CUDA | `keeping 13 CUDA libraries (1870 MB), dropping 2403 MB of torch` + `dropped 3 duplicate torch/lib binaries (766 MB)` |
| payload | **2239 MB**（0.9.0 は 4780 MB）。`_internal/torch` は存在しない |
| exe 本体 | **13.2 MB**（0.9.0 は 45.6 MB）。PYZ から torch の Python モジュールが消えたぶん |
| バナー | `whisp-carrier 0.9.1 \| ctranslate2 4.8.1 \| torch not bundled \| CUDA: True` |
| **字幕出力の回帰** | **16kHzモノラルWAVで SRT/VTT の MD5 が記録値と完全一致**（`AE218346...` / `6A6791FD...`） |
| **J の回帰** | **AAC 入力・PATH から ffmpeg を除去して exit 0・字幕生成・stderr 空**（0.9.0 は `[WinError 2]` / exit 1） |
| `silero_v5` | exit 1 で理由と代替（`silero_v5_fw`）を表示 |
| `silero_v5_fw` | exit 0（onnxruntime 経由なので torch 不要） |
| 生きた設定の保全 | `whisp-carrier.yaml` の MD5 が前後で不変（`399D11649E65FF7102DAE603ECBD7ECD`） |
| 同梱物 | LICENSE / LICENSE.ffmpeg.txt / LICENSE.ten-vad.×2 / README.md（48,249B・更新後）/ THIRD-PARTY-NOTICES.md（17,403B・更新後）/ .yaml.example |

**パックする前に `whisp-carrier.yaml`（3,163B の生きた設定）を外へ出すこと。**
手順は[この節の後半](#版番号とタグの運用スクリプト版だけの修正では版を上げない)にある `Move-Item` の形。`-x` による除外は効かない。

**次に版を上げるのは、[再パックの発動条件](#再パックの発動条件ドキュメントだけの変更では固め直さない)に
該当する変更が入って exe を作り直すとき。** そのとき
`whisp_carrier.VERSION` とタグとアーカイブ名と SHA-256 が一斉に動く。

**手順。`-x` による除外は使わない**（下の罠を踏むため）。
**生きた設定を一時的に外に出してから固める。**

```powershell
# 1. 開発機の設定をアーカイブ対象の外へ退避する
Move-Item dist\whisp-carrier\whisp-carrier.yaml dist\whisp-carrier.yaml.devbak -Force

# 2. 退避できたことを目で確認する（.example だけが残っていること）
Get-ChildItem dist\whisp-carrier -File | Select-Object -ExpandProperty Name

# 3. 除外スイッチなしで固める。フォルダ名を渡すこと（末尾に \* を付けない）
cd dist
& "C:\Program Files\7-Zip\7z.exe" a -v1900m -bsp0 whisp-carrier-0.1.0.7z whisp-carrier

# 4. 中身を確認する。whisp-carrier.yaml が無く .example があること
& "C:\Program Files\7-Zip\7z.exe" l whisp-carrier-0.1.0.7z.001 > list.log 2>&1
& "C:\Program Files\7-Zip\7z.exe" t whisp-carrier-0.1.0.7z.001    # 整合性

# 5. 設定を戻す（MD5 で戻ったことを確認する）
Move-Item dist\whisp-carrier.yaml.devbak dist\whisp-carrier\whisp-carrier.yaml -Force
```

**所要は圧縮だけで6〜10分。** 進行状況は `.001.tmp` の**サイズ**で見ること。
**ファイル名の有無で判断すると誤る**（`.tmp` は開始直後に作られ、完了時に消えるので、
「名前がある＝進行中」も「名前が無い＝失敗」もどちらも成立しない）。
`7z` プロセスの有無とログの `Everything is Ok` / `PACK_EXIT=0` で確定させる。

**`\*` を付けないのは展開時の体裁のため。** `.\dist\whisp-carrier\*` にすると
中身だけが詰まるので、利用者が展開したときに3000個のファイルがその場に散る。
フォルダ名を渡せば `whisp-carrier\` ごと復元される。

> ## ⚠ `-xr!whisp-carrier.yaml` は黙って効かなかった
>
> **このドキュメントは当初 `-xr!whisp-carrier.yaml` で除外する手順を書いていたが、
> 実測すると除外されずにアーカイブへ入った。** 小さなフォルダで再現を取ってある
> （`whisp-carrier.yaml` と `.example` と `sub\keep.txt` を置いて実行 →
> **3ファイルすべてが入る**）。エラーも警告も出ない。
>
> **設定やライセンスの同梱可否を「効いたか分からないスイッチ」に委ねてはいけない。**
> ファイルを物理的に外へ出す方式なら、固める前に `Get-ChildItem` で目視できる。
>
> あわせて `.\dist\whisp-carrier -xr!...` の組み合わせでは
> `指定されたファイルが見つかりません` の警告つきで **0 files, 32 bytes の空アーカイブ**が
> できた。原因は詰めていない（除外方式を捨てたので追う価値が無い）。
> **空アーカイブは `Archive size: 32 bytes` で判別できる。**
> `-xr!*.yaml` のようなワイルドカードも論外で、`.example` まで落ちる。

**同梱物は8点 + `_internal/`。** アーカイブの一覧で確認した内容。

```
whisp-carrier\LICENSE                        MIT（このプロジェクト）
whisp-carrier\LICENSE.ffmpeg.txt             LGPL v3（同梱 ffmpeg）
whisp-carrier\LICENSE.ten-vad.license.txt    Apache-2.0（TEN VAD）
whisp-carrier\LICENSE.ten-vad.notices.txt    同 NOTICE
whisp-carrier\README.md                      日本語マニュアル（spec がコピーする）
whisp-carrier\THIRD-PARTY-NOTICES.md         同梱物の一覧
whisp-carrier\whisp-carrier.exe              47,802,820 バイト
whisp-carrier\whisp-carrier.yaml.example     設定ファイルの雛形
whisp-carrier\_internal\...                  3021 ファイル
```

**`whisp-carrier.yaml`（開発機の生きた設定）が入っていないことを確認済み。**
入れてはいけない理由は、`override: true` + `language: ja` + `standard_asia: true` が
**呼び出し側の指定を打ち消す**こと。利用者が whisper-option 欄に `--language en` と
書いても黙って `ja` になり、頼んでいない16字2行の整形も掛かる。
**このファイルのコメントには訂正済みの誤解も残っている**
（「Amatsukaze は `--beam_size 10 --best_of 10` を渡してくる」は
[実際には whisper-option 欄の文字列](MEASUREMENTS.md#13-実運用設定beam-10は測定設定beam-5よりわずかに悪い)）。
配ると誤解も一緒に配ることになる。

**精度側と連携側は閉じている。** 追加で測るものは無い
（[A の結果](MEASUREMENTS.md#-完了実運用と測定が同一素材で一致した2026-08-22)・[15本の総括](MEASUREMENTS.md#15本の総括2026-08-22)）。

---

## 運用上の注意

踏むと測定結果や配布物が静かに壊れるもの。

- **`dist/whisp-carrier/whisp-carrier.yaml` は「生きた設定」として置いてある。**
  この開発機の Amatsukaze がこれを読んでいる（`vad_method: silero_v5` ほか）。
  exe の隣にあると自動検出されるので、**素の挙動を見たいときは `--no_config`**
  を使うこと。再ビルドしても spec が退避・復元するので消えない。
  **配布物に含めるかは未決定。**
- **`eval/run.py` は常に `--no_config` を渡す。** 上の設定ファイルが転がっていても
  実験条件が変わらないようにするため。手で `whisp_carrier.py` を叩いて
  比較実験するときは自分で付ける必要がある。
- **本家XXLを比較で回すときは `-ct float32` が必須。** 無料版 r245.4 は
  RTX 5090 で float16/int8 系が cuBLAS で落ちる（[測定結果 #10](MEASUREMENTS.md#10-faster-whisper-xxl-との比較既定同士推奨設定では未実施)）。
  `--model_dir` 未指定でも既存のHFキャッシュは使わず自前でダウンロードするので、
  初回は large-v3 の3.09GBを取りに行く。
- **`ctranslate2.converters` は絶対に `excludes` に入れないこと。**
  `ctranslate2/__init__.py` が無条件に `from ctranslate2 import converters, ...`
  しているので、除外すると `import ctranslate2` が壊れて **exe が起動しなくなる。**
  ビルドは成功して実行時に全滅するので気付きにくい。一度踏んだ。
- **Amatsukaze は子プロセスの出力を UTF-8 として読む。固定値でオフにできない。**
  `TranscodeManager.cpp` が両経路で `param.isUtf8Log = true` を代入している。
  だから `_setup_console_encoding()` で stdout / stderr を UTF-8 に固定してある。
  **`PYTHONIOENCODING` を cp932 等に設定すると、この固定を尊重して従うため
  ログの日本語が `\uXXXX` にエスケープされる**（化けはしないが読みにくい）。
  Amatsukaze から呼ぶ時は設定しないこと。
- **`whisp-carrier.yaml` に `vad_method` を書かないこと。書くと新既定を潰す。**
  既定が `ten` になったので無指定で最良の構成になる。
  以前は既定が内蔵VAD経路（`silero_v5_fw`）で、Amatsukaze は `--vad_method` を
  渡してこないため yaml に `vad_method: silero_v5` と書く必要があった。
  **その行が残っていると `override: true` で旧VADに固定される。**
  この開発機の `dist/whisp-carrier/whisp-carrier.yaml` からは外してある。
  `--language` は yaml 側で `ja` を指定したままだが、
  [出力は1セグメントも変わらない](MEASUREMENTS.md#13-実運用設定beam-10は測定設定beam-5よりわずかに悪い)ので実害はない。
- **Amatsukaze のパスは現行に載せ替え済み。**
  `C:/Users/Owner1/whisper-xl/dist/whisp-carrier/whisp-carrier.exe` を叩いている。
  **再ビルドは同じパスを上書きするので、Amatsukaze 側の設定変更は不要。**
  旧ログにある `whisper-xl.bat` はワークスペースに存在しない（git化前のローカル版）。
- **`_venv_inass/` と `_venv_fsmn/` は測定用の隔離環境で、主環境には入れられない。**
  inaSpeechSegmenter は `tensorflow[and-cuda]`（Windows で解決不能）、
  funasr-onnx は `numpy<=1.26.4` を要求する。**主環境に入れると
  記録されている全数値が再現できなくなる。** 区間は `eval/ext_vad_dump.py` が
  JSON に書き、`--vad_method precomputed` が読む（[測定結果 #18](MEASUREMENTS.md#18-ten-vad-が-silero-を置き換えた取りこぼしは実装で直せる)）。
  `.gitignore` に `_venv*/` が無ければ足すこと
- **凍結ビルドは（1）起動時に `CUDA_PATH` / `CUDA_HOME` を自分のプロセスから捨て、
  （2）`device=cuda` のとき同梱 cuBLAS を絶対パスで先に読む。この2つを消すと、
  CUDA Toolkit の世代が違う環境で exe が落ちる。**
  層ごとの検証には `WHISP_CARRIER_CUDA_FIX`（`preload` = 層2のみ / `off` = 両方無効）。
  **README には書かない開発用の穴で、「配布する exe そのままで不具合を再現する」ために置いてある。**
  先読みは `cublasLt` → `cublas` の順（依存の向き）。**PATH に足す方式は採らなかった** —
  子プロセスに波及し、PATH のどこかにある同名 DLL が勝つ余地が残るため。
  CTranslate2 は cuBLAS を遅延ロードし、**`CUDA_PATH` が設定されているとそこの
  `bin` を見て、同梱ぶん（`_internal`）を検索対象から外す。** CUDA 13/11 や
  アンインストール後の残骸を指していると `cublas64_12.dll` が見つからず、
  `Library cublas64_12.dll is not found or cannot be loaded` で停止する
  （[記録](MEASUREMENTS.md#n-cuda_path-が同梱の-cublas-を隠していた2026-08-24-報告092-で修正)）。
  実装は `whisp_carrier._use_bundled_cuda()` で**凍結ビルド限定**。
  スクリプト版に効かせてはいけない（同梱ぶんが無く、torch が自分の
  `torch/lib` を登録して解決している）。
  **この開発機には CUDA Toolkit 12.8 が入っているので、この修正が入るまで
  同梱ぶんは一度も使われていなかった。** 「同梱した」と「同梱ぶんが読まれた」は
  別のことで、後者は `Get-Process` の `.Modules` でフルパスを見るまで確認できない。
- **`sample/`（録画物）と `_eval/` はコミットしないこと。** `.gitignore` 済み。
  `build/` `dist/` `build-full/` `dist-full/` `_models/` `_tmp*/` も同様。
  `_tools/` は三段構成で `_tools/ffmpeg/PROVENANCE.txt` だけを追跡している。

---

## アーキテクチャ

```
whisp_carrier.py         ← エントリポイント。argparse、transcribe、出力処理
audio_filter.py          ← ffmpegラッパー + 段ごとの計測/検査（audio-separator
                           はスクリプト版のみ。exeでは除外）
vad.py                   ← 外部VADバックエンド。**既定の ten**（TEN VAD）、silero、
                           precomputed、pyannote, auditok, webrtc。区間の集約は
                           segments_from_probabilities() が全モデル共通で持つ
whisp_vad_patch.py       ← 内蔵VADの差し替え（ONNX / セグメント供給経路）
loop_filter.py           ← ループ出力の検出と破棄（測定結果 #11・#12）
subtitle_format.py       ← 字幕整形（文分割・折り返し・再タイミング）+ sanitize
whisp_models.py          ← モデルエイリアスとCTranslate2変換
whisp_config.py          ← YAML設定ファイル / プロファイル
whisp-carrier.bat        ← Amatsukaze互換ランチャー（スクリプト版を呼ぶ）。
                           引数を素通しするだけ。現行版では未検証
whisp-carrier.yaml.example ← 設定ファイルのサンプル
whisp_carrier.spec       ← PyInstaller用。ffmpegのライセンス検査、同梱物の
                           除外、ライセンス文・README・設定の配置もここ

README.md                ← 日本語・利用者向けマニュアル（exe 版前提）。配布物にも入る
README_en.md             ← 上の英語版（要約）
THIRD-PARTY-NOTICES.md   ← 同梱物の一覧（出典・ライセンス）。配布物にも入る
_tools/ffmpeg/PROVENANCE.txt ← 同梱ffmpegの出所と検証記録（バイナリは非追跡）

eval/                    ← 精度検証用。CLI本体からは独立、exeには入らない
  ext_vad_dump.py        ← 主環境に入れられないVADを別venvで走らせ区間をJSONに出す
                           （inaSpeechSegmenter / FSMN-VAD。測定結果 #18）
  arib_vtt.py            ← ARIB B24字幕VTTのパーサ（ルビ分離・DRCS・消去キュー）
  normalize.py           ← 正規化3段階とCER・LCS（帯状DP）
  prep.py                ← 素材の調査と16kHzモノラルWAV抽出
  run.py                 ← 1設定ぶんの実行とJSONキャッシュ
  score.py               ← 幻覚・対応差分・CER・カバレッジ/精度の採点
  strip_loops.py         ← ループ判定セグメントを落としたコピーを作る（上限測定用）
  xxl_convert.py         ← Faster-Whisper-XXL のJSONを採点スキーマに変換
```

### 依存関係の要点

| パッケージ | 役割 | 備考 |
|-----------|------|------|
| torch 2.8.0+cu128 | GPU推論 | sm_120対応の最低バージョン |
| faster-whisper 1.2.1 | 文字起こし本体 | 内蔵VADがsilero v6。requirements の下限もこれに合わせている |
| **ten-vad 1.0.6.8** | **既定のVAD** | Apache-2.0、1.3MB。wheel に prebuilt な `ten_vad.dll` が入り `ctypes` で読む。**exe では spec が `ten_vad/lib/Windows/x64` に置く**（`collect_dynamic_libs` の平坦化ではパスが合わない） |
| audio-separator 0.44.5 | ボーカル抽出 | スクリプト版のみ。exeでは除外（凍結環境でscipyが解決できない） |
| stable-ts | --realign機能 | 実験的、不安定。`WHISP_CARRIER_FULL=1` でのみ同梱 |
| PyYAML 6.0.3 | 設定ファイル読み込み | 6.0.3 で動作確認 |
| transformers 5.15.0 | transformers形式モデルのCT2変換 | 変換時のみ必要。exeには同梱しない。ctranslate2 4.8系のコンバータが `from_pretrained(dtype=...)` を使うため 4.56 以上が必要 |
| safetensors 0.8.0 | 同上（重みの読み込み） | transformers が要求する |

### 配布形態の切り分け：exe が主経路、スクリプト版が実験的機能

**exe 版を通常の経路とし、`-m anime-whisper` と `--realign` と
`--ff_vocal_extract` を使いたいときだけスクリプト版を使う。**

exe には `transformers` を同梱しないので、transformers 形式モデルの初回変換が
できない。影響するのは `whisp_models.resolve()` の5分岐のうち1つだけで、
ビルトインサイズ・CT2形式のHFリポジトリ・ローカルのCT2ディレクトリ・
変換済みキャッシュはすべて影響を受けない。実質 `-m anime-whisper` を
exe に直接渡せなくなるだけで、**黙って壊れず exit 2 で案内を出す。**

```
[MODEL] error: this exe build cannot convert transformers-format models
        (No module named 'transformers').
Convert once with the script version, then pass the converted directory:
  python whisp_carrier.py <input> -m anime-whisper
  whisp-carrier.exe <input> -m <path>\_models\ct2-litagin-anime-whisper-float16
Built-in sizes (large-v3 etc.) and CTranslate2 models work here as usual.
```

判断の根拠は3つ。**回避策がある**（スクリプト版で1回変換して変換済み
ディレクトリを渡す。変換は初回のみ）、**対象モデルが非推奨**
（[測定結果 #4](MEASUREMENTS.md#4-anime-whisper-は-large-v3-に負けたモデルカードと逆) で large-v3 に17pt負け、9本全敗）、
**未検証の経路を配布物に載せない**こと。

**サイズは判断理由ではない。** spec の旧コメントは「transformers を入れると
数GB増える」と書いていたが、**実測は381MB**（transformers 本体は37MBで、
残りは llvmlite 115MB / cv2 112MB / scipy 53MB などの連れ子）。除外の理由は
容量ではなく、凍結ビルドでの変換という未検証経路を抱えないこと。
[罠2](#罠2-蒸留モデルの-alignment_heads-はデコーダ層数と合っていない)（alignment_heads の修復）は
失敗すると Python 例外もトレースバックも出さずにプロセスが即死するので、
非推奨モデルのためにその経路を配布物に載せる価値がない。

`WHISP_CARRIER_FULL=1` は **`--realign` 専用**（audio-separator は全ビルドから
除外した）。ただし **`--realign` は推奨設定では必ずスキップされる**ので、
**配布は通常ビルド。** 詳細は[フルビルドの結果](MEASUREMENTS.md#フルビルドの結果whisp_carrier_full1)。

### スクリプト版の実務（exe版との差分）

**位置づけは人柱版で、サポート対象ではない**（README にもそう書いた）。
記録している精度の数値はすべて exe と同じ主経路のもので、
スクリプト版でしか動かないのは[3つ](#4-exe化しても一部機能は動かない)だけ。

| 項目 | exe 版 | スクリプト版 |
|------|--------|-------------|
| 必要なもの | Windows x64 + NVIDIA ドライバ（CUDA 12.8以上）。**ffmpeg は同梱** | 追加で Python 3.11・CUDA Toolkit 12.8・PATH の通った ffmpeg |
| 設定ファイルの探索先 | exe の隣（`sys.executable` の親） | **`whisp_carrier.py` と同じフォルダ**（`whisp_config.base_dir()` が `__file__` を返す） |
| Amatsukaze から呼ぶ | exe のパスを直接指定（**検証済み**） | `whisp-carrier.bat` を指定（**現行版は未検証**） |
| モデル変換・ボーカル抽出・realign | 使えない | 使える |

**`whisp-carrier.bat` は引数をそのまま `whisp_carrier.py` に渡すだけのランチャー。**
Python の解決順は `WHISP_CARRIER_PYTHON` →
`%LOCALAPPDATA%\Programs\Python\Python311\python.exe` → `py -3.11` → `python`。
終了コードは `endlocal & exit /b %ERRORLEVEL%` で正しく伝わる
（`%ERRORLEVEL%` の展開はこの行の実行時なので、python の終了コードが入る）。
**旧版（`whisper-xl` 名義）では Amatsukaze から .bat を叩いて動いていた**
（[該当節](MEASUREMENTS.md#amatsukazeがどこで呼ぶかログで確定)の旧ログがそれ）が、
**現行版で通した記録は無い。**

**設定ファイルの探索先の違いが実際に効く。** 開発機では
`dist/whisp-carrier/whisp-carrier.yaml` が生きた設定として置いてあるが、
**.bat 経由ではこれを読まない。** スクリプト版で同じ設定を使いたいなら
リポジトリ直下にもう1つ置くことになる（現在は `.example` だけ）。
`eval/run.py` は常に `--no_config` を渡すので測定には影響しない。

#### コマンドの形（README から移したもの）

**オプションは exe と同一。** `whisp-carrier.exe` を `python whisp_carrier.py` に
置き換えるだけ。README は exe 版のマニュアルなので、こちらの形は載せていない。

```powershell
python whisp_carrier.py "動画.mp4" -m large-v3 -l ja --standard_asia -o source -pp
```

**transformers 形式モデルの変換は初回だけスクリプト版で走らせる。**
できた `_models/ct2-*` を exe に渡せば、以降は exe で使える。

```powershell
# 1回だけ。短い音声で走らせれば変換だけ済む
python whisp_carrier.py test_speech.wav -m anime-whisper -o . -f srt

# 以降は exe でも使える
whisp-carrier.exe "動画.mp4" -m _models\ct2-litagin-anime-whisper-float16
```

変換の中身（`tokenizer.json` の生成と alignment heads の検証、なぜ黙って壊れるか）は
[モデルの差し替え](#モデルの差し替えはエイリアス--初回変換にした)の罠1・罠2。
初回はモデル約3GBを取得し、`_models/` 配下に float16 で約1.5GB 置く。

**`--ff_vocal_extract` の実務。** `mb-roformer` は初回にモデル約900MBを落とす。
**アニメのように声とBGMの帯域が近い素材では声まで消える**ことがある。
そして **`--ff_loudnorm --ff_lowhighpass` と併用しないこと**（帯域通過の後に
ボーカル抽出を掛けると音声が消える）。
どちらか一方にする。なおノイズの多い実況系では
`--ff_loudnorm --ff_lowhighpass` のほうが安定する（こちらは exe でも使える）。
**いずれも精度の数値には無関係**で、`eval/run.py` は `--ff_*` を1つも渡さない。

## ドキュメント構成（README は日本語がトップ、開発版は HANDOVER に寄せた）

> **更新（2026-08-23）。開発ドキュメントを寿命で3つに割った。**
> 以前は HANDOVER.md 1本で 6088行あり、**半分が追記専用の実験ノート、
> 4分の1が「前回セッション時点の状態」**という混在状態だった。
> 症状は長さではなく、**揮発性の節に完了項目が居残って冒頭に警告バナーが必要になること。**
>
> | ファイル | 寿命 | 扱い |
> |---------|------|------|
> | [STATUS.md](STATUS.md) | **揮発性**。今日正しくないと困る | **置換する。** 終わったら本文ごと MEASUREMENTS へ |
> | **HANDOVER.md**（この文書） | 長寿命。設計判断・罠・eval | 設計が変わったときだけ書き換える |
> | [MEASUREMENTS.md](MEASUREMENTS.md) | **追記専用**。測定と作業の記録 | **消さない。** 伸びるのが正常 |
>
> **ルールは2つ。** 揮発性は置換・記録は追記。そして**数値の出どころは1箇所**
> （利用者向けは README、開発側の基準値は STATUS、詳細は MEASUREMENTS の各エントリ）。
>
> **分割で踏んだ罠。** GitHub のアンカーは見出し文字列から作られるので、
> 節を別ファイルへ動かすと、その節を指す同一ファイル内リンクが全部切れる。
> **359本あったので機械で書き換えた**（[手順](#見出しを直したらアンカーを確認する)）。
> あわせて、分割スクリプトが自分の入力（HANDOVER.md）を上書きして1回やり直した。
> **入力と出力を別名にすること。**

**2026-08-23 に入れ替えた。** `README.md` が日本語、`README_en.md` が英語。
以前は `README.md` が英語で `README_ja.md` が日本語だった。

| ファイル | 役割 |
|---------|------|
| `README.md` | **日本語・利用者向けマニュアル。exe 版を前提に書く。** GitHub のトップに出るもので、**配布アーカイブにも同梱する**（spec が post-COLLECT でコピーする） |
| `README_en.md` | 上の英語版。要約でよい（オプション全表は日本語側にしかない） |
| `HANDOVER.md` | **開発・測定・設計判断。スクリプト版の実務もここ**（上の節）。利用者向けの導線は README から張る |

**README は exe 版のマニュアルであって、スクリプト版のコマンドは載せない。**
`pip install` も `python whisp_carrier.py …` も、**配布アーカイブを解いた利用者には
実行できない指示**で、置いても迷わせるだけ。exe 利用者に必要なのは
**「exe では何ができないか」**（[違いの表](#4-exe化しても一部機能は動かない)相当）だけで、
**「スクリプト版をどう動かすか」は HANDOVER の担当**にした。
オプション一覧には `スクリプト版のみ` の印だけを残してある（これは exe の制約情報）。

**分け方の原則は「exe版マニュアルとスクリプト版マニュアルを2本作らない」。**
共通のものは README に1つだけ置き、スクリプト版で違うところだけを
HANDOVER に書く。**理由は重複が測定値を腐らせるから**で、
実際に旧 README は beam 10 を「基本（推奨）」として全7例に載せ、
[否定された後も残っていた](MEASUREMENTS.md#13-実運用設定beam-10は測定設定beam-5よりわずかに悪い)。
**数値の出どころは README の「測定した精度」1箇所**に集約してある。

**README から HANDOVER へ送っているもの**（README 側には概要とリンクだけ置く）。

- 開発環境の構築手順（pip・CUDA Toolkit・隔離環境）→ [該当節](#開発環境再構築手順)
- モデル変換の罠2つ（tokenizer.json / alignment heads）→ [該当節](#モデルの差し替えはエイリアス--初回変換にした)
- ファイル構成と依存関係 → [アーキテクチャ](#アーキテクチャ)
- exe のビルド方法とライセンス検査 → [開発環境再構築手順](#開発環境再構築手順)
- 音声フィルターがなぜ実験的なのか → [既知の問題 2](#2-音声フィルターが実用レベルに達していない)

### 参考プロジェクト表と THIRD-PARTY-NOTICES の欠落を直した（2026-08-23）

**README の「ベースとなったプロジェクト」表から TEN VAD が抜けていた。**
**現在の既定VADで、DLL を同梱していて Apache-2.0（表記義務がある）**のに、
旧既定の silero-vad だけが載っている状態だった。既定を差し替えたときに
表を直し忘れたもの。あわせて、同梱しているのに載っていなかった
**onnxruntime / PyAV / libsndfile / PyYAML / PyInstaller** を追加し、
**ライセンス列**を足した（copyleft の ffmpeg と libsndfile がその場で見えるように）。
`audio-separator` と `stable-ts` には「スクリプト版のみ」と付けた。
表の末尾から [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) に送る導線も入れた。

**THIRD-PARTY-NOTICES.md 側にも1件欠落があった。PyInstaller のブートローダー。**
**exe に物理的に組み込まれるが、`_internal/` にも PYZ の TOC にも現れない**ので、
あのファイルの再生成手順（2つのソースを突き合わせる方法）では原理的に拾えない。
**同梱物の中で唯一 GPL 由来のコード**で、
[ブートローダー例外](https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt)が
まさにこのケース（凍結物を自分のライセンスで配る）のために書かれている。
節2・節5・再生成手順の3箇所に追記した。

**教訓。既定を変えたら「参考プロジェクト表」も差分の対象に入れること。**
測定値と違って誰も再現確認をしないので、放っておくと残る。

**英語側にだけ意図的に残した重複が1つある。ファイル構成。**
HANDOVER は日本語しか無いので、英語の読者にはコードの地図が他に無い。
**数値ではないので腐っても被害が小さい**という判断で `README_en.md` に短い版を置いた
（`HANDOVER.md` が日本語であることも明記した）。逆に日本語の README からは削って
[アーキテクチャ](#アーキテクチャ)に一本化している。

### 見出しを直したらアンカーを確認する

**GitHub のアンカーは見出しの文字列から作られるので、括弧を1つ足すだけで
その節を指す全リンクが切れる。** しかも切れても何も起きない（黙ってページ先頭に飛ぶ）。

2026-08-23 の整理で実際に5件出した。`配布アーカイブ（2分割で作った・2026-08-23）` に
`、0.9.0` を足して**4箇所**、`F. TEN VAD を既定にする（最優先）` を `（完了）` にして1箇所。
前者は見出しを元に戻して直した（参照が4つある見出しは、正確さより安定を採る）。

**確認は機械でやること。** 現在 HANDOVER には内部リンクが 359 本ある。

```python
# 本文中の内部リンク参照を集めて、見出しから作ったスラッグ集合と突き合わせる。
# スラッグ規則: 小文字化 → **`~~ を除去 → [x](y) は x に → 空白をハイフン →
#               \w と - 以外を除去（日本語はそのまま残る）
import re, pathlib
text = pathlib.Path('HANDOVER.md').read_text(encoding='utf-8')

def slug(h):
    s = re.sub(r'^#+\s*', '', h)
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    for t in ('**', '`', '~~', '*'):
        s = s.replace(t, '')
    s = re.sub(r'[ \u3000]+', '-', s.lower().strip())
    return re.sub(r'[^\w\-]', '', s, flags=re.UNICODE)

anchors = {slug(m.group(2)) for m in
           re.finditer(r'^>?\s*(#{1,6})\s+(.*)$', text, re.M)}
missing = [t for t in re.findall(r'\]\(#([^)]+)\)', text) if t not in anchors]
print(len(missing), sorted(set(missing)))
```

**スラッグ規則は公開仕様ではないので近似**で、誤検出はありうる。
出たら「見に行く」の合図として扱うこと。0件なら安心してよい。

**README を直すときの注意。** 利用者向けの数値を動かすなら、
[現行の基準値](STATUS.md#現行の基準値歌唱除外指標は-19-修正後)と出典の節を確認してから書くこと。
`README_en.md` は要約なので、**日本語側にしかない節（オプション一覧・記載例・注意事項）を
英語側に足す義務は無い**。ただし**数値を書くなら両方を同時に直す。**

---

## 開発経緯

- Faster-Whisper-XXL Pro（Purfview作）はRTX 5090対応だが有料（£50）かつソース非公開
- 同等機能をOSSで再実装し、RTX 5090 (sm_120) でネイティブ動作するCLIを作成
- Amatsukaze（rigaya改造版）からの呼び出しを主なユースケースとして開発
- **無料版XXL（r245.4）は RTX 5090 で既定のまま動かないことを実測で確認した。**
  `-ct float32` を明示すれば動くが float16 の2倍遅い。精度は健全な素材で
  こちらが勝っている（[測定結果 #10](MEASUREMENTS.md#10-faster-whisper-xxl-との比較既定同士推奨設定では未実施)）

---

## 設計判断と理由

### condition_on_previous_text = False（デフォルト）

Whisperはデフォルトで前セグメントのテキストを次のプロンプトに渡す。
これにより文脈が繋がるメリットがある反面、一度ハルシネーションに入るとループが止まらない。

テストの結果：

- `True` → 76セグメント（ハルシネーションで後半が壊れる）
- `False` → 200セグメント（正常動作）

精度よりも安定性を優先し `False` をデフォルトにした。
なお本家XXLは `true` のままで、代わりに `hallucinations_list` /
`ignore_dupe_prompt` などの対策群を既定ONにしている。

**ループ抑制を入れた後に実素材で再評価し、`False` が正しいと確定した。**
`loop_filter` があれば `true` を救えるのではないかという読みで
死亡遊戯 #09 に当てたところ、**カバレッジ 59.7% → 21.3%、推論時間 5.9倍**で、
出力が 18:16（尺30:00）で止まった。しかも**`loop_filter` は1件も落とせない**
（句読点だけのセグメントは `normalize()` が空文字列にし、
残りの反復はセグメントをまたぐので、どちらも検出器の対象外）。
上の「76セグメント」の観測と同じ壊れ方が、後処理を足しても残る。
**XXL が `true` で運用できているのは対策群と組み合わせているからで、
`true` だけを借りてくることはできない。** 詳細は[測定結果 #15](MEASUREMENTS.md#15-condition_on_previous_texttrue-は死亡遊戯を破壊した仮説の否定)。

### initial_prompt = None（デフォルト）

以前は `default="auto"` だったが、argparse は文字列をそのまま渡すため、
実際には「auto」という単語が毎回初期プロンプトとしてWhisperに投入されていた。
`None` に修正済み。互換のため `-prompt None` と `-prompt auto` は
どちらも「プロンプトを渡さない」として扱う。

### hallucination_silence_threshold = 0（無効）

faster-whisperの内蔵機能。無音後のセグメントをハルシネーションとして破棄する。
しかしテストでは正常なセグメントも巻き添えで消えた（212→104）。
アニメ素材は台詞間の無音が長いため、この機能は使えない。

### 音声フィルター = 実験的

`--ff_loudnorm`, `--ff_lowhighpass`, `--ff_vocal_extract` はいずれも
セグメント数が激減する。フィルターなしが最も安定する。

長らく「内蔵VADとの相性問題」と説明していたが、実測したらそれ以前の
順序バグだった（チャンネル選択が無音を作る、帯域通過が実質無効）。
順序バグは修正済みで、段ごとの計測も入った。詳細は[既知の問題 2](#2-音声フィルターが実用レベルに達していない)。

**このプロジェクトが報告している精度数値はすべてフィルター未経由。**
`eval/run.py` は `--ff_*` を一切渡さない。実運用も渡さない方針で決着した。

### VADバックエンド：既定は TEN VAD（silero 系は全部同じモデルだった）

> **この節のタイトルはもともと「カスタムVAD = 実質不要」だった。**
> silero をどう指定しても同じモデルが動くので差が出ない、という意味で正しかったが、
> **モデルを silero 以外に替える発想が抜けていた。** TEN VAD にすると
> 9本で全文CER 19.3% → 16.1%、カバレッジ 82.6% → 86.6% になり、
> [既定を TEN にした](MEASUREMENTS.md#修正後の9本2026-08-22)。以下は silero 系の内訳として今も有効。

**`--vad_threshold` の既定はバックエンド別に解決される**（`ten` 0.75 / silero 0.45、
`whisp_carrier.VAD_THRESHOLD_DEFAULTS`）。確率スケールがモデル固有で、
**同じ数値が同じ厳しさを意味しない**ため。TEN に silero の 0.45 を当てると
精度が 74.6% まで落ちる。yaml やCLIで明示すると解決を上書きするので、
片方に合わせた値を書き置かないこと。

faster-whisper 1.2+ は内蔵で silero v6 (ONNX) を使用。
`--vad_method` で silero_v4_fw / v5_fw / v6_fw を指定しても全て同じ内蔵モデルが動く。
`faster_whisper/vad.py` の `get_vad_model()` が
`assets/silero_vad_v6.onnx` を決め打ちしているため。

whisp-carrier 側 `vad.py` の `silero_v3` / `silero_v4` / `silero_v5` も同様で、
`load_silero_vad()` はインストール済み silero-vad パッケージのモデルを返すだけなので
バージョン指定は効いていない。torch.hub フォールバックも master を引く。
つまり**7つの silero 系 `--vad_method` はすべて同じモデル**だった。
`--vad_onnx` を足したのはここを本当に切り替えられるようにするため。

外部VAD（pyannote 等）がハルシネーションを起こしやすいのは事実だが、
VADモデルの精度そのものより Whisper への音声の渡し方が効くと考えて
`--vad_segment_mode` を足した。**渡し方が効くのは実測で確かめられたが、
効く向きは予測と逆だった。** 詳細は次項。

### 外部VADの区間は2通りの渡し方がある（既定は clip）

**結論を先に書く。この節はもともと「collect を既定にする」理由を説明していたが、
実測でその理由は否定された。既定は `clip`（従来動作）である。**
コードの読み解き自体は正しいので、なぜ予測が外れたかの記録として残す。

`faster_whisper/transcribe.py` を読むと、発話区間を Whisper に伝える2つの方法は
等価ではない。

内蔵VAD経路（`if vad_filter and clip_timestamps == "0"`、890-891行）:

```
get_speech_timestamps() → collect_chunks() が無声区間を波形から物理削除
→ np.concatenate → 30秒窓が発話で埋まる
→ restore_speech_timestamps() が元の時刻に戻す（1009-1010行）
```

`clip_timestamps` 経路（既定）:

```
波形は無加工。generate_segments() が seek_clips を作り
segment_size = min(nb_max_frames, content_frames - seek, seek_clip_end - seek)
→ pad_or_trim()
```

つまり**1秒のクリップは「1秒の音声＋29秒のゼロ埋め」としてエンコーダに入る**。
これは Whisper が最も幻覚を出す入力形と考えられていた。さらに `all_tokens` と
`prompt_reset_since` はクリップ境界を越えて持ち越されるので、
片方のクリップで始まったループが次のクリップに供給される。
既定の `speech_pad_ms=900` / `min_speech_duration_ms=250` は短いクリップを
大量に作るので、これは例外ケースではない。

そこで `--vad_segment_mode collect` を追加し、外部VADの結果を内蔵経路に
流せるようにした。当初はこちらを既定にした。どちらを使ったかは `[VAD]` 行に
必ず出る。

**ここまでの推論が実測で否定された。** 24分もの9本では差が出ず（[#2](MEASUREMENTS.md#2-24分ものでは-collect-と-clip-に差が無かった一挙で決着)）、
一挙放送5時間22分では **collect のほうがループ文字を3.3倍出した**
（1628字 vs 491字）。CERも clip が2pt良い。ゼロ埋めが幻覚を誘発するという
読みは、少なくともこの素材では成立しない。詳細は[#6](MEASUREMENTS.md#6-一挙放送で-clip-が勝ち既定を-clip-に戻した)。

外していた理由は推測になるが、collect は無音を削って発話を詰めるので、
**30秒窓が「切れ目のない音声」で埋まり、デコーダに止まる手がかりが無くなる**。
ゼロ埋めは無駄な計算だが「ここには何も無い」という明示的な信号にはなる。
`all_tokens` の持ち越しは両経路に共通なので差にならない。

さらに collect には固有の副作用がある。`restore_speech_timestamps()` が
削除した無音をまたぐセグメントを引き伸ばすため、**30秒超セグメントの発生源が
この経路**になる（[#1](MEASUREMENTS.md#1-既定経路に30秒超セグメントの欠陥があった既存バグ修正済み)）。一挙では54箇所・最長190秒。clip 側は
波形を加工しないので原理的に発生しない。

以上から既定は `clip`。`collect` は残してある。内蔵VADと同じ渡し方なので
比較の対照として要るし、速度は collect のほうが15%速い（一挙で1036s vs 1196s）。

外部VADが0区間を返した場合の扱いも変えた。以前は空の `clip_timestamps` 文字列を
渡していて、faster-whisper はこれを「制限なし」と解釈してファイル全体を
黙って書き起こしていた。今は理由を表示して内蔵VADに落とす。

### 内蔵VADの差し替えは本体パッチではなくモンキーパッチ

`whisp_vad_patch.py` に閉じてある。site-packages は触らない。
ここには知っておく必要のある非対称性がある。

`get_speech_timestamps()` の中の `model = get_vad_model()` は**呼び出し時に**
自分のモジュール名前空間から引かれるので、`faster_whisper.vad.get_vad_model` を
差し替えれば `transcribe.py` に触らずに効く。`functools.lru_cache` が付いているので
差し替え前に `cache_clear()` を呼ぶ順序だけ守る（差し替え後のラムダには
`cache_clear` が無い）。

逆に `get_speech_timestamps` 自体は `transcribe.py` が23行目で
`from faster_whisper.vad import get_speech_timestamps, ...` と
**import 時に束縛済み**なので、`faster_whisper.vad.get_speech_timestamps` を
差し替えても何も起きない。当てる名前は
`faster_whisper.transcribe.get_speech_timestamps`。
`WhisperModel.transcribe` と `BatchedInferencePipeline.transcribe` は
どちらもこの1つの属性を読むので、1回のパッチで両エンジンに効く。

ONNX の差し替えには入力名の検証を入れた。`SileroVADModel.__call__` は
v5/v6 の signature（`input` / `h` / `c`、状態は 1x1x128）に対して書かれている。
v4 以前は `sr` 入力を要求し状態の形も違うので、ファイルを差し替えるだけでは
動かない。該当する graph を渡したら onnxruntime の分かりにくいエラーではなく
「モデルクラスの差し替えが必要（未実装）」と言って止まる。

区間の正規化（ソートと重複マージ）は省略できない。`SpeechTimestampsMap` は
`chunk["start"] - previous_end` を累積して「ここまでに見た無音」を数えるので、
重複した2区間があるとこの累積が逆行して以降のタイムスタンプが全部ずれる。
外部バックエンドは実際に重複を返す（`vad.py` は `speech_pad_ms` を区間ごとに
適用するので、パディング2倍より近い隣接区間は重なって返ってくる）。

パッチは `with` で囲んだ範囲だけ有効にしている。VADフックは
`transcribe()` の中で、セグメントのジェネレータが返る前に同期的に呼ばれるので、
`engine.transcribe()` の呼び出しだけを囲めば足りる。フックが1回も呼ばれなかった
場合は警告を出す（faster-whisper が内蔵VAD分岐に入らなかったということなので、
起きていない結果を報告するより言った方がいい）。

なお内蔵VADは `providers=["CPUExecutionProvider"]` 決め打ちなので、
`--vad_device cuda` は内蔵側には効かない。`vad.py` のバックエンド専用。

### 字幕整形は文字列ではなく単語ストリーム上で行う

`subtitle_format.py` は生テキストを折り返すのではなく、
単語タイムスタンプの列を分割・グループ化してから各行を組み立てる。
こうしないと1セグメントを複数キューに割るときにタイムスタンプが作れない。

`--word_timestamps` が無効な場合は、セグメントのテキストから
文字数比で時間を按分した擬似単語を生成して同じ経路に流す。
実測では擬似単語経路でも実単語経路と同じ分割結果になった。

禁則処理は軽量版のみ。行頭に来てはいけない文字（`、。）」…` 拗音促音など）が
次行の先頭になる場合、`max_line_width` を1文字超えることを許して前行に残す。
つまり指定幅は厳密な上限ではなく、禁則優先の目安になる。

### 30秒超セグメントのサニタイズは常時通す

`sanitize_segments()` を推論直後・字幕整形の前に必ず通す。
これは整形オプションの有無に関係ない。設計上のポイントが4つある。

**単語タイムスタンプで分割する。** 単語は個別に復元されるので実時刻を持つ。
セグメント内の単語間ギャップが `--max_gap`（既定3.0秒）を超えたら分割し、
テキストは単語列から再構成する。

**30秒保証は閾値に依存させない。** `--max_gap 0` で分割を無効にしても、
30秒を超えるピースは最も広いポーズで強制的に切る（`_enforce_duration()`）。
複数のポーズをまたぐ場合は再帰的に切る。Whisperの窓が30秒である以上
それを超える出力は復元の産物と断定できるので、ここはチューニングの
余地を残さないほうがいい。**単体テストでこの経路を通していなかったら
`--max_gap 0` で穴が開いたままだった。**

**擬似単語は使わない。** `--word_timestamps false` のときは分割の根拠が無いので
30秒でクランプするだけにする。`_pseudo_words()` は等間隔に補間するので、
まさに修復対象のセグメントに対して存在しないポーズを捏造してしまう。

**何もなければ入力リストをそのまま返す。** 修復不要なら同一オブジェクトを返すので
通常出力のIDが変わらない。修復が起きたファイルだけ `[FIX]` 行を出す。

背景と実測値は[測定結果 #1](MEASUREMENTS.md#1-既定経路に30秒超セグメントの欠陥があった既存バグ修正済み)。

### ループ抑制はセグメント単位で落とす

`loop_filter.py`。既定で有効、`--loop_filter false` で無効化。
字幕整形の前に通し、落としたものは `[LOOP]` 行に理由とタイムスタンプ付きで出す。

判定は3条件。

- 12字以上で文字種2以下
- 同一文字8連続以上
- 1〜6字の単位が4回以上**かつ繰り返し部分が12字以上**

**3つめのスパン条件が採点用検出器（`score.py` の `is_loop()`）との唯一の違い。**
これで `うわあああああ!` `そそそそんなわけ` のような実在の発話が残り、
健全6本の出力が無変更になった。それでいて合計CERは上限と同じ22.0%。
根拠は[測定結果 #11](MEASUREMENTS.md#11-ループを潰した場合の上限既定に載せるだけで本家を抜く)・[#12](MEASUREMENTS.md#12-実装した検出器は上限にそのまま到達した誤爆ゼロで)。

**セグメント単位で丸ごと落として構わない**ことも実測で確認済み。
カバレッジは 80.4% → 80.3% しか動かない（ループセグメントは参照に一致する
文字をほぼ持っていない）。

### モデルの差し替えは「エイリアス + 初回変換」にした

`whisp_models.py` を追加し、`-m anime-whisper` のような短い名前で
transformers 形式のモデルを扱えるようにした。faster-whisper は CTranslate2 形式しか
読めないため、初回実行時に `_models/ct2-<名前>-<量子化>/` へ変換してキャッシュする。

呼び出し側（Amatsukaze）から見ればモデル名が1個増えただけで、
CLIの形は変わらない。これを別スクリプトの手動変換にしなかったのは、
後述の2つの罠がどちらも「エラーにならずに静かに壊れる」種類のもので、
手順書に書いても踏むと判断したため。

エイリアスにはモデル固有のオプション既定値も持たせた。
`anime-whisper` は `language: ja` と `no_repeat_ngram_size: 5`（モデルカードのCER測定条件）を
自動で選び、`initial_prompt` が指定されていれば警告する。
これはモデルの性質であって呼び出し側の都合ではないので、モデル定義の隣に置くのが自然。
CLI/YAMLで明示された値は常に優先し、採用/非採用は `[MODEL]` 行に必ず出す。
`[CONFIG]` 行と同じ方針。

解決順は「エイリアス展開 → ローカルディレクトリ判定 → ビルトインサイズ判定 →
変換キャッシュ確認 → HF リポジトリ形式判定 → 必要なら変換」。
キャッシュ確認をHF問い合わせより先に置いたので、変換済みなら完全オフラインで動く。

#### 罠1: tokenizer.json が無いモデルは黙って壊れる

faster-whisper は `model_path/tokenizer.json` を探し、無い場合は
**openai/whisper-tiny のトークナイザを黙ってダウンロードして使う**
（`transcribe.py` の `WhisperModel.__init__`）。
large-v3 系（語彙 51866）に whisper-tiny（51865）のトークナイザを当てても
例外は出ず、それらしいが誤ったテキストが出るだけ。

`litagin/anime-whisper` は vocab.json + merges.txt しか持たないためこれに該当する。
変換時に fast tokenizer の `backend_tokenizer.save()` で tokenizer.json を生成し、
生成できない場合は変換を中止する（黙って進めるより止めた方が安全）。

#### 罠2: 蒸留モデルの alignment_heads はデコーダ層数と合っていない

単語タイムスタンプは特定のクロスアテンションヘッド（デコーダ層, ヘッド番号）から読む。
`anime-whisper` の `generation_config.json` に入っている alignment_heads は

```
[[7,0],[10,17],[12,18],[13,12],[16,1],[17,14],[19,11],[21,4],[24,1],[25,6]]
```

で、これは **whisper-large-v3（デコーダ32層）のもの**。
一方このモデルはデコーダ2層しかない。蒸留の際に教師の設定を引き継いだまま。

CTranslate2 のコンバータはこのリストをそのまま `config.json` に書き、
`align()` が範囲外のレイヤーを参照して**プロセスが即死する**。
Pythonの例外ではないのでトレースバックも `[ERROR]` 行も出ず、
`[STT] Transcribing...` の直後に終了する。`--word_timestamps` は既定 true なので、
素直に変換したモデルは100%この症状になる。

対策として変換後に `config.json` の alignment_heads を実際の
`decoder_layers` / `decoder_attention_heads` と突き合わせ、範囲外を除去する。
全滅した場合は CTranslate2 自身の既定（デコーダ後半層の全ヘッド）を書き込む。
anime-whisper は10個すべて範囲外なので、実際には後者（層1の全20ヘッド）になる。

修復後の単語タイムスタンプは実用範囲だった（7秒のテスト音声で文字単位の境界が
音声とずれていない）。ただし2層デコーダの1層から取った値なので、
large-v3 と同じ精度が出ている保証はない。長尺での再タイミング品質は未検証。

なお HuggingFace 上に第三者が上げている anime-whisper の CT2 変換物は、
この修復が入っていなければ同じクラッシュをするはず（未検証）。

変換物には `whisp-carrier-model.json` を書き込み、`format` の値で
古い（修復前の）キャッシュを検出して自動的に再変換する。
量子化が `--compute_type` と食い違う場合も同様に再変換する。

### 設定ファイルは Amatsukaze を触らないための仕組み

Amatsukaze の「追加オプション」欄を書き換えるのが面倒で、精度検証のたびに
設定を差し替えるのが現実的でなかった。そのため
`whisp_carrier.py`（exe版なら exe）と同じフォルダの `whisp-carrier.yaml` を
自動で読むようにしてある。呼び出し側は一切変更しなくてよい。

`override` の意味付け：

- `override: false`（コード上のデフォルト） … CLI 指定が優先。設定ファイルは
  CLI が指定しなかった項目だけを埋める。一般的な設定ファイルの挙動
- `override: true` … 設定ファイルが優先。呼び出し側（Amatsukaze の
  whisper-option 欄など）が渡したオプションを無視して上書きする

デフォルトを `false` にしたのは、設定ファイルに古い `language` が残っていた場合に
呼び出し側の指定が黙って無効化されるのを避けるため。
サンプルの `whisp-carrier.yaml.example` は狙いに沿って `override: true` にしてある。

どの値がどこから来たかは `[CONFIG]` 行に必ず出力する。精度検証では
結果とログを対応付けられることが重要なため。

設定ファイルの未知のキーは警告ではなくエラーにしている。
`beam_size` を `beem_size` と書き間違えたまま黙って無視されると、
比較実験そのものが無効になるため。

### 実行時の出力は英語、ストリームは UTF-8

`[CONFIG]` 行やエラーメッセージを日本語にすると、コンソールのコードページ次第で
Amatsukaze のログ画面が文字化けする。実行時出力は英語のみとし、
日本語はドキュメントと `whisp-carrier.yaml.example` のコメントに閉じている。

そのうえで、**素材由来のテキストは必ずログに載る**（`[LOOP]` が落とした文字列、
入力パス、`[OUT]` のパス、`[ERROR]` のファイル名）。ここは英語に寄せられないので、
**`_setup_console_encoding()` で stdout / stderr を UTF-8 に固定している。**
Amatsukaze が UTF-8 前提でデコードするため（ソースで確認済み）、これが正しい形。
`PYTHONIOENCODING` があれば尊重する。

`console_safe()` は保険として残してあり、**UTF-8 以外のストリームに書く時だけ**
`\uXXXX` にエスケープする。判定が「書けるか」ではなく「UTF-8 か」なのが要点で、
そこを間違えたのが元のバグ。経緯は[該当節](MEASUREMENTS.md#ログの文字化け原因は自分の出力エンコーディングだった解決済み)。

`eval/` 配下も同じ方針。調査中に自分の書いたスクリプトを cp932 で
回して文字化けさせたので、日本語ラベルと生グリフは全部 codepoint 表記
（`U+266A EIGHTH NOTE x14` の形）にした。cp932 に無い外字を素で書くと
`UnicodeEncodeError` で調査が途中で死ぬ。

---

## 精度検証の環境（eval/）

### 素材

Amatsukaze 最終出力の MKV + ARIB字幕VTT。`sample/` に置く。
**初期の10本**（`sample/mkv/` + `sample/short vtt/` + `sample/lomg vtt/`）と、
**後から追加した5本**（`sample/complete-test/`）がある。全ファイルが
aac 2ch stereo 48kHz の単一音声トラックなので `--ff_track 1` で正しい。

**初期10本。** 一挙放送1本（ハクメイとミコチ #01-#12、5時間22分）と24分アニメ9本。
全10本が別作品で、局は AT-X 7本 / テレ東2本 / 日テレ1本。
**記録されている測定結果はすべてこの10本のもの。**

**追加5本**（`sample/complete-test/`）。`prep.py` で MISSING も重複警告もなく
WAV 抽出済み。

| 素材 | 尺 | 備考 |
|------|-----|------|
| ニンジャラ「花火が開く夜空に」 | 23:54 | ループ最悪だったニンジャラの別話 |
| ぷにるはかわいいスライム #20 | 24:09 | 子供向け |
| 名探偵プリキュア！ #21 | 24:40 | 子供向け |
| **株式会社マジルミエ #01-#12** | **4:51:59** | **2本目の一挙放送** |
| おねがいアイプリ 情熱のトップ・パワー | 24:39 | 取りこぼし最悪だったアイプリの別話 |

**ニンジャラとアイプリの別話が入ったのが大きい。** どちらも
[問題の3本](#解決済み問題の3本は局ではなく内容だった)で、
「あの2本固有の事故か、作品の性質か」を初めて切り分けられる。
マジルミエは長尺の2本目なので、一挙の結論（[#6](MEASUREMENTS.md#6-一挙放送で-clip-が勝ち既定を-clip-に戻した)）が1本のみの観測でなくなる。

#### `sample/complete-test/` のフォルダ構成と命名規則

中身を読んで確定させた。**フォルダ名から想像する内容と違う。**

| 場所 | 中身 |
|------|------|
| `新規mkv/*.mkv` | 追加5本の映像 |
| `新規ARIBVTT/<stem>.vtt` | **その参照**（ARIB。`b24caption` + `%` エスケープの生データ） |
| `新規WhispVTT/<stem>-A.vtt` | **参照のコピー**（`新規ARIBVTT` と同一内容） |
| `新規WhispVTT/<stem>-A-gen.vtt` | **whisp-carrier の出力**（整形済み、16字2行） |
| `既存VTT/<stem>.vtt` | 初期10本のうち4本の参照 |
| `既存VTT/<stem>-gen.vtt` | **同・whisp-carrier の出力** |

つまり **`-gen` が付くものだけが仮説側**で、残りは全部参照。
`既存VTT` にあるのは 死亡遊戯 #09 / ニンジャラ（ガムッチ）/ アイプリ（はじめての）/
マジルミエ の4本で、前3本は測定で問題を出した3本そのもの＝**回帰確認用。**

**stem の一致は `新規mkv` ↔ `新規ARIBVTT` だけで成立する。** `-A` / `-gen` が
付くものは MKV と stem が違うので `prep.py` は拾わない（意図どおり）。

**`既存VTT` の3本は `sample/short vtt/` と stem が衝突する。** `prep.py` は
「先勝ち」で `complete-test` 側を採るが、**MD5 で照合したら3組すべて同一コピー**
だったので実害はない。**ただし PowerShell のワンライナーで照合すると
「3組すべて相違」という誤判定が出た。** この環境では比較は Python で行うこと。

**`-gen.vtt` をそのまま採点に使うことはできない。** `score.py` は
`_eval/hyp/<tag>/*.json` を読む前提で SRT/VTT パーサを持たない
（[次の着手 B](MEASUREMENTS.md#済み素材の追加と採点)）。加えて `-gen.vtt` は
**整形済み・実運用設定（beam 10 / `language: ja`）で生成されている**ので、
記録値（beam 5・整形なし）と条件が違う。目視確認用と割り切るか、
`eval/run.py` で条件を揃えて回し直すこと。

`python eval/prep.py sample --out _eval/wav` で調査と抽出をまとめて行う。
16kHz モノラル pcm_s16le に落とす。これは faster-whisper の `decode_audio()` と
`vad.py` が内部でやるのと同じ形なので測定内容は変わらないが、`vad.py` は
入力が `.wav` でないと ffmpeg を起こしてテンポラリWAVを作るため、
外部VADを使う比較では MKV のまま回すとデコードが2回走る。10本で約1.07GB。

#### 素材を足すべきか（判断済み）

**すでに出した結論には足す必要がない。** 深夜の会話劇は3本が9〜14%に
固まっていて分散が小さく、同系統を足しても数字が動かない。
AT-X の24分深夜アニメも7本あるので、局・尺・時間帯が同じものは飽和している。

当初は「発話率の低い素材がアイプリ0.32・死亡遊戯0.23の2点しかないので、
ルールを2点フィットさせるのは過学習になる」と考え、歌の多い作品と台詞の薄い
作品をあと2〜3本足す想定だった。**先にスイープを回して正解だった。**
新素材なしで応答曲線を取ったところ、**その2本はどちらも `--vad_threshold` に
反応しなかった**（[測定結果 #8](MEASUREMENTS.md#8---vad_threshold-は取りこぼしの-knob-ではなかった否定的結果)）。つまり素材を増やしても、
動かない knob のルールを校正することになっていた。

切り分けが済んで「このパラメータで直る」が見えたら、そのときに
同系統をあと2〜3本足して校正する。欲しいのは
**歌の多い作品**（アイドル系・幼児向け。OP/ED/挿入歌まで字幕が付く）と
**台詞が薄い作品**（日常系。尺に対して字幕文字数が少ない）。
なお**この2本は故障モードが別**である点に注意。死亡遊戯は novad で
カバレッジ+13.5pt、アイプリは+25.9pt で、アイプリのほうは歌唱区間が
主因の可能性が高い（silero は歌を発話として拾わない）。
**1つのルールで両方直る前提を置かないこと。**

追加のコストは安い。24分もの1本で5設定なら推論8分程度、
WAV抽出込みで10分。採点は全文CER+LCSが乗るので1本あたり数分。

**5.1ch の素材は要件から外した。** 既知の問題 2-a のガード
（要求チャンネルが実レイアウトに無ければ止める）を検証するのに
5.1が要ると当初考えたが、日本の地上波TVアニメはほぼ全部ステレオで、
5.1が乗るのは劇場作品のノーカット放送やBS/4Kの一部に限られる。
現実的に集まらない。そして**ガードの効く側はステレオ素材で全部テストできる**
（`--ff_fc` がステレオで無音になるのは手元10本すべてが該当する）。
5.1で正しく FC が取れる経路だけが未検証になるが、そちらは
`ffmpeg -af "pan=5.1|..."` で合成すれば足りる。

#### （解決済み）「問題の3本」は局ではなく内容だった

長らく「ループ最悪のニンジャラと取りこぼし最悪のアイプリは、番組名から推すと
どちらもテレ東の子供向け」という推測を置き、**交絡候補として
「局（エンコード特性・音声レベル）」と「子供向けという内容」の2つ**を挙げていた。
局のメタデータを確認するのが先だと書いていたが、
**追加素材で内容側が確定したので、局を調べる必要はなくなった。**

根拠は[測定結果 #14](MEASUREMENTS.md#14-追加素材5本取りこぼしは問題の3本ではなく子供向け番組の性質だった)。

- **新規のぷにる（30.6%）とプリキュア（34.8%）が同じ故障モードを示した。**
  「問題の3本」という括り自体が間違いで、**子供向け番組で一貫して起きる**
- **アイプリは別話でもカバレッジ47.0%**（既存話は47.3%）。1話の事故ではない
- **XXL でも同じ**（子供向け4本でカバレッジ 69.1% 対こちら 67.7%）。
  つまり局のエンコード特性ではなく、**音声そのものか Whisper の性質**

**ループと取りこぼしは別現象だったことも分かった。** 子供向け4本のうち
ループを出したのはニンジャラとぷにるだけで、プリキュアとアイプリは0件。
**ニンジャラのループは別話でも再発したが、アイプリでは再発していない。**

**次に素材を足すなら狙いは変わる。** 「子供向けを増やして原因を特定する」は
もう要らない（実装で直せないと分かった）。価値があるのは
**会話劇・職場ものを増やして「良い側の数字」の確度を上げること**、
または**死亡遊戯型（静かな台詞・低発話率）を増やして
[次の着手 C](MEASUREMENTS.md#不要になったc-死亡遊戯の取りこぼし)の校正に使うこと。**
後者の価値は上がった。**`novad` の発動条件を発話率で作るなら、
低発話率の素材が現状2本しかないので校正点が足りない**（[#15](MEASUREMENTS.md#15-condition_on_previous_texttrue-は死亡遊戯を破壊した仮説の否定)）。

### ARIB字幕を正解に使うときの実測事実

素材を実際に読んで分かったことで、指標の設計がここで決まった。

**字幕は連続チェーンで、無字幕区間はほとんど無い。** 字幕は次の字幕に
置き換わるまで表示され続けるので、10本中7本が消去キュー（clear-screen）ゼロ。
実際のギャップは OP前・ED後・一挙の話間だけ。採点可能な無字幕域の合計は約70分で、
そのうち **47分44秒（68%）が一挙1本の13区間**。公女殿下・Summer Pockets・
アイプリの3本は字幕被覆が97%超でギャップがゼロ、つまり幻覚の一次指標には使えない。

このため当初は**一挙が一次指標の主力**になると考えた。
**実測でこの読みは外れた**（[指標の設計欠陥](MEASUREMENTS.md#指標の設計欠陥記録)）。
一挙の話間ギャップもOP・ED・次回予告で埋まっているので、
無字幕域指標は結局どのファイルでも使えない。

**キュー終了時刻は発話終了を意味しない。** 連続チェーンなので終了時刻は
「次の字幕が置き換えた時刻」でしかない。25秒続くキューが実際にある。
タイムスタンプ精度の評価には使えないので、採点は開始時刻ベースで行う。

**DRCS（外字）は私用領域 U+EC00〜EC02 にマップされて本文に入っている。**
全10本にあり19〜300個。`%+{...%+}` のビットマップ定義も残っている。発話ではない。

**U+266A（♪）は1文字も無い。** 桃源暗鬼に U+266C BEAMED SIXTEENTH NOTES が17個。
「♪で歌唱区間を除外する」という当初の設計は成立しないので、
音楽記号は U+2669〜U+266F の範囲で拾う。同ファイルには U+27A1 が29個、
U+269E/U+269F が各3個ある。

**その U+2669〜U+266F 版も実質発動していなかった。** 後から全参照を調べたら、
**この局は音楽記号を DRCS ビットマップで送っていて私用領域にマップされる**ので、
照合すべき codepoint がそもそも無い。**歌唱区間の識別は
「波ダッシュ1文字だけのキュー」と「全体が `「」` で囲まれたキュー」で行う**
（`normalize.is_sung()`、`--exclude-songs`）。
**U+EC00 を音符と読むのは誤り**で、通常台詞にも大量に乗っている。
詳細は[測定結果 #16](MEASUREMENTS.md#16-歌唱区間を除外したら構図が変わったexclude-songs)。

**ルビが本文と同列の位置付きテキストとして入っている。** ARIB の文字サイズ制御
（`%^H`=SSZ 小 / `%^I`=MSZ 半角 / `%^J`=NSZ 標準）を追跡すれば分離できる。
SSZ 状態で描かれた実行がルビ。量は極端に偏っていて、ニンジャラ354ラン(939字)、
アイプリ410ラン(1024字)、他は0〜27。**この2本はルビ分離なしだとCERが無意味になる。**

**（）の中身は話者名だけではない。** 死亡遊戯に「スマホゲームの音」「すすり泣き」
「食べる音」、終末ツーリングに「鼻歌」が入る。効果音の注記と話者名が同じ記法なので
機械的に分離できない。どちらも認識器が出す表記ではないので一括で落としている。

**平文の非言語音は残す。** 「えっ…」「ぐっ…」「ハァ…」は（）の外に平文で入っている。
これを書き起こすのが anime-whisper の評価対象なので、消すと測りたいものが消える。

**字幕が番組末尾を数秒はみ出すのは良性。** 桃源暗鬼 +3.8s、ニンジャラ +4.5s、
アイプリ +2.1s。最終字幕の表示時間が番組末尾を超えるだけで、別編集ではない。
`prep.py` は10秒超のみ警告する。

### 指標

`eval/score.py` が3つ出す。狙いが違うので分けてある。

**無字幕域での誤検出（当初の一次指標。使えない）。** 参照が空と言っている区間に
落ちた仮説セグメントの本数・秒数・文字数。中点包含で判定するので、
字幕域との境界をまたぐセグメントは誤検出に数えない。**この素材では交絡していて
解釈できない**（[指標の設計欠陥](MEASUREMENTS.md#指標の設計欠陥記録)）。レポートには残っているが読まないこと。

**対応差分。** 2設定の占有区間の差。参照を必要としないので、
ギャップゼロの3本もこれで使える。数字より「聴くべきタイムスタンプの一覧」が本体。
差分スパンは元のセグメントより短いことが多いので、テキストは中点ではなく
重なりで拾う（中点だと空欄が出て一覧の意味が無くなる）。

**CER は2種類出る。必ず両方見る。**

`whole-region` は字幕域全体を1本の文字列にして突き合わせる。
セグメント粒度に一切依存しないので、**認識精度を見るならこちら**。
`plain` レベルのみ計算する。

`per-block` は字幕域を約30秒ブロックに束ね、仮説セグメントを中点で
割り当てる。タイミングとセグメント分割の善し悪しも一緒に課金される。
3段階の正規化（`asis` / `markup` / `plain`）を出すのはこちらだけ。

**2つの差がセグメント粒度の影響量を示す。** large-v3 系は1pt前後しか
開かないが、anime-whisper は 42.2% → 62.6% と20pt開く。
セグメントが4分の1しか無いので、1つの30秒セグメントが5つの参照キューぶんの
テキストを抱え、中点が入った1ブロックに全部落ちて隣が空になる。
**per-block だけを見ると粗く区切るモデルを不当に罰する。**
最初はこれで anime-whisper を誤読したので、比較の際は必ず全文側で判断すること。

#### 全文CERの実装欠陥2つ（修正済み・性質を知らないと直した意味が分からない）

どちらも一挙で踏んだ。数字の変化は[測定結果 #6](MEASUREMENTS.md#6-一挙放送で-clip-が勝ち既定を-clip-に戻した)にある。

**欠陥1: 仮説テキストの範囲がブロック外まで広がっていた。** 以前は
「最初のブロックの開始〜最後のブロックの終了」で仮説セグメントを拾っていた。
24分ものは無字幕域が最終キューより後ろにあるのでこの範囲の外だが、
**一挙の無字幕域は話間にあるので範囲の内側**で、OP・ED・次回予告・提供の
47分43秒ぶんが丸ごと挿入として課金されていた。長さ比が全文1.11 /
per-block 0.97 と食い違うのがその症状で、CERを14pt膨らませていた。
今はブロックに重なるセグメントだけを拾う。音楽記号で落としたブロックの
区間も同様に除外される（参照が歌詞を持たないので課金する意味がない）。

**欠陥2: 帯の打ち切り判定が「倍化した後」にあった。** `edit_distance_large` は
`if distance < width or width >= max_band` で抜けていたので、初期幅が
`max_band` の半分をわずかに超えるか下回るかで、厳密解になるか打ち切られるかが
変わった。一挙で実際に**長さ差が282字違うだけで片方だけ打ち切られた**
（collect は帯4208→8416で打ち切り＝上限値、clip は3926→7852→15704で厳密解）。
その状態で 36.0% vs 34.1% を並べて「clipが1.9pt良い」と読みかけた。
今は幅を上限にクランプしてから計算し、**上限値になった場合は
`CerResult.approximate` に記録してレポートに `UPPER BOUND` と出す。**
`max_band` の既定も8192→32768に上げた（38225字の一挙が厳密に回る）。
帯が文字列長以上になった場合は全幅＝厳密なので、そこも上限値扱いしない。

**全文CERを読むときは `UPPER BOUND` が付いていないことを先に確認すること。**

#### カバレッジと精度（CERの内訳）

全文CERの下に必ず1行出る。参照と仮説のLCS（最長共通部分列）から出す。

```
coverage  = LCS / 参照長    参照のどれだけを回収できたか
precision = LCS / 仮説長    出した文字のうちどれだけが参照に一致するか
```

**CERだけでは取りこぼしと誤認識を区別できない。** 削除も置換もコスト1なので、
同じCERに2つの原因があり得る。実例として、死亡遊戯 42.4% と
ニンジャラ 53.7% は数字が近いが中身が真逆で、
死亡遊戯はカバレッジ59.7%／精度87.3%（出したものはほぼ合っている＝取りこぼし）、
ニンジャラはカバレッジ75.4%／精度63.4%（出力の37%が参照外＝幻覚）。
**対処法が別なので、この区別が付かないと実装方針を決められない。**

レポートには判定を一言添える（`MISSED SPEECH dominates` /
`INVENTION dominates` / `both` / `close on both`）。しきい値は
「片方が他方の1.5倍以上」。

実装は帯状LCS（`normalize.lcs_large`）で、返り値は `(一致数, 厳密か)`。
`参照長 + 仮説長 - 2×LCS` は挿入削除のみの距離で、
その回数より遠くへ対角線から離れられないので、これが帯幅未満なら
帯は拘束していない。打ち切られた場合は LCS が**下限値**になるので
レポートに `LOWER BOUND` と出る（CER側の `UPPER BOUND` と向きが逆）。

コストは編集距離と同オーダーなので、全文CERの計算時間はおよそ倍になる。
per-block 側には付けていない。粒度の影響を受ける数字に内訳を足しても
読み解きが増えるだけなので。

#### 帯状DPと正規化

全文CERの実装は帯状DP（`normalize.edit_distance_large`）。
5000字×5000字は25Mセルで Python では回らないため、対角線から一定幅の
バンドだけ計算し、結果がバンド幅未満なら最適経路がバンドに触れていない
＝厳密解と判定する。触れていたらバンドを倍にして再計算する。
返り値は `(距離, 厳密か)` のタプル。
ランダム300件で厳密版と完全一致を確認済み。5000字で2〜13秒。
一挙（38225字）は帯9896まで広げて厳密解、1設定あたり十数分。

正規化は3段階。`plain` で句読点を落とすのは、日本語CERが句読点の選択に
支配されるうえ、anime-whisper は文末の「。」を省く傾向がモデルカードに
書かれているため（書式の癖で不当に減点される）。長音記号は音韻なので残す。

### 使い方

```powershell
# 素材の調査とWAV抽出（1回だけ。済んでいる）
python eval/prep.py sample --out _eval\wav

# 1回で1設定。--tag が出力フォルダ名になる
python eval/run.py --config ext-clip    -m large-v3 --tag clip-fixed
python eval/run.py --config ext-collect -m large-v3 --tag collect-fixed
python eval/run.py --config builtin     -m large-v3 --tag builtin-fixed
python eval/run.py --config ext-collect -m anime-whisper --tag anime-whisper

# 好きな設定を並べて採点。フォルダ名が設定名になる
python eval/score.py --hyp _eval\hyp\clip-fixed --hyp _eval\hyp\anime-whisper
```

`--config` は `builtin` / `ext-collect` / `ext-clip` / `novad` の4つ。
`novad` はVADが何を買っているのかの対照用。

`--only SUBSTR` / `--exclude SUBSTR` でファイルを絞れる。一挙を外すなら
`--exclude 2025032314`（ASCIIなのでシェルの文字化けを踏まない）。
`score.py` にも同じ2つがあり、既存のJSONから再採点するだけなので推論はやり直さない。

既存の出力はスキップするので中断しても再開できる。子プロセスには
`PYTHONIOENCODING=utf-8` を渡している（渡さないと日本語ファイル名がログで
cp932 化けする）。各実行のコンソール出力は JSON の隣に `.log` として残る。
`[VAD]` `[MODEL]` `[FIX]` 行が実際に効いた設定と修復量の記録。

実際に記録を作ったときのコマンド。

```powershell
# 24分もの9本（一挙を除外）
python eval/run.py --config ext-clip    -m large-v3 --tag clip-fixed    --exclude 2025032314
python eval/run.py --config ext-collect -m large-v3 --tag collect-fixed --exclude 2025032314
python eval/run.py --config builtin     -m large-v3 --tag builtin-fixed --exclude 2025032314
python eval/run.py --config novad       -m large-v3 --tag novad         --exclude 2025032314
python eval/run.py --config ext-collect -m anime-whisper --tag anime-whisper --exclude 2025032314
python eval/score.py --hyp _eval\hyp\clip-fixed --hyp _eval\hyp\collect-fixed `
                     --hyp _eval\hyp\builtin-fixed --hyp _eval\hyp\novad `
                     --hyp _eval\hyp\anime-whisper --report _eval\score-final.txt

# 歌・効果音注記の多い2本を外した「実運用で期待できる」条件
python eval/score.py --hyp _eval\hyp\clip-fixed --hyp _eval\hyp\novad `
                     --hyp _eval\hyp\builtin-fixed `
                     --exclude 2026050309 --exclude 2026030522 `
                     --report _eval\score-nosong.txt

# 一挙のみ
python eval/run.py --config ext-clip    -m large-v3 --only 2025032314 --tag clip-long
python eval/run.py --config ext-collect -m large-v3 --only 2025032314 --tag collect-long
python eval/score.py --hyp _eval\hyp\clip-long --hyp _eval\hyp\collect-long `
                     --report _eval\score-long.txt

# VADパラメータのスイープ。値が - で始まるので --extra= の形が必要
python eval/run.py --config ext-clip -m large-v3 --tag clip-t030 `
                   --extra=--vad_threshold --extra=0.30 --exclude 2025032314
```

ループ抑制の上限測定と実装検証は `eval/strip_loops.py`。推論はしない。

```powershell
# 採点用検出器（誤爆に甘い）で上限を測る
python eval/strip_loops.py --src _eval\hyp\clip-fixed --dst _eval\hyp\clip-noloop
# 実物の loop_filter.py を当てる
python eval/strip_loops.py --src _eval\hyp\clip-fixed --dst _eval\hyp\clip-prodloop `
                           --detector production
```

本家 Faster-Whisper-XXL と比べる場合。**`-ct float32` は必須**で、
これが無いと RTX 5090 では cuBLAS で落ちる（[測定結果 #10](MEASUREMENTS.md#10-faster-whisper-xxl-との比較既定同士推奨設定では未実施)）。

```powershell
$exe = "...\Faster-Whisper-XXL\faster-whisper-xxl.exe"
Get-ChildItem _eval\wav\*.wav | Where-Object { $_.Name -notlike "*2025032314*" } |
  ForEach-Object { & $exe $_.FullName -m large-v3 -ct float32 -f json `
                     -o _eval\hyp-xxl-raw --language ja --beep_off }

# 採点スキーマに変換（duration はXXLの出力に無いので既存JSONから借りる）
python eval/xxl_convert.py --src _eval\hyp-xxl-raw --dst _eval\hyp\xxl `
                           --duration-from _eval\hyp\clip-fixed

python eval/score.py --hyp _eval\hyp\clip-fixed --hyp _eval\hyp\builtin-fixed `
                     --hyp _eval\hyp\xxl --report _eval\score-xxl.txt
```

採点は全文CERの帯状DPが効くので、9本4設定で数分かかる。

---

## 既知の問題

### 1. ハルシネーション（ループは解決。残るのは「ループ形をしない余剰出力」）

`condition_on_previous_text=False` で大幅に改善し、**`loop_filter.py` で
繰り返し型は上限まで潰した**（[測定結果 #12](MEASUREMENTS.md#12-実装した検出器は上限にそのまま到達した誤爆ゼロで)）。既定で有効。

**残っている穴は3つ。**

**(1) 余剰出力がループ形をしていないケース。**
[測定結果 #11](MEASUREMENTS.md#11-ループを潰した場合の上限既定に載せるだけで本家を抜く) の novad 対照で見えたもので、クレバテスと終末ツーリングは
VADを外すと出力が増えるがその増分が繰り返しになっていないので検出器が拾えず、
精度が77.3〜86.5%に留まる。ここを扱うには内容ベースの判定が必要になる。

**(2) 句読点だけで構成されたセグメント。** `loop_filter.normalize()` は
`eval/normalize.py` の `plain` と同じ文字クラスを落とすので、
`. . . . .` のようなセグメントは折り畳むと空文字列になり、
`reason()` が判定に入る前に None を返す。**20.6秒のキューが素通りする。**

**(3) セグメントをまたぐ反復。** `loop_filter` は設計上セグメント単位で、
またぐぶんは `whisp_carrier.py` の `is_hallucination()` の担当。
そちらは `MAX_DUPES = 2` なので**連続3回目以降しか捨てず、対では残る。**
非連続の反復（間に別のセグメントが挟まる）は両方とも見ていない。

**(2) と (3) は [測定結果 #15](MEASUREMENTS.md#15-condition_on_previous_texttrue-は死亡遊戯を破壊した仮説の否定) で
`condition_on_previous_text=true` を試したときに露出した。既定（`false`）では
この縮退そのものが起きないので優先度は低い**が、`true` を将来検討するなら
先に塞ぐ必要がある。

**追加素材でも誤爆ゼロを維持した。** 子供向け4本で落としたのは
ニンジャラ444字・ぷにる267字で、プリキュアとアイプリは0件。抑制後の残存は
合計1件6字。一挙2本目（4h52m）でも29字だけ。**新しい素材群に当てても
挙動が変わらないことの確認になった**（#14）。

まだ試していない改善案。

- Purfviewの `--ignore_dupe_prompt` 相当の実装
- `--hallucinations_list` による既知パターンリスト
- セグメント内フレーズ繰り返し検出（実装したが誤爆が多く無効化した経緯がある。
  ただし**誤爆のCERコストは ±0.1pt** と実測済みなので、精度の理由で避ける必要はない。
  避ける理由は「悲鳴や言い直しという実在の発話が字幕から消える」ほうだけ）

**優先度は下がっている。** 本家に対する勝敗はすでに入れ替わっており、
これ以上のループ対策で得られる合計CERの余地は 0.1pt 台。

### 1-b. 子供向け番組の取りこぼし（**結論が覆った。VAD起因ぶんは直せる**）

> **この節の結論は [#18](MEASUREMENTS.md#18-ten-vad-が-silero-を置き換えた取りこぼしは実装で直せる) で覆された。**
> 「実装では直せない」の根拠は「別実装（XXL）でも同じ壁に当たる」だったが、
> **XXL も silero を使っている**ので、それは「silero の限界」を2回測っただけだった。
> TEN VAD に替えるとアイプリのカバレッジは 61.3% → **81.7%**、
> 死亡遊戯は 59.7% → **73.8%** に上がる。**以下は silero 前提での記録として読むこと。**
> 残り5本と長尺での確認が済むまで、どこまで直るかの最終的な量は未確定。

**silero を使う限りいちばん大きい精度の制約。** 子供向け番組では参照の3割前後を
「出せない」。カバレッジで見ると 47〜80% まで落ちる。

| 素材 | CER全文 | coverage | precision |
|------|--------|----------|-----------|
| おねがいアイプリ（2話とも） | 54% | **47%** | 86〜89% |
| 名探偵プリキュア！ #21 | 34.8% | 67.5% | 88.9% |
| ぷにるはかわいいスライム #20 | 30.6% | 72.1% | 87.5% |
| ニンジャラ（2話） | 21〜32% | 74〜80% | 83〜91% |

**精度は落ちていない**（86〜91%で会話劇と同水準）。出したものは合っていて、
出していないだけ。

**当時は「実装の問題ではない」と3方向から確認したつもりだった。
3つとも silero 前提だったのが盲点。**

1. **silero のパラメータ空間は打ち止め**（[#8](MEASUREMENTS.md#8---vad_threshold-は取りこぼしの-knob-ではなかった否定的結果)・[#9](MEASUREMENTS.md#9-vadパラメータ空間は打ち止め既定が最良だった否定的結果)・[#17](MEASUREMENTS.md#17---vad_neg_threshold-も-knob-ではなかったvad空間は本当に打ち止め否定的結果)）。
   threshold / min_silence / speech_pad / neg_threshold の4つすべてを振って既定が最良。
   **`neg_threshold` の区間数応答から検出側の失敗と確定し、
   それが「モデルを替える」という発想に繋がった**（[#18](MEASUREMENTS.md#18-ten-vad-が-silero-を置き換えた取りこぼしは実装で直せる)）
2. **本家XXLも同じ**（[#14](MEASUREMENTS.md#追加4本で-xxl-と直接比較した決定的)）。
   子供向け4本でカバレッジ 69.1% 対こちら 67.7%。**別実装が同じ壁に当たる。
   ここが誤りだった。XXL は `silero_v4_fw`＝同じ silero を使うので、
   「別実装」ではあっても「別モデル」ではない。同じ壁を2回測っていた**
3. **別話でも再現する**（アイプリ 47.3% → 47.0%）。素材個別の事故ではない。
   これは今も有効で、**作品の性質だという読みは正しい**（silero が苦手な音声）

**決着した（2026-08-22、[追加4本の TEN 測定](MEASUREMENTS.md#追加4本の-ten-測定2026-08-22決着)）。** 子供向け4本に TEN を当てると
合計 31.1% → **21.9%**、カバレッジ 70.3% → **82.1%**。4本すべてで silero に勝ち、
XXL にも 4勝0敗。**アイプリは 44.6% → 23.6%、プリキュアは 35.2% → 23.0%。**
この節の「実装では直せない」は完全に否定された。上の表は silero 前提の記録として読むこと。

**配布時の言い方も決まった。** 「会話劇なら10%前後、子供向けでも20%台半ば」。
30%を超えるのは死亡遊戯 #09（30.5%）だけになった。

**例外が1本ある。** 死亡遊戯 #09 は既定（clip）だと XXL に9.4pt負けていた
（69.1% 対 59.7%）。**ただしここは既に埋まっている。`novad` + ループ抑制で
カバレッジ 71.6% / CER 30.9% となり XXL を上回る**ので、
残っているのは発動条件だけ（[#15](MEASUREMENTS.md#15-condition_on_previous_texttrue-は死亡遊戯を破壊した仮説の否定)・
[次の着手 C](MEASUREMENTS.md#不要になったc-死亡遊戯の取りこぼし)）。
**つまり子供向けの3割取りこぼしとは別の現象**で、そちらは実装で埋まらない。

原因の仮説は未検証。子供向け番組の音声は BGM と効果音が密で、
叫び・歌・早口が多く、声も高い。silero がこれを発話として拾わないのか、
拾っても Whisper がデコードしないのかは切り分けていない。
**`novad` 対照を追加素材に当てれば「VADが落としているのか」は分かる**
（[#7](MEASUREMENTS.md#7-vadはカバレッジを売って精度を買う交換だったnovad対照)と同じ方法。24分もの4本で約10分）。

### 2. 音声フィルターが実用レベルに達していない

当初の症状：

- `--ff_lowhighpass` → 後半の音声が消える
- `--ff_vocal_extract mb-roformer` → アニメ声まで除去される
- 併用すると0セグメントになる

当初は「フィルター後の音声がVADの期待する特性と合わない」と説明していたが、
コードを読んで ffmpeg で実測したところ、**その前に順序バグが2つあった。**

**現状。順序バグ（2-a）は修正済み、段ごとの検査（2-b が要求していたもの）も
実装済み。残っているのは「`--ff_lowhighpass` で後半が消える」の再現と原因特定**で、
7秒のテスト音声では起きないため実素材が必要。`--ff_vocal_extract` の
「アニメ声まで除去される」はモデルの性質なので別の話（**そもそも exe では
使えない**。[該当節](MEASUREMENTS.md#audio-separator-を全ビルドから除外した)）。

**実運用は `--ff_*` を使わない方針で決着しているので、この経路は誰も通らない。**

#### 検査基盤（実装済み）

- **チャンネル選択を `extract_audio()` に統合した。** 独立した段として
  downmix の後に走っていたのが原因なので、同じ ffmpeg 呼び出しに入れて
  順序を保証した。`select_channel_fc` / `select_channel_left` / `invert_mix`
  の3関数は削除（順序は別段のままでは直せない）
- **セレクタをインデックス指定に変えた。** `c0=FL` → `c0=c0`、
  `c0=FL-FR` → `c0=c0-c1`。入力レイアウトのチャンネル名に依存しなくなった。
  FC だけは名前指定が必要なので残してあり、存在しない場合は下記の検査で止まる
- **`measure()` を追加。** `volumedetect` 1パスで peak / mean / duration を
  取る。ffmpeg の stderr から読むので ffprobe は要らない
- **段ごとに検査して `[FF]` 行で出す。** 無音化（peak <= -80dB）と
  長さの逸脱（±2%超）で `FilterStageError` を投げる。`--ff_tempo` と
  `--ff_silence_suppress` は長さが変わって当然なので長さ検査を外している
- **`_run()` が stderr を返すようにした。** 捨てていたので volumedetect の
  結果も警告も読めなかった
- **排他チェック。** `--ff_fc` `--ff_lc` `--ff_invert` は別のチャンネルを
  選ぶので同時指定をエラーにした（従来は3段が連鎖して無音になっていた）

多段チェーンの出力。これが 2-b の調査に必要だったもの。

```
  [FF] extract:     peak=-1.2 dB mean=-20.4dB dur=7.10s
  [FF] gate:        peak=-1.2 dB mean=-20.4dB dur=7.10s
  [FF] speechnorm:  peak=-0.4 dB mean=-16.2dB dur=7.10s
  [FF] loudnorm:    peak=-1.5 dB mean=-17.0dB dur=7.10s
  [FF] lowhighpass: peak=-0.7 dB mean=-17.1dB dur=7.10s
```

**`--ff_lowhighpass` が mean を 0.1dB しか動かさないことがログに出る。**
2-b の「ほぼ何もしていない」という手測の結論が自動的に見えるようになった。
`--ff_gate` も peak / mean が完全に不変で、この素材では何もしていない。

#### 2-a. チャンネル選択が無音を作っていた（原因の記録）

`audio_filter.preprocess()` はチェーン先頭で
`extract_audio(..., -ar 16000 -ac 1)` を走らせてから
`--ff_fc` / `--ff_lc` / `--ff_invert` を適用していた。モノラルに落ちた後の
`pan=mono|c0=FL` は入力レイアウトに FL が無いので完全な無音になる。

| 段 | mean | max |
|----|------|-----|
| 16kHz/モノラル変換直後 | -20.4 dB | -1.2 dB |
| `pan=mono\|c0=FL` 後 | -91.0 dB | -91.0 dB |
| `pan=mono\|c0=FL-FR` 後 | -91.0 dB | -91.0 dB |

全 113,640 サンプルが 91dB ビンに入る、つまりデジタル無音。
`--ff_fc` だけは偶然通っていた（モノラルの唯一のチャンネル名が FC なので恒等変換）。
VAD閾値をどう調整してもこの0セグメントは直らない。

**修正後。** ステレオ入力に対する `--ff_fc` は依然として無音になる
（ステレオのレイアウトは FL/FR で FC を持たない）が、
**段検査が peak=-91.0dB を検出して exit 1 し、原因まで表示する。**
5.1で正しく FC が取れる経路だけが未検証で、そちらは
`ffmpeg -af "pan=5.1|..."` で合成すれば確認できる。

#### 2-b. `--ff_lowhighpass` は今のパイプラインではほぼ何もしていない（実測）

`extract_audio` が先に 16kHz に落とすのでナイキストは 8kHz。
`lowpass=f=7800` は帯域上端をわずかに削るだけ。実測でも
mean -20.4 → -20.5 dB、max -1.2 → -2.1 dB。
**「後半の音声が消える」の原因は周波数特性ではない。**

**残っているのはここ。** `_run()` が returncode しか見ていないので、
**ffmpeg が 0 を返しつつ壊れた出力を書いた場合は依然として検知できない。**
段検査で peak と duration は見るようになったが、報告された症状（後半が消える）
そのものの再現が取れていないので、原因は特定できていない。
再発したらどの段で壊れるかログに残る。

#### 2-c. そのうえでのVAD自動調整（前提が崩れている）

`VadOptions` には `neg_threshold`（終話判定のヒステリシス、未指定時は
`max(threshold - 0.15, 0.01)`）がある。かつては `vad_parameters` に
threshold / min_speech / max_speech / min_silence / speech_pad の5つしか
渡していなかったので常に自動値だった。`loudnorm` や `speechnorm` は
全体のレベルを持ち上げるのでVAD確率分布も持ち上がる。効くのは threshold の
絶対値より **threshold と neg_threshold の間隔**という読みだった。

**`--vad_neg_threshold` は露出済み（両経路に配線）で、振った結果は否定的。**
0.01 まで下げても死亡遊戯のカバレッジは +1.1pt しか動かない
（[#17](MEASUREMENTS.md#17---vad_neg_threshold-も-knob-ではなかったvad空間は本当に打ち止め否定的結果)）。
**「間隔が効く」という読みも外れた。** この項を実装する前提は
もう1つ減っている。

**ただし「speech ratio を見て threshold を動かす」というルール自体は
[測定結果 #8](MEASUREMENTS.md#8---vad_threshold-は取りこぼしの-knob-ではなかった否定的結果) で否定されている。** 効かせたい相手（発話率の低い2本）に
効かず、効いた相手（ニンジャラ）は発話率が健全側だった。
**この項を実装するなら、まず発動条件を作り直すこと。**

実装自体は `whisp_vad_patch.py` の `external_segments()` を再利用できる
（VADをこちら側で先に走らせて結果を渡す仕組みは既にある）。

### 3. --realign が不安定

（字幕整形を破棄する問題はガードで対処済み。残っているのは二重ロードの無駄。）

stable-tsの `align()` がSRTのテキストを音声全体に引き伸ばすバグがある。
現在は `transcribe()` を再実行する形に変更したが、モデルを2つ目としてロードするため
VRAMと時間が二重にかかる。実用性は低い。

`--realign` は書き出し済みのSRTを stable-ts の出力で上書きするため、
`--standard` 等で整形した行組みを破棄してしまう。現在は整形オプションが
有効なとき realign 側をスキップして理由を表示する
（`[REALIGN] Skipped: incompatible with the subtitle formatting options`）。
SRTを出力していない場合も同様にスキップ理由を出す。

**推奨設定（`anime` プロファイル）は `standard_asia: true` なので、
この条件に常に該当する＝`--realign` は必ずスキップされる。**
だから配布は通常ビルドでよい。

### 4. exe化しても一部機能は動かない

`excludes` に入っているもの、および同梱していないもの。

| 機能 | 通常ビルド | フルビルド | 理由 |
|------|-----------|-----------|------|
| `--ff_vocal_extract` | 使えない | **使えない** | audio-separator を全ビルドから除外。同梱しても凍結環境で scipy の拡張モジュールが解決できず失敗する |
| `--realign` | 使えない | **動く** | `WHISP_CARRIER_FULL=1` で stable-ts を同梱。ただし推奨設定では必ずスキップされる |
| `-m anime-whisper` | exit 2 | **exit 2** | `transformers` は両ビルドで常に除外。回避策の案内を出して止まる |
| `--vad_method pyannote_v3` / `pyannote_onnx_v3` | 使えない | 使えない | pytorch-lightning と speechbrain を引き込むうえ、自前のテストで内蔵 silero より劣ると判明している |

いずれも凍結時は `pip install` を勧めても無意味なので、
`vad.py` の `_missing_backend` と `audio_filter._load_separator()` が
「この exe ビルドでは使えない」と分かる文面でエラーを出す。

`transformers` を除外した理由は容量ではない（実測381MB）。詳細は
[配布形態の切り分け](#配布形態の切り分けexe-が主経路スクリプト版が実験的機能)。

<details>
<summary>（解決済み・記録）PyInstallerでのexe化ができなかった問題の切り分け</summary>

かつての既知の問題 #4。**現在の `whisp_carrier.spec` は動く exe を生成する。**
原因の切り分け過程に再利用価値があるため記録を残す。

旧ビルドで実測したエラー（PyInstaller 6.11.1 / numpy 2.4.4）：

```
File "numpy\_core\__init__.py", line 24, in <module>
File "numpy\_core\multiarray.py", line 11, in <module>
ModuleNotFoundError: No module named 'numpy._core._exceptions'
ImportError: Importing the numpy C-extensions failed.
```

重要なのは、**hiddenimports は効いている**という点。
`build/<name>/PYZ-00.toc` を見ると `numpy._core._exceptions` は
`PYMODULE` として PYZ アーカイブに収録済みだった。
つまり収集漏れではなく凍結後のモジュール解決の問題で、
hiddenimports をこれ以上追加しても直らない。

最小構成では**再現しない**。`import numpy` と `import numpy._core._exceptions`
だけのスクリプトを onedir ビルドしたところ、`collect_data_files('numpy')` と
hiddenimports を手で足したもの、numpy 自身のフックに任せたもの、
どちらも正常に動いた。よって「numpy 2.4.4 と PyInstaller 6.11.1 の非互換」
という当初の説明は誤り。

もう一点、**壊れていた exe は現存しない `whisper_xl.spec` から
ビルドされたもの**で、`whisp_carrier.spec` とは別物だった。
`whisp_carrier.spec` で新規ビルドしたら動いた。

将来また同じ numpy エラーが出た場合の手順：

1. PyInstaller を最新に上げて再ビルド
2. `excludes` を一時的に空にして再ビルドし、除外の副作用かを見る
3. それでも駄目なら numpy 1.x に下げるか、Nuitka を試す

</details>

### 5. 宣言した動作環境の内側で落ちる経路（イシュー待ちにした・2026-08-23）

**README 導入の3行が盾になっている。**

> - Windows 10/11 (x64)
> - NVIDIA RTX GPU + CUDA 12.8 以上のドライバ
> - **Python も CUDA Toolkit も ffmpeg も不要です**（ffmpeg は同梱、LGPL版）

**非NVIDIA・非Windows・ドライバが古い、はこれで対象外に落ちる。**
しかも**ドライバ起因は実装側でも落ちない。** torch を外した結果
`CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0` になっているので、
0 なら `device="cpu"` / `compute="int8"` に落ちるだけでトレースバックにならない
（`-d cpu` を明示した経路は 0.9.1 exe で exit 0・字幕生成を実測）。
**つまり発火条件のうち一番広いものは、宣言と実装で二重に処理済み。**

> **ただし「動かない」を守っていて「異常に遅い」を守っていない。**
> CPU に落ちるとバナーに `device=cpu | compute=int8` が出るが、
> Amatsukaze 経由の利用者はバナーを読まないので、報告は「終わらない」の形で来る。
> **README に1行足せば閉じる**（コード修正ではない）。
>
> なお「古いドライバで `get_cuda_device_count()` が 0 を返す」は**コードから読んだ結論で
> 未実測。** `CUDA_VISIBLE_DEVICES=''` では ctranslate2 が無視して 1 を返したので
> シミュレートできなかった。**再現機を持っている人の報告が要る。**

#### 盾を貫通する4件（全員が宣言の内側にいるまま、`whisp_carrier.py:1128` の `WhisperModel(...)` で落ちる）

`WhisperModel(...)` は try/except に包まれていないので、**素のトレースバック +
`[PYI-*:ERROR] Failed to execute script 'whisp_carrier'` が出て exit 1** になる。

| | 発火条件 | 出るもの | 確認 |
|---|---------|---------|------|
| **a** | モデルのローカルパス誤り。README が `-m _models\ct2-litagin-anime-whisper-float16` を正式な使い方として載せているので、**推奨手順のタイポで踏む** | `ValueError: Invalid model size 'Z:\no\such\ct2-model', expected one of: tiny.en, tiny, ...`。**ディレクトリを渡したのに「モデルサイズが不正」と言われ、無関係なビルトイン一覧が出る。**実際の問題は「そのフォルダが無い」だけ | **0.9.1 exe で実測** |
| **b** | モデル取得の失敗（オフライン・プロキシ・社内網・リポジトリ名違い） | **約60行の二段トレースバックで、最後の行が `Invalid username or password.`** 公開モデルなのに認証エラーに見える | **0.9.1 exe で実測** |
| **c** | VRAM 不足。「NVIDIA RTX GPU」は **RTX 3050 Laptop 4GB も満たす**。large-v3 float16 は入らない | CTranslate2 の内側で落ちる | 未確認（再現機なし） |
| **d** | cuDNN のエンジンが世代で噛み合わない。spec は CUDA DLL を名前接頭辞（`_CUDA_KEEP`）で選抜しているので、`cudnn_engines_precompiled64_9.dll`（514MB）が sm_75 / sm_89 のカーネルを持つか、足りないぶんを同梱の nvrtc + nvJitLink が実行時コンパイルで埋めるか、**RTX 5090 以外で一度も確認していない** | 同上 | 未確認（再現機なし） |

**a と b は共通の前段がある。** どちらも直前に

```
[MODEL] cannot inspect 'Z:\no\such\ct2-model' (offline?), passing it to faster-whisper as is
```

が出る。**`whisp_models` は異常に気づいた上で faster-whisper に丸投げしている。**
ローカルパスなら「存在しない」、リポジトリIDなら「取得できない」と断定できる場所なので、
**直すときはここを `ModelError` にして既存の exit 2 に乗せるだけ**（`main()` の
`except whisp_models.ModelError` が既にある）。`-m anime-whisper` の案内と同じ形になり、
`_missing_backend` / `_no_conversion_error` と文体も揃う。**分岐1つとメッセージ2本。**
c と d は CTranslate2 の内側なので、`WhisperModel` の周りに1つ置くしかない。

#### なぜ先に直さずイシュー待ちにしたか

**理由は3つ。**

1. **[イシューテンプレート](../../tree/main/.github/ISSUE_TEMPLATE)が4件すべてを切り分けられる。**
   `01_bug_report.yml` は `log` を `required: true` で「**省略せず**貼ってください」と要求し、
   `environment` も `required: true` で `GPU（VRAM）` と `NVIDIA ドライバ` を明示的に聞く。
   `exit_code` のドロップダウンもある。**そして受け取る側にとって、トレースバックは
   整えたメッセージより情報量が多い**（例外型と行番号が付く）。
   「メッセージが汚いと切り分けられない」という反対理由は成立しない
2. **今日検証が通った成果物を捨てることになる。** SHA-256 が STATUS と一致、
   アーカイブに生 yaml も `_models` も無い、出力 MD5 が記録値とバイト一致、
   J の回帰が実機で通る、ffmpeg のハッシュが pin と一致 —
   **この状態を、J や `ctranslate2.converters` を生んだのと同じビルド工程に
   もう一度賭ける**ことになる。しかも[版番号の運用](#版番号とタグの運用スクリプト版だけの修正では版を上げない)で
   再ビルド＋再パック＋新 SHA-256＋版上げ＋リリース作り直しが付いてくる
3. **J は公開中の 0.9.0 に実在する不具合**（クリーン環境で非WAV入力が落ちる）で、
   その修正が 0.9.1 に入って検証済み。**確定した修正を、仮定のケースの
   メッセージ改善のために止めるのは損な取引。**

**待ちが唯一カバーしないもの：黙って帰る人。** 60行のトレースバックの末尾が
`Invalid username or password.` だと「アルファ版だ」と判断して報告しない可能性がある。
テンプレートは来なかった報告を集められない。**母数が小さいプロジェクトなので
1件の取りこぼしは重いが、この4件を踏む人数の期待値も1未満**という読みで待つ側に倒した。

**この節の使い方。** イシューが来たら上の表と突き合わせる。
**再導出せずに紐づけるためにここに置いてある。** 該当したら
[再パックの発動条件](#再パックの発動条件ドキュメントだけの変更では固め直さない)の
「コードの変更」に当たるので、版を上げて作り直す。

---

## 今後やるなら

優先度順ではなく、独立した候補として並べてある。
**[次の着手](STATUS.md#次の着手)の A（連携テスト）と B（配布アーカイブ）が終わるまで、
ここは手を付けなくてよい。** 1と2は着手不要の判断記録に変わった。

### 1. `--batched` は追わない（判断記録・2026-08-22）

**動く経路としては残すが、性能評価も最適化もしない。** 理由は4つで、
どれも「速くする価値がもう無い」側と「速くすると売りが削れる」側に分かれる。

**(1) 速度が制約になっていない。** 現行既定で 24分もの1本 70〜100秒（実時間の6%）、
一挙 5h22m が 1214秒（6.3%）、4h52m が 1338秒（7.6%）。
**Amatsukaze の運用ではエンコードのほうが長い**ので、ここを削っても体感が変わらない。
弱いGPUで2〜3倍遅くなっても「24分ものが3〜5分」で実用範囲。

**(2) バッチ化の余地がそもそも小さい。実測で確認した。**
追加4本を**2プロセス並列**で回したところ、1本あたり 162〜170秒だった（単体 70〜100秒）。
2本を162秒で処理していて、単体で順に回すと 170〜200秒。
**スループット利得は 5〜15% しかない＝1プロセスで既に GPU が飽和している。**
beam 5 はデコーダ内で既に5系列を並べているので埋める隙間が少ない。
バッチ化は総FLOPsを変えずに並べ方を変えるだけなので、飽和点では大きく伸びない。

> **厳密な上限ではない。** 2プロセスは重みを共有せずコンテキストスイッチも起きるので、
> 1プロセス内のバッチ化のほうが効率は良い。**「5090なら劇的に効く」という前提を
> 否定する材料**として読むこと。

**(3) こちらの差別化点を直接削る。** `BatchedInferencePipeline.transcribe` は
`vad_parameters` から `max_speech_duration_s` を pop して `chunk_length`（既定30）で
置き換え、`collect_chunks(..., max_duration=chunk_length)` で30秒までまとめ直す。
つまり **`--vad_max_speech_duration_s` は黙って無視される。**
売りは構造（30秒超0件・thin 0〜1件 対 XXL 9件602秒・15件）なので、
**30秒にまとめ直す機構はそこを攻める方向**。セグメントが粗くなる理由
（テスト音声で 3 → 1）もこれ。

**(4) フラグ1個では済まない。** batched の `clip_timestamps` は
**`List[dict]`（`{"start":…, "end":…}`）で、非batched のカンマ区切り文字列とは別物**
（`inspect.signature` と実ソースで確認）。現在のコードは文字列を渡しているので、
**TEN 区間の受け渡しから書き換えが必要**で、30秒超の区間をどう扱うかも未確認。
なお batched でも `clip_timestamps` を渡せば `vad_filter` は無視されるので、
外部VADの区間を通す道自体は閉じていない。

**再検討する条件。**

- CPU専用環境や大幅に弱いGPUからの要望が来て、実時間の6%が問題になる場合
- faster-whisper 側が batched でも同じ形で `clip_timestamps` を受けるようになり、
  区間の受け渡しが共通化できる場合
- `max_speech_duration_s` を尊重するようになった場合（構造の売りが守られる）

### 2. 推奨設定同士での本家比較（**決着。既に測っていた**・2026-08-22）

**「未測定」ではなかった。XXL の推奨＝既定なので、既定同士の比較がそのまま
推奨設定同士の比較になっている。** 調べた結果が3点。

**(1) 開発者は推奨プリセットを公表していない。**
[Purfview の README](https://github.com/Purfview/whisper-standalone-win/blob/main/README.md) にあるのは
「既定値は映画の書き起こし向けに調整済み」という趣旨の記述と、「モデルは medium 以上を使え」、
そして使用例に出る `--sentence` / `--standard` / `--batch_recursive` だけ。
**追加設定の推奨リストは無い**（内容はライセンス配慮のため要約している）。

**(2) XXL の既定は「素」ではない。** 手元の `_eval/_xxl_help_utf8.txt`（r245.4 の実物）より。

| XXL の既定 | 値 | こちら |
|-----------|-----|-------|
| `--vad_filter` | True | True |
| `--vad_min_silence_duration_ms` / `--vad_speech_pad_ms` | 3000 / 900 | 同じ（こちらが合わせた） |
| `--beam_size` / `--best_of` | 5 / 5 | 同じ |
| `--patience` | 2.0 | 同じ |
| `--condition_on_previous_text` | **True** | **False**（[#15](MEASUREMENTS.md#15-condition_on_previous_texttrue-は死亡遊戯を破壊した仮説の否定) で測って選んだ） |
| `--hallucinations_list_off` | False（**既知幻覚リストが既定で有効**） | 相当機能なし |
| `--v3_offsets_off` | False（**large-v3 用の擬似VAD閾値オフセットが既定で有効**） | 相当機能なし |
| `--reprompt` / `--ignore_dupe_prompt` | True / 有効 | 相当機能なし |
| `--ff_*`（音声フィルター全部） | **False** | 使わない方針 |
| `--standard` / `--standard_asia` | False | 推奨は `standard_asia: true` |

**つまり既定同士の比較は、こちらの既定 対「開発者がチューニングした既定」だった。**
ハルシネーション対策が3つ既定で入っている相手に勝っている。

**(3) 整形を揃えても全文CERは動かない。** `standard_asia` 等が足すのは改行とキュー分割で、
`eval/normalize.py` の `collapse_space()` が**両陣営の出力に対して** `\s+` を完全除去する。
だから「両者を推奨の整形で揃える」は全文CERに対して no-op で、
**やっても 16.1% 対 20.5% が再現するだけ。**

**残る差分はこちらが与えた favor 2つで、どちらもXXL不利にはなっていない。**

- `-ct float32` — RTX 5090 で float16 が cuBLAS で落ちるので不可避。同じ重みで
  **精度は float32 のほうが有利側**（代わりに約2倍遅い）
- `--language ja` — こちらでは[出力が1セグメントも変わらない](MEASUREMENTS.md#13-実運用設定beam-10は測定設定beam-5よりわずかに悪い)と確認済み

**本当に未測定なのは per-block CER だけ。** 整形は per-block を動かす（中点の割り当てが
変わる）ので、「両者を整形した per-block」は測っていない。ただし
**両陣営とも整形なしで揃っている**ので現在の比較は成立している。

**XXL 側の `--ff_*` と `--batched` は測らない。** どちらも既定オフで
開発者の推奨にも入っておらず、こちらはフィルターを使わない方針で決着しているので、
**XXL だけフィルター有利で回すのは非対称な比較**になる。

### 3. novad を活かす道と、VADログの不足

[測定結果 #11](MEASUREMENTS.md#11-ループを潰した場合の上限既定に載せるだけで本家を抜く) で **novad + ループ抑制の合計が 21.4%**（既定 22.0%）と出ている。
既定にできないのは、病的3本で大勝ちする代わりに健全2本で負けるため
（クレバテス +6.6pt、終末ツーリング +12.2pt）。この2本は novad の
余剰出力が**ループ形をしていない**ので検出器が拾えない。

つまり「素材依存の判断を1つのルールに落とす」問題で、
[既知の問題 2-c](#2-c-そのうえでのvad自動調整前提が崩れている) と同じ構造。

**[次の着手 C](MEASUREMENTS.md#不要になったc-死亡遊戯の取りこぼし) がここに合流した。**
死亡遊戯で XXL に負けている差は `novad` + ループ抑制で埋まる
（CER 30.9% / カバレッジ 71.6% 対 XXL 35.5% / 69.1%）ので、
**この項は「合計をもう少し良くする」話から「XXL に負けている最後の1本を消す」話になった。**

**#8 で否定されたのは「発話率で `--vad_threshold` を動かす」ルールであって、
「発話率で VAD を外す」ではない。** 否定の根拠は2つあり、どちらも状況が変わっている。

- 「閾値では取りこぼしが取り返せない」→ `novad` なら取り返せる（死亡遊戯 +11.9pt、アイプリ +25.9pt のカバレッジ）
- 「最大の利得が出たニンジャラは発話率49%で健全側」→ **ニンジャラはループ抑制で解決済み**なので、
  もう発話率トリガーで拾う必要が無い

**残る利得は発話率の低い2本（死亡遊戯 0.23、アイプリ 0.32）に集中していて、
損失は健全な2本（クレバテス・終末ツーリング、発話率0.6前後）に集中している。**
つまり分離できる可能性がある。**ただし校正点が2点しかないので、
[素材の項](#素材を足すべきか判断済み)のとおり低発話率の素材を足さないと過学習になる。**

先に測るべきは混成の数字（推論不要）。`strip_loops.py --detector production` で
`novad-prodloop` を作り、9本のうち2本だけ差し替えて採点すれば、
**ルールが成立した場合の上限が推論なしで出る。**

直す価値があるものが1つ残っている。

- **clip経路が発話秒数をログに出していない。** collect側は
  `[VAD] silero_v5: 517 chunks | 13530.5s of speech` と出すのに clip側は
  区間数だけなので、**speech ratio が運用ログから読めない**

（`--vad_neg_threshold` の露出は済み。振った結果は否定的で、
[#17](MEASUREMENTS.md#17---vad_neg_threshold-も-knob-ではなかったvad空間は本当に打ち止め否定的結果)。）

### 4. anime-whisper の扱い（決定済み・記録）

**エイリアスとして残すが非推奨。既定にはしない。** 24分もの9本で
**全9本 large-v3 に負けた**（全文CER 41.8% vs 24.3%、測定結果 #4）ので、
`--list_models` の note に測定値と「NOT RECOMMENDED for subtitles」を明記した。
残す理由はスコアではなく、非言語音の書き起こしが実際に違うものだから。

> **現行パイプラインで測り直した（[#22](MEASUREMENTS.md#22-anime-whisper-を現行パイプラインで再検証した4-の結論は維持)）。結論は維持。**
> ニンジャラ1本で **27.4% 対 50.7%（23.3pt 負け）**。
> **下に書いていた「救えるか試す手がかり」は両方潰れた。**
> 整形は per-block しか動かさないが**全文CERで既に負けている**し、
> 取りこぼし側は TEN VAD でカバレッジ +7.8pt 取れたのに**精度が 8.1pt 落ちて相殺**した。
> ループ抑制も効かない（製品版検出器はこのモデルの繰り返しをほぼ拾わない）。

以下は #22 より前の記録。もし救えるか試すなら手がかりは2つ、と書いていたもの。
セグメントが粗すぎる問題は `--standard_asia` を併用すれば整形側で吸収できるので、
整形ありの条件で再測定する価値はある。取りこぼし（長さ比0.80）は VAD側のパラメータで動く
可能性がある。**ただし17ptの差を埋める見込みは薄い。**

### 5. faster-whisper 本体へのパッチ

**`whisp_vad_patch.py` として部分的に実装済み。** 本体は触らずモンキーパッチで
済んだ。残っているのは silero v4 以前を載せるための代替モデルクラス
（`sr` 入力と状態形状が違う）と、`get_speech_timestamps()` 側の
512サンプル窓・ヒステリシスが v6 前提でチューニングされている点。
後者は上流に投げるなら
`get_vad_model(path=None)` + `VadOptions.vad_onnx_path` が最小差分になる。

### 6. モデルバックエンドの抽象化

現状 `transcribe_file()` は `faster_whisper.WhisperModel` に直結している。
返り値は `{id, start, end, text, words[]}` のリストに正規化されているので、
Whisper以外を第2バックエンドとして足す場合はここを差し替え点にできる。

Qwen3-ASR を候補にする場合、調べた範囲では設計上の前提が1つ崩れる。
公開されているのは `Qwen/Qwen3-ASR-1.7B-hf` / `Qwen3-ASR-0.6B-hf` /
`Qwen3-ForcedAligner-0.6B-hf` で、transformers に取り込まれている
（`AutoModelForMultimodalLM` + `AutoProcessor.apply_transcription_request`）。
日本語は ASR 側・アライナ側とも対応（アライナの ja は `nagisa` が必要）。
ただし **ASR 本体はタイムスタンプを一切出さない。** 出力は
`language <NAME><asr_text>...` というテキストだけなので、
セグメント境界はVADから、単語時刻は ForcedAligner から作ることになる。
`{start, end, words[]}` を満たすには「VAD → ASR → アライナ」の3段構成が必要で、
Whisper バックエンドと同じ形にはならない。CTranslate2 変換も無いので
`whisp_models.resolve()` の前提（変換先はCT2）にも手が入る。

ForcedAligner は単一forwardのNARでモデル非依存（Parakeet CTC の
書き起こしに当てる例が公式ドキュメントにある）。これは
[既知の問題 3](#3---realign-が不安定)（`--realign` の不安定さ）の代替として、
バックエンド抽象化より先に単独で入れられる。stable-ts の二重ロードより
軽い可能性がある。

出典: [Qwen3 ASR - transformers docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_asr)、
[Qwen3-ASR Technical Report](https://huggingface.co/papers/2601.21337)
（ライセンス上の制約に配慮して要約・言い換えしている）

---

## 開発環境再構築手順

```powershell
# Python 3.11 インストール（python.org から）
# CUDA Toolkit 12.8 インストール（developer.nvidia.com から）

pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# TEN VAD（--vad_method ten）。まだ requirements.txt に入れていない。
# 既定にするなら入れること（測定結果 #18）
pip install ten-vad==1.0.6.8
```

### 測定用の隔離環境（inaSpeechSegmenter / FSMN-VAD）

**主環境には入れられない。** inaSpeechSegmenter は Windows で解決できない
`tensorflow[and-cuda]` を、funasr-onnx は `numpy<=1.26.4` を要求する。
入れると記録されている全数値が再現できなくなる。

```powershell
# inaSpeechSegmenter（MIT）。[and-cuda] を避けるため依存を手で入れて --no-deps
python -m venv _venv_inass
.\_venv_inass\Scripts\python.exe -m pip install numpy pandas scikit-image `
    "pyannote.core" matplotlib Pyro4 pytextgrid soundfile onnxruntime tensorflow
.\_venv_inass\Scripts\python.exe -m pip install --no-deps inaSpeechSegmenter==0.8.0

# FSMN-VAD（MIT）。モデルは初回に ModelScope から落ちる
python -m venv _venv_fsmn
.\_venv_fsmn\Scripts\python.exe -m pip install funasr-onnx==0.4.2 modelscope

# 区間を JSON に出して、主環境から precomputed で読ませる
.\_venv_inass\Scripts\python.exe eval/ext_vad_dump.py --backend inass `
    --wav-dir _eval/wav --out _eval/vad-inass.json
python eval/run.py --config ext-json -m large-v3 --tag inass-clip `
    --extra=--vad_segments_json --extra=_eval/vad-inass.json
```

`transformers` を入れると `tokenizers` が 0.23.1 から 0.22.2 に落とされるが、
faster-whisper 1.2.1 の要求は `tokenizers>=0.13,<1` なので問題ない。

exe のビルド。

```powershell
# 通常ビルド（配布用）
python -m PyInstaller whisp_carrier.spec --noconfirm

# フルビルド（--realign の検証用。通常ビルドを上書きしない）
$env:WHISP_CARRIER_FULL='1'
python -m PyInstaller whisp_carrier.spec --noconfirm --distpath dist-full --workpath build-full
```

ビルド時は spec が ffmpeg のライセンスを検査するので、
`[spec] ffmpeg verified (known LGPL build)` が出ることを確認する。
GPL / non-free ビルドを掴んだ場合は SystemExit で止まる。

以上。
