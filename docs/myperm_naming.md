# myperms effect naming

`myperms` の内部キーは `MypermKey(effect_name, transform_index)` とする。
従来の手順名はaliasとして保持し、旧名からも新しいキーを解決できる。

## 基本形式

```text
<Part><Count><Kind>[<MappingsOrCycles>][+...]
```

Part code:

- `C`: Corner
- `E`: Edge
- `ME`: Rubik's Cube / Master Pyraminx などの MidEdge
- `W2`, `W3`: Rubik's Cube の 2列目 / 3列目 Wing
- `EAll`: 同じ辺のMidEdgeと全Wingが同じ移動・向きを持つEdge bundle
- `OE`: Master Pyraminx の Outer Edge
- `CtrX`, `CtrPlus`, `CtrObl`, `CtrCore`: Rubik's Cube の Center family
- `CtrBar`: Rubik's Cube の Center bar。`CtrX` / `CtrPlus` / `CtrObl` が同じbar単位で動く場合に統合する
- `CtrMidBar`: Rubik's Cube の middle Center bar。`CtrPlus` が複数深さで同じbar単位に動く場合に統合する
- `Ctr`, `CtrA`, `CtrB`: その他 puzzle の Center / Center orbit

Operation:

- `3[A>B>C]`: A のパーツが B、B が C、C が A へ移る3-cycle
- `4s[A<>B;C<>D]`: 4パーツ、2組の交換
- `8p[3+5]`: 8パーツが3-cycleと5-cycleを作る
- `96p[2x24+4x12]`: 2-cycleが24個、4-cycleが12個
- `3[UF>LU>RU]`: Edge / MidEdge の向き込み状態 cycle
- `3[DB>BD;UF>FU;...]`: cycleに潰せない向き変化を含む mapping
- `2[URF>RFU;UBL>BLU]`: Corner orientation を循環順序で表す

位置の列挙は移動パーツ数6以下に限定する。それより大きい場合は、個数とcycle構造だけを表示する。
複合作用を含む名前全体が160文字を超える場合も位置を省略し、詳細は解析データ側に保持する。

## Orientation

解析は色ではなく、各ステッカーに付けた一意なIDで行う。同色のWingやCenterが複数存在しても移動元を失わない。

- Wingは `DB@2L` の位置だけで向きが一意に決まるため、追加の向き記号を付けない
- MidEdgeは `DB` と `BD` を別の向きとして扱い、可能なら `ME3[UF>LU>RU]` のように向き込み状態のcycleで表す
- Cornerは `URF`, `RFU`, `FUR` のような循環順序で向きを表す
- 移動と向きの変化は `source>oriented_destination` で同時に表す
- 全Edge bundleに共通する反転は `XY>YX` と短縮できる

例:

```text
C3[UBR>UFL>URF]
C4s[UBR<>URF;UFL<>ULB]
E2[UL>LU;UR>RU]
ME3[UF>LU>RU]
ME2[UB>BU;UF>FU]
ME4s[UB<>UF;UL<>UR]
ME4[DB>DF;DF>FU;UB>BU;UF>DB]
C2[UBR>BRU;ULB>BUL]
W2-2s[RF@U<>UF@R]
W2-3[DB@R>UF@R>UB@R]
W2-4[UL@B>UR@B>UL@F>UR@F]
W2-4s[UL@B<>UR@F;UL@F<>UR@B]
W2-6p[3x2][BR@D>LB@U>FL@D;BR@U>LB@D>FL@U]
W2-6[BR@D>LB@U>FL@D>BR@U>LB@D>FL@U]
EAll12[XY>YX]
CtrX8p[4x2]+W2-2s[FL@U<>UF@L]
CtrX3[F@2R.2U>U@2R.2F>R@2F.2D]
CtrBar3[F@2L>U@2R>U@2L]
CtrBar4s[D@2L<>U@2F;D@2R<>U@2B]
CtrMidBar6p[3x2][F@D>U@F>F@L;F@R>F@U>U@B]
CtrCore4[D>L>U>R]+ME2s[UL<>UR]
C2[UBR>RFU]+CtrCore4[D>L>U>R]
C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]
CtrPlus12p[3x4]+ME5[DF>FU>FR>LF>BL]
```

## Position notation

- Rubik's Corner: `UFL`, `UBR`
- Rubik's Edge / MidEdge: `UF`, `BR`
- Rubik's Wing: `W2-... UF@R`（`W2` が2列目、`@R` が辺上の位置）
- Rubik's Center: `U@2L.2F`
- Skewb/Pyraminx/Megaminx: 接しているface名
- FTO/CTO Corner: `U`, `D`, `F`, `B`, `L`, `R`
- FTO/CTO Edge: `UF`, `DL` など
- Pyraminx/MasterPyraminx Center: `U@L`, `U@C` のように面名と寄っている頂点。`@C` は中央寄り
- MasterPyraminx OuterEdge: `UL@R`, `UL@B` のように辺名と寄っている頂点。同じ辺上の2つを区別する
- Skewb Center: `U`, `R` のように面名
- FTO Center: `URF@F` のようにFTO face名と寄っている頂点

Rubik's Cubeで同じ辺のMidEdgeと全Wingが同じ辺へ移動し、向きも一致する場合は、個別の `E` と `W` を `EAll` に統合する。
例えば7x7 SuperFlipはMidEdgeとWingを分離せず、`EAll12[XY>YX]` とする。

Rubik's Cubeで `CtrX` / `CtrPlus` / `CtrObl` が同じCenter bar単位で動く場合は、個別のcenter familyを連結せず `CtrBar` に統合する。
例えば `OuterCenterBar-A` は `CtrBar3[F@2L>U@2R>U@2L]` とする。
`F@2L` は F面の `2L` 列にあるcenter bar全体を表し、`CtrX`, `CtrPlus`, `CtrObl` の各深さは省略する。

Rubik's Cubeで `CtrPlus` だけが複数深さにわたって同じmiddle bar単位で動く場合は、`CtrMidBar` に統合する。
例えば `MidCenterBar(VV)` は `CtrMidBar6p[3x2][F@D>U@F>F@L;F@R>F@U>U@B]` とする。
`F@D` は F面のD側 middle center barを表し、`2D` / `3D` などの深さは省略する。

CenterとMidEdge、CenterとCornerが同時に動く手順は、effect componentを `+` で連結する。
例えば `CenterMidEdgeSwap-QA` は `CtrCore4[D>L>U>R]+ME2s[UL<>UR]`、`CenterCornerSwap-A00` は `C2[UBR>RFU]+CtrCore4[D>L>U>R]` 系になる。

Commutator系も同じくeffect componentを `+` で連結する。
ただし `OutCommutator` などはサイズによって存在するWingが変わるため、3x3では `E3[...]`、7x7では `EAll3[...]` のように正規化先が変わる。
source keyは大きいキューブ側の情報量が多い表記を置き、各サイズの初期化時に実効果へrenameする。

一意なCenter IDによる同一face内の入れ替えは詳細解析には残すが、通常の色状態では観測できないため短縮名から除外する。

## Name collisions and migration

同じ効果を持つ別手順、または位置を省略した大規模手順は同じ短縮名になることがある。
その場合だけ `~v01`, `~v02` を付ける。従来名は `alias` として保持する。

`effect_name` は代表変換 `#00` の効果から生成し、`#NN` はその対称変換番号とする。
特定の `#NN` における実位置が必要な場合は `MypermEffectAnalyzer` で変換後のmove列を解析する。

登録元ソースも可能な範囲で新命名に寄せる。
旧ソース名との対応が必要な場合は `self._add_myperm2("新名", moves, legacy = "旧名")` の形で同じ行に残す。

`myperms` に一致しない探索手順の `last_perms_key` は、その手順を直接解析し、`LP:` を先頭に付けた効果名を生成する。
例えば `LP:C2[UBR>BRU;ULB>BUL]` のように表示する。

## Representative transform points

`Points.txt` は、同じmyperm系列のどの対称変換を代表 `#00` とするかを決めるための位置スコア定義として扱う。
point計算は、原則として移動したパーツのsource positionを加算する。
Cornerは `UFR` と `URF` のような向き違いを同じ物理位置として扱う。
MidEdgeは `UF@M` を `UF` に正規化し、Wingは `UF@2R` / `UF@3R` を `UF@R` に正規化する。
辺位置のpoint lookupは物理辺として扱うため、`RF@U` は `FR@U`、`LB@D` は `BL@D` のように反転辺ラベルでも同じ点数を参照する。
Centerはプログラム側の `CtrX` / `CtrPlus` / `CtrObl` と `Points.txt` の `XCenter` / `PlusCenter` / `ObliqueCenter` を対応させる。
Center座標は実装側と `Points.txt` 側で軸順が逆になる場合があるため、`R@2U.2F` と `R@2F.2U` は同じpoint entryとして扱う。
短縮名から除外している同一face内Center permutationは、既定ではpoint計算からも除外する。
通常の `Rubiks_3` 初期化では起動コストを避けるためpoint reindexを自動実行しない。
検証や段階的なsource移行では、`Rubiks_3(size = 7, PointReindex = ("source-name", ...))` のように対象myperm系列だけを指定して、point最大のtransformを `#00` に再割当する。
全系列を対象にする場合は `PointReindex = True` を指定できるが、全transformのpoint計算が必要なため重い。
effect解析や局所レポート生成だけで固定myperms registryが不要な場合は、`Rubiks_3(size = 7, RegisterMyperms = False)` で SingleMove/Rotate 以外の myperms 登録と transform 展開をスキップできる。
`tools/generate_myperm_point_report.py` はこの軽量初期化を使い、`--name-prefix` 指定時は対象系列だけを transform 展開する。

提案一覧は次のコマンドで再生成できる。

```bash
python3 tools/generate_myperm_name_report.py
```

既定では `reports/myperm_name_proposals.csv` に、旧名、現在名、提案名、移動数、向き変化数、手数、move列を出力する。
