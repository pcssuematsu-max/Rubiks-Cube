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
C2[UBR>BRU;ULB>BUL]
W2-2s[RF@U<>UF@R]
EAll12[XY>YX]
CtrX3[F@2R.2U>U@2R.2F>R@2F.2D]
```

## Position notation

- Rubik's Corner: `UFL`, `UBR`
- Rubik's Edge / MidEdge: `UF`, `BR`
- Rubik's Wing: `W2-... UF@R`（`W2` が2列目、`@R` が辺上の位置）
- Rubik's Center: `U@2L.2F`
- Skewb/Pyraminx/Megaminx: 接しているface名
- FTO/CTO Corner: `U`, `D`, `F`, `B`, `L`, `R`
- FTO/CTO Edge: `UF`, `DL` など
- FTO Center: `URF6` のようにface名とface内index

Rubik's Cubeで同じ辺のMidEdgeと全Wingが同じ辺へ移動し、向きも一致する場合は、個別の `E` と `W` を `EAll` に統合する。
例えば7x7 SuperFlipはMidEdgeとWingを分離せず、`EAll12[XY>YX]` とする。

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

提案一覧は次のコマンドで再生成できる。

```bash
python3 tools/generate_myperm_name_report.py
```

既定では `reports/myperm_name_proposals.csv` に、旧名、現在名、提案名、移動数、向き変化数、手数、move列を出力する。
