// The `--help` text, kept in its own module so that changing a line of prose
// does not touch the code that parses the flags it documents.

export const USAGE = `mjgame — 雀鬼流ルールの4人麻雀 (人間1 + CPU3)

  deno run --allow-read --allow-write src/main.ts <command> [options]

コマンド:
  play       半荘を1回プレイする (端末UI)。手牌の下の「助言」行は、同じ盤面を
             見た k席 (計算) の打牌選択。d キーで候補の評価順を表示
  selfplay   CPU同士の対局を回して着順統計を出す
  paired     同一牌山で --seats と hhhh を2回ずつ回し、席0の対応差を出す
  bench      同上、半荘/秒だけを出す

オプション:
  --seed=N            乱数シード (再現用)
  --games=N           selfplay / paired / bench の対局数 (既定 100)。
                      paired では1シードにつき2半荘なので実対局数は2倍
  --glyphs=ascii      CJKフォントが弱い端末向けの2桁ASCII表記 (既定 kanji)
  --speed=MS          play でのCPU1手あたりの待ち時間 (既定 350)
  --timer=10+3        持ち時間: 半荘通しの持ち時間10秒 + 毎打3秒 (既定)。
                      毎打の3秒を超えると持ち時間を消費し、使い切ると
                      マイナス表示になる。表示だけの目安で、打牌は強制されず、
                      遅くても罰則は一切ない
  --seats=hrrn        CPUの種類: h=凍結基準席 (2026-08-25 時点の既定 k席の凍結
                      コピー。設定は一切受け付けない), r=ランダム,
                      n=学習済みニューラルポリシー, o=オラクル増補,
                      k=計算 (公開情報だけの組合せ読み) (既定 hhhh)。
                      短く書くと最後の文字を繰り返す ("hr" ⇒ "hrrr")。
                      selfplay/bench では席番号ごと。play では人間の席を
                      飛ばして先頭3文字を席順に割り当てるので、"nhhh" なら
                      必ずAI(学習済み)が1人入る。n のCPUは AI東 のように表示。
                      o は隠蔽情報 (他家の手牌・山) を直接読むので headless 専用。
                      k は隠蔽情報を一切見ない (スジ・カベ・現物・見えている枚数・
                      副露・リーチ・巡目だけを数える) ので play でも使える
  --plan              k席で最大利益ロックオン立案 (C7) を有効にする (既定 無効)。
                      o席の立案は --oracle=C7O/C7P 側で指定する
  --standings         A腕席0で順位効用レイヤを有効にする (順位分布モデルで押し引きを
                      尺度化)。持ち点・局数・供託・本場という公開情報だけから
                      最終着順分布を閉形式で解き、和了の値打ちと放銃の代償を
                      平場 (全員25000) 比の2つの倍率にして押し引きに掛ける。
                      雀鬼流の補正として「順位−1の仮想プレイヤー」を常に自分の
                      8000点上に置くので、独走トップでも打つのを止めない。
                      席0の k席に効き、対照の B腕 (hhhh) には決して渡らない
                      (h席は凍結済み・2026-08-25)。selfplay / bench / paired 専用
  --oracle=C1,C2,C3   o席が読んでよい情報チャネル (既定 C1,C2,C3)。
                      C1=放銃真値 C2=聴牌真値 C3=打点真値 C4=次のツモ
                      C5=次の槓ドラ C6=リーチ者の次のツモ。none で全部切る
                      (= h と完全に同一の打牌になる対照群)
                      C7O/C7P=最大利益ロックオン立案 (完成形を列挙し
                      P(完成)×打点 が最大の一つに狙いを固定する)。
                      C7O は山の残り牌構成と他家の手牌を真値で読み、
                      C7P は誰でも数えられる未見枚数だけを使う対照群
  --noise=E           オラクルの劣化度 0..1 (既定 0)。1判断ごとに、情報の
                      グループ (放銃/聴牌/打点/ツモ/ドラ/リーチ者ツモ/残り枚数)
                      それぞれを独立に確率Eで落とす。落ちたグループは「無い」
                      扱いになり、その項だけ手作り評価関数の推測に戻る。
                      E を振ると「どこまで読みが粗くなると優位が消えるか」が測れる。
                      E=1 では C7O が C7P と同じ挙動まで落ちる (立案機構は残る)
  --curriculum=E      A腕席0 (k席) の読みを「オラクル→計算」のカリキュラムにする。
                      1判断ごとに情報グループを確率Eで落とし、落ちた分は
                      「無い」ではなく計算 (公開情報だけの読み) の答えで埋める。
                      E=0 は純オラクル席と、E=1 は素の k席とビット単位で同一。
                      --oracle= で読ませるチャネルを選ぶ。selfplay/bench/paired 専用。
                      学習用: 消費曲線を鍛えるとき、読みの精度だけを連続に劣化させる
  --table=PATH        卓の完全な記述: 4席ぶんの SeatSpec を並べた JSON。席ごとに
                      kind (h/k/o/n/r) と、その席だけの ktune (path か inline)・
                      plan・standings・consumer (path)・curriculum・weights・temp
                      を書く。ktune 等のパスは table file からの相対。例:
                        {"seats": [
                          {"kind":"k", "ktune":"champion.json", "plan":true},
                          {"kind":"k", "ktune":"other.json"},
                          {"kind":"h", "ktune":{"hand":{}}},
                          {"kind":"h"} ]}
                      同じ部品の別重み・別構成を4席に同時に座らせられる (モジュラー
                      構成の本形)。--seats/--ktune/--ktune-opp/--plan/--standings/
                      --consumer/--curriculum/--weights/--temp とは併用不可 (卓が
                      全てを決める)。--oracle/--noise/--record 等の配線は併用可。
                      selfplay / bench / paired 専用。paired では --table-b も必須
                      (対照腕を暗黙の hhhh にしない — 環境一致検査のため)
  --table-b=PATH      paired の対照 (B腕) を明示的な卓にする。--table が必要で、
                      席1-3 (環境) は --table と完全一致しなければ拒否される —
                      両腕は席0だけが異なる、が対照実験の定義 (M11 の交絡の教訓)。
                      --ktune-b / --consumer-b とは併用不可
  --ktune=PATH        k席の感性ベクトル {heuristic, augment, computed, hand,
                      riichi, sense, fold, dealin, ev} のJSON。k席だけに効く
                      (h席は凍結済み・2026-08-25 — hand/riichi の決定モデル節も
                      k席専用になった)。ev (M15の期待値核) は libmjev が必須で、
                      ev.discard は consumer/hand/fold/--foldcalib と、
                      ev.riichi は riichi 節と併用不可 (置き換える側なので)。
                      paired では A腕の k席だけに効き、対照の B腕 (hhhh) には
                      決して渡らない。play では k席と助言席 (既定は
                      weights/champion.json) に効く。
                      scripts/tune.ts が書き出す形式
  --ktune-opp=PATH    相手3席 (席1-3) の感性ベクトル。席0の --ktune とは別の file を
                      積めるので、二つ目の対戦相手集団を組める (当てはめた
                      パラメータが他の相手にも効くかを測るのに要る)。ベクトルは
                      その席の完全な記述なので hand 節も相手側の file のものになる。
                      未指定なら従来どおり全席が --ktune を共有 (ビット単位で同一)。
                      paired では相手は「環境」なので A腕・B腕の両方に同じものが
                      渡り、--ktune-b でも上書きされない。
                      selfplay / bench / paired 専用
  --ktune-b=PATH      paired の対照 (B腕) にも感性ベクトルを積む。B腕は hhhh ではなく
                      A腕と同じ席種・同じ読み・同じ曲線のまま、--ktune の file だけ
                      こちらになる。つまり測るのは「候補 − 現行」であって
                      「候補 − 素の hhhh」ではない。--consumer-b のスカラ版で、
                      小さな摂動は大半の局をビット単位で不変に保つので、同じシード数で
                      桁違いに細かい差が読める (探索器の分散削減)。--consumer-b とも
                      併用でき、その場合 B腕は自前の感性ベクトルと自前の曲線を持つ。
                      paired 専用
  --consumer=PATH     M9: 打牌評価の「消費」を単調曲線に差し替える (証拠ベクトルの
                      名前つき素性 → 4節点の区分線形写像17本)。selfplay / bench /
                      paired の席0だけに効き、対照の B腕 (hhhh) には決して渡らない。
                      計算 (証拠の作り方) は一切変えない — 変わるのは消費だけ。
                      scripts/consumer_init.ts が書き出す初期値は現行の手書き評価と
                      ビット単位で同一なので、--consumer=weights/consumer-init.json
                      を渡した paired は必ず全局同着になる (これが健全性検査)。
                      --ktune と併用可 (危険度の梯子や降り倍率は感性側が決める)
  --consumer-b=PATH   paired の対照 (B腕) にも曲線を積む。B腕は hhhh ではなく
                      A腕と同じ席種・同じ --ktune・同じ読みのまま、曲線だけ
                      この file になる。つまり測るのは「候補 − 現行」であって
                      「候補 − 素の hhhh」ではない。小さな摂動は大半の局を
                      ビット単位で不変に保つので、同じシード数で桁違いに
                      細かい差が読める (探索器の分散削減)。paired 専用
  --calibrate=PATH    M10a: 席0 (k席) の1判断ごとに「計算の予測」と「真値」を
                      対にした較正記録を JSONL で書き出す。selfplay / paired 専用で、
                      paired では A腕だけ。打牌は一切変わらない (記録は out-param で、
                      席が読む Reads は素の計算のまま — だから記録あり/なしで
                      全局ビット単位で同一になる)。
                      1行1判断: 他家3人ぶんの聴牌確率・待ちの形の素の枚数 (パラメータ
                      非依存の整数) ・副露の内容読み・打点の材料と、真値の聴牌/ロン牌
                      集合/打点。パラメータを変えた再評価は再対局なしで閉じた式で
                      できる。読むのは scripts/calibrate_report.ts。
                      (dealin ブロックとは併用不可 — レーンは素の計算の読みの上で録る)
                      1半荘あたり約220KB (判断190行) — 出力先は作業用ディレクトリに
  --handcalib=PATH    M11: 席0 (k席 か h席) の自摸番ごとに「手牌価値の読み」と
                      「その局の結末」を対にした較正記録を JSONL で書き出す。
                      --calibrate の裏返し — あちらは他家3人について読んだことを、
                      こちらは自分の手について信じたことを採点する。
                      selfplay / paired 専用で、paired では A腕だけ。打牌は一切
                      変わらない (記録は out-param)。
                      1行1自摸番: 選んだ13枚形の HandFacts (向聴・受入・未見枚数・
                      ドラ・役牌・他家の聴牌読み …すべてパラメータ非依存) と
                      handOutlook の答え、そして局の結末 (和了/放銃/流局・実収支・
                      終局巡目)。札は局単位なので記録は局末までバッファされる。
                      1半荘あたり約30KB (自摸番120行)。
                      読むのは scripts/hand_report.ts、当てはめは scripts/hand_fit.ts。
                      --jobs とは併用不可
  --foldcalib=PATH    M13: 席0 (k席) の押し引き判断ごとに「ヘッドが読む37個の特徴量」
                      「席の規則が出した判定」「実際に打った側」「局の収支」を
                      対にした記録を JSONL で書き出す。selfplay / paired 専用で、
                      paired では A腕だけ。
                      --calibrate / --handcalib と違い、これは当てはめる先が
                      教師データではなく反実仮想 — 降りていたらどうだったかは
                      局の結末に書いてないので、--fold-eps で判定をわざと
                      ひっくり返して両側を打ち、傾向スコア (p) と一緒に記録する。
                      報酬は局の deltas[0] ただ一つ (道場の違反数 vio0 は
                      データとして載るだけで目的関数には入らない)。
                      --fold-eps なしなら打牌は一切変わらない (乱数も引かない)。
                      読むのは scripts/fold_report.ts、当てはめは train/fold_fit.py。
                      --jobs とは併用不可
  --evcalib=PATH      M15b: 席0 (k席) の自摸番ごとに「EV核が読む13枚形の全入力
                      (ints/dbls の wire そのもの)」と「DEFAULT_EV の下での
                      P(聴牌)/P(和了)/E[打点]/E[放銃コスト]」、そして局の結末を
                      対にした記録を JSONL で書き出す。selfplay / paired 専用で、
                      paired では A腕だけ。打牌は一切変わらない (記録は out-param —
                      席は核を持たず、wire を組んで書き手に渡すだけ)。
                      当てはめは母集団スカラー (ronFactor / oppHazard / oppGrowth /
                      dealinRate / tsumoShare / foldHazard) で、読むのは
                      scripts/ev_fit.ts。
                      ev ブロックとは併用不可 — レーンは EV核 を積んでいない
                      素の席で録る (自分の降りで打ち切った局は当てはめに使えない)。
                      1行あたり約1.7KB (wire が528数値) なので 1半荘 約220KB
                      (2000半荘のレーンで 448MB — 作業用ディレクトリに)。
                      --jobs とは併用不可
  --fold-eps=X        M13: 押し引きの判定を確率 X (0<X<1) で反転する。席が持つ
                      専用の乱数列から1判断につき最大1回引く — 対局用の乱数
                      (--epsilon) には触らないので、シードと打牌の対応は保たれる。
                      X=0.05 が既定の目安。--foldcalib と併用必須
  --export=PATH       打った半荘を天鳳形式の牌譜XMLで書き出す (play / selfplay 専用)。
                      PATH は拡張子なしの基底名 (.xml で終えればそのまま使う)。
                      同じ基底名で .mjgame.json も並べて書く — 天鳳XMLに載らない
                      赤5筒2枚目の区別と道場の違反台帳がこちら。
                      selfplay で --games=N>1 なら PATH-0001.xml … と連番、
                      N=1 なら連番なし。姉妹ツール ../mjrender にそのまま渡せる:
                        cd ../mjrender && deno task render ../mjgame/PATH.xml
  --jobs=N            selfplay を N 個の Worker で並列に回す (既定 1 = 逐次)。
                      i 番目の半荘 (seed+i) を i%N 番の Worker が打ち、結果は
                      Worker の終わった順ではなく必ず対局順に並べ直される。
                      軌跡JSONL・--export の牌譜・表示される表はすべて
                      --jobs=1 とバイト単位で同一 (違うのは所要時間の行だけ)。
                      --games より大きい N は --games に丸める。
                      --record / --export とは併用可、
                      --calibrate / --handcalib / --foldcalib / --evcalib とは併用不可
  --json              paired の結果を1行のJSONで出す (表の代わり)。
                      scripts/tune.ts が読む機械可読出力
  --weights=PATH      n席が読む manifest.json (既定 weights/manifest.json)。
                      読めなければ起動時にエラー — trainer か train/randinit.py で作る
  --temp=T            n席の方策温度。0=決定的(既定)、1=PPO自己対戦のサンプリング。
                      正の値なら合法手のソフトマックスから席ごとの乱数で1手引く。
                      selfplay / bench / paired 専用 (play の n席は常に決定的)
  --record=PATH       selfplay の全判断を軌跡JSONL (trajectory) に書き出す。
                      1行1判断 ("d") + 局結果 ("r") + 半荘結果 ("m")。学習器の入力。
                      "d" 行には非対称critic用のオラクル情報 (他家3人の手牌・
                      残り山・裏ドラ = "o"、他家の向聴数 = "sh") も必ず入る。
                      1判断あたり向聴計算が3回増える分だけ遅くなる (推論側は不使用)
  --record-all        n席以外の判断も記録する (BC教師データ用)。ppo.py には
                      渡せない — 挙動方策が --init と一致する前提が壊れる
  --no-intro          開始演出と配牌アニメを飛ばす
  --help, -h          このヘルプ
`;
