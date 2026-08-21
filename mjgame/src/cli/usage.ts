// The `--help` text, kept in its own module so that changing a line of prose
// does not touch the code that parses the flags it documents.

export const USAGE = `mjgame — 雀鬼流ルールの4人麻雀 (人間1 + CPU3)

  deno run --allow-read --allow-write src/main.ts <command> [options]

コマンド:
  play       半荘を1回プレイする (端末UI)
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
                      マイナス表示になる。打牌は強制されず、遅さの代償は
                      雀鬼流の長考ペナルティのみ
  --seats=hrrn        CPUの種類: h=手作り評価関数, r=ランダム,
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
                      h席・k席に効き、対照の B腕 (hhhh) には決して渡らない。
                      selfplay / bench / paired 専用
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
  --ktune=PATH        k席の感性ベクトル {heuristic, augment, computed} のJSON。
                      selfplay / bench / paired のみ。paired では A腕の k席
                      だけに効き、対照の B腕 (hhhh) には決して渡らない。
                      scripts/tune.ts が書き出す形式
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
                      1半荘あたり約220KB (判断190行) — 出力先は作業用ディレクトリに
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
                      --record / --export とは併用可、--calibrate とは併用不可
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
