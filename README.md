# Tool for verify Tenhou's mahjong log

Note: this tool is written in Objective-C++ thus macOS
environment is a requirement.

See https://tenhou.net/stat/rand/
https://81100118.github.io/2021/01/01/天凤牌山生成算法及其验证/

There is a flaw in the method of checking yama on the replay against  
the yama recomputed from the seed: the mjlog file does not actually record  
the yama, and that information is also likely to be recomputed from the  
seed in online replay mode.  

Thus, the most secure method is to check directly against the mjlog.
(Although the fact that the seed is used for the replay already
says something about the credibility of the claim.)

Mjlog is an Gzipped XML file, and can be downloaded from

https://tenhou.net/0/?log=xxxxxxxxgm-xxxx-xxxx-xxxxxxxx&tw=x

While log=xxxxxxxxgm-xxxx-xxxx-xxxxxxxx&tw=x is taken from the replay URL.

Unzip the file to XML and use `./out 1.xml` to check the integrity of
the log content against the seed recorded in the log.

The example output would be like:

```text
nKyoku=0 yama=五發三北白一７發二五９⑥中中⑤４３北⑤東⑨北七９９５１⑧東⑤③６⑦八四一一⑦④南５南７白⑦六６２二２⑥３二⑤四九４五①⑥四５九３八⑥中④８８中西西②東七西南④①６⑧２６⑦四⑧三２一六④南９１七②①六７８西５二白白③七北７八三３⑧４②８②三⑨九③東①五１③發⑨４⑨九發八六１
dice0=4 dice1=2
Hand passes check.
nKyoku=1 yama=發東⑦六東４②①⑥７２北一東一⑥６南二南二東④五九６④８３４四⑨６一①⑥⑦⑦西４六５八③⑤９北５⑦２７二北白六③五４南中中七西發５⑧②２１二八三⑨②７白西２中七四四九９１８⑧⑤８⑥１９三九八八１①③３南七６⑧８中①五五白５⑤３三白７九②⑨西④發３⑤七六④三四⑧北９⑨③發一
dice0=4 dice1=0
Hand passes check.
nKyoku=2 yama=②北發５１１１西一七６⑧東⑨⑧白八２二⑦六三３８９東五②４九①南②西２⑤中四一五９中⑧３５五⑨４二⑥北白一六５①８南南③白三⑦九北④⑥東⑦７⑨６２一８白９四四七三５二７３⑤八九７南⑨③四①４⑧４①８西６七五發七發西６３中④六六八７三９九④中⑥③⑤②④發北⑥③⑤⑦八東２１二
dice0=4 dice1=2
Hand passes check.
nKyoku=3 yama=８白３四１七２南發１３九①９９１二⑥中⑤⑥白③五⑦中東５④①⑨③九５北⑦發⑨南三９５五１７８②④五８⑦④白④南⑥北中六２⑤東⑧九六七八西四四４七３②三６⑤②西二九４⑦北６５一南⑨４③八二８中７７一八７六①七６９６①二⑧四③西發三東⑤２東北⑨五白②４西⑧一八三２⑧發一六⑥３
dice0=3 dice1=2
Hand passes check.
nKyoku=4 yama=五③八②４⑨８中六三⑨八六東西四七７９九９中九白二發⑥③東２五③②６七８２⑤④③南３南七⑤六六發⑦１①⑨④５⑧７⑦北九二白三４八７西三３一２９３一北１二⑤白四白一①⑥４⑦南⑧④一⑧９西５五６１７中４南北九東七６⑥６⑦發②３⑨５北二東②④五發８中①８四四⑧①⑤５１２三⑥八西
dice0=2 dice1=4
Hand passes check.
nKyoku=5 yama=七６⑨西⑦③⑧中四一七①五⑥三南二五白②八④中三３８⑤９⑤９南⑨九④④一八１六四一⑨⑥①六４八４西５發⑧西③四④①②⑤南５北白４②⑦東中⑧發②７三白７１４２四⑥北３⑧六⑦③九８７①⑨１五七白南北３６⑥３⑤發８發８一５６２③東１東９二北７七中２東二⑦二九６五三９西九六５２八
dice0=1 dice1=2
Hand passes check.
nKyoku=6 yama=⑧９２⑦九４四１中２九發五一八北②四７⑤４一⑨東東⑤四⑦②⑧②二４５九六⑥③④東８中１八９③８①６２二白北８④發西北⑧③七⑦⑧發１７九二⑨⑥北三４９白白２８⑤④一七９⑨④六３①７３六５⑨③七７①⑤西５南②東八⑥西中六五中１二５６①白五⑥三６南西七一五三３３６南三⑦八南四發
dice0=1 dice1=3
Hand passes check.
nKyoku=7 yama=④⑧７七３③白三南六中二②發③⑧５發５九白八７⑥⑥②北９白北３⑤④發⑨中五四五一⑦５４北④９一六九２一⑤⑨南九５中⑥八８２９⑦２北６二六二七⑨７⑤四１１２④９東六⑧②白西②４４七６三九中⑨１東南西三６⑥①一西①４西東３③三３８７五八８七五１南⑧二６⑤⑦①８⑦發③八四四東①
dice0=0 dice1=1
Hand passes check.
nKyoku=8 yama=中中北２四七四發②南南１２發五九９四３④南３北７東２８發三⑦三白西①１③北八⑨六一６⑥⑤④⑧１白②①５５⑧⑤九一西①⑨④６七９⑦一七③西②⑥９７７⑧二八⑦白四⑥七二４③⑤東８①５東南三６六九２②３⑥五１８六⑨３中六⑤九４⑦發４二８⑨⑧６北中二９一八五７五④東４白５③西八三
dice0=2 dice1=0
Hand passes check.
nKyoku=9 yama=９２６九９５７八①２８⑧⑥①１⑧二五一２五５④一七⑤４白１②白⑨⑨四９４八１３６七五二４⑧③３５九⑦南二５⑦④⑧發７北南②八北六①⑤⑥８四西西④中④２白⑦８３③６②８中九⑨七七４東三⑤東東６六三中９⑥北一發九南北發⑤３７西⑦③發②③①東７１四二五八六白一六中⑥南⑨四三西三
dice0=3 dice1=5
Hand passes check.
nKyoku=10 yama=７３五發西３６四白②四③①５３２⑦１１９四八三４１四東五８北⑥③南八六５２⑤一４六⑥④⑨西九④２七④六三北①５西２②⑥五東⑧①②８北８７４⑤中８八７⑨②七①七發發白八白南二南三１白一９５二９二４⑤⑨④東九南中九６⑦３７發⑨二東⑧中五⑧６③一九中六９⑦一三６③七⑧⑤西北⑥⑦
dice0=4 dice1=1
Hand passes check.
```

## TODOs

- Not ready for 3 player mahjong.
- Does not yet verify Dora and Rinshan.
