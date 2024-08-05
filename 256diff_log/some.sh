#!/bin/bash

files=(
1V20yAxVtF.dffr
rX742wx4Ms.dffr
5i07zqDcdK.dffr
w8Amx7Sjrt.dffr
49lTb19lm8.dffr
1d1L3HE1HE.dffr
R60m6k2XCj.dffr
fX2MZW7oNE.dffr
7G99L249AT.dffr
2q9GJ0790k.dffr
y46g2Mz1UM.dffr
uiX3y2a62p.dffr
3551PXv5D2.dffr
5oN9MyfjuI.dffr
l058vL29uX.dffr
9030355iyf.dffr
14GgSwXnw8.dffr
WED37XQ2I9.dffr
3lf9pjAvs4.dffr
0Y778Dh46X.dffr
S470J5xJCA.dffr
01bfKq61fx.dffr
1bwrzQd27b.dffr
OmBL3t7AOu.dffr
f4Mvu8OCXG.dffr
Dt6k3Dn6Wr.dffr
vor6o4p10x.dffr
hAzf75894q.dffr
AJU0U1F8UZ.dffr
BBDwqTPCvu.dffr
2Lh67T9ORw.dffr
3iGVtbRyTl.dffr
CRGPRMOa4I.dffr
L7f8JcFC48.dffr
P1C8cagU4S.dffr
av49S8YeyX.dffr
h0xgUYiHXW.dffr
)

if ! [ -d bad/ ]; then
	mkdir bad/
fi

for file in discarded_pats/*; do
	mid="$(echo $file | awk -F '/' '{ print $NF }' | awk -F '.' '{ print $1 }')"
	tobmp --cmap mono data/"$mid".dffr -o bad/"$mid".bmp
done
