wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1Wh1jW6Wy_B_le2wcea63nz9UZ5GCurFH' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Wh1jW6Wy_B_le2wcea63nz9UZ5GCurFH" \
 -O model/bert_best_clf.pt && rm -rf /tmp/cookies.txt
