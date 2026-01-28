#!/bin/zsh

grep -i accuracy *txt | awk '{print }' | awk '{print $2}' > accuracy.csv 
grep -i auc *txt | awk '{print }' | awk '{print $2}' > auc.csv 
grep -i recall *txt | awk '{print }' | awk '{print $2}' > recall.csv 
grep -i precision: *txt | awk '{print }' | awk '{print $2}' > precision.csv 
grep -i f1-score *txt | awk '{print }' | awk '{print $2}' > f1-score.csv 


grep -o "^\[\[.." *txt | grep -o "..$" > true_negatives.csv		   # True Negatives
grep "^\[\[.." *txt | grep -o "..]$"| grep -o "^.." > false_positives.csv  # False Posives
grep -o "..\]\]$" *txt | grep -o "^.." > true_positives.csv 		   # True Positives
grep "..\]\]$" *txt | grep -o "\[.." | grep -o "..$" > false_negatives.csv # False Negatives

./plot.py
