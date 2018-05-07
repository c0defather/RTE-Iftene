import xml.etree.ElementTree
from Utils import Helper
import sys

TP = 0
TN = 0
FP = 0
FN = 0
threshold = 2.8
penalty = 10
if len(sys.argv) == 3:
    penalty = float(sys.argv[1])
    threshold = float(sys.argv[2])
print(sys.argv)
helper = Helper(penalty=penalty, threshold=threshold)

root = xml.etree.ElementTree.parse('rte3_dev.xml').getroot()
texts = []
hypos = []
entails = []
scores = []
for child in root:
    entails.append(True if child.attrib['entailment'] == 'YES' else False)
    texts.append(child[0].text)
    hypos.append(child[1].text)
cnt = 50

for i in range(0, cnt):
    answer, score = helper.process(hypo=hypos[i], text=texts[i])
    real = entails[i]

    scores.append(score)
    print(real, answer, score)

    if answer == real and answer:
        TP += 1
    elif answer == real and not answer:
        TN += 1
    elif answer:
        FP += 1
    else:
        FN += 1


def findBestThreshold():
    ans = 0
    for i in range(0, 1000):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        threshold = i / 10.0
        for j in range(0, cnt):
            real = entails[j]
            score = scores[j]
            answer = True
            if score > threshold:
                answer = False

            if answer == real and answer:
                tp += 1
            elif answer == real and not answer:
                tn += 1
            elif answer:
                fp += 1
            else:
                fn += 1

        #recall = tp / (tp + fn)
        #precision = tp / (tp + fp)
        #f = 2 * recall * precision / (recall + precision)
        acc = (tp+tn)/(tp+tn+fp+fn)
        if acc > ans:
            ans = acc
            print(acc, i)

    return
