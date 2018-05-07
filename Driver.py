import xml.etree.ElementTree
from Utils import Helper

# Configuration
threshold = 2.5
penalty = 10
lower = 0
upper = 50
dataset = 'rte3_dev.xml'

# Properties
TP = 0
TN = 0
FP = 0
FN = 0
helper = Helper(penalty=penalty, threshold=threshold)

root = xml.etree.ElementTree.parse(dataset).getroot()
texts = []
hypos = []
entails = {}
scores = {}
cnt = lower

# Parse <T,H> pairs from dataset
for child in root:
    entails[cnt] = True if child.attrib['entailment'] == 'YES' else False
    cnt = cnt + 1
    texts.append(child[0].text)
    hypos.append(child[1].text)

# Classify pairs in given range
for i in range(lower, upper):
    ANSWER, SCORE = helper.classify(hypo=hypos[i], text=texts[i])
    REAL = entails[i]

    scores[i] = SCORE
    print(REAL, ANSWER, SCORE)

    # Calculate TP, TN, FP, FN values
    if ANSWER == REAL and ANSWER:
        TP += 1
    elif ANSWER == REAL and not ANSWER:
        TN += 1
    elif ANSWER:
        FP += 1
    else:
        FN += 1