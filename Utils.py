import json

from nltk.corpus import wordnet
from nltk.parse.stanford import DependencyGraph, StanfordDependencyParser
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import RegexpTokenizer
from zss import simple_distance, Node


class Helper:
    def __init__(self, penalty: float, threshold: float):
        self.ner = StanfordNERTagger(
            'libs/english.all.3class.distsim.crf.ser.gz',
            'libs/stanford-ner-3.9.1.jar')

        path_to_jar = 'libs/stanford-corenlp-3.9.1.jar'
        path_to_models_jar = 'libs/stanford-corenlp-3.9.1-models.jar'
        self.dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

        self.penalty = penalty
        self.threshold = threshold

    # Return dependency parse tree.
    def dep_parse(self, text: str):
        return self.dependency_parser.raw_parse(text).__next__()

    # Function checks if all name entities in first text exists in second one.
    # 't1' and 't2' are the first and the second texts respectively.
    # return True if all name entities in 't1' exists in 't2'.
    def ne_match(self, t1: str, t2: str):
        tokenizer = RegexpTokenizer(r'\w+')
        words1 = tokenizer.tokenize(t1)
        words2 = tokenizer.tokenize(t1)
        nes1 = self.ner.tag(words1)
        for ne in nes1:
            if ne[1] != 'O':
                contains = False
                for w in words2:
                    if w.lower() == ne[0].lower():
                        contains = True
                        break
                if not contains:
                    return False
        return True

    # Returns True if word is Named Entity; False, otherwise.
    def is_ne(self, word: str):
        ne = self.ner.tag([word])
        return True if ne[0][1] != 'O' else False

    # Constructs tree required for calculating ZSS tree edit distance.
    # 'dep_graph' is a dependency graph constructed with Stanford Parser.
    @staticmethod
    def construct_zss_tree(dep_graph: DependencyGraph):
        a = str(dep_graph)
        a = a[a.find(',') + 1:-1]
        a = a.replace("defaultdict(<class 'list'>,", '').replace("),", ",").replace("'", '"').replace("None", '""')
        for i in reversed(range(0, 100)):
            a = a.replace(str(i) + ":", '"' + str(i) + '":')

        dep = json.loads(a)
        rootIndex = dep['0']['deps']['root'][0]
        root = Node(dep[str(rootIndex)]['word'])

        def helper(node: Node, index: int):
            children = dep[str(index)]['deps'].values()
            children = sum(children, [])
            for c in children:
                cNode = Node(dep[str(c)]['word'])
                helper(cNode, c)
                node.addkid(cNode)
            return

        helper(root, rootIndex)

        return root

    # Returns word similarity between two words.
    def word_sim(self, word1: str, word2: str):
        sims = []
        wordFromList1 = wordnet.synsets(word1)
        wordFromList2 = wordnet.synsets(word2)
        if wordFromList1 and wordFromList2:
            s = wordFromList1[0].wup_similarity(wordFromList2[0])
            sims.append(s)
        if len(sims) == 0:
            return self.penalty
        elif len(sims) == 1 and not sims[0]:
            return self.penalty
        res = max(sims)
        if res < 0.5:
            return self.penalty
        return 2-res

    # Returns mapping between hypothesis and text trees using ZSS.
    @staticmethod
    def zss_distance(hypo_root: Node, text_root: Node):
        return simple_distance(hypo_root, text_root, return_operations=True)

    # Process one pair of hypothesis and text
    def classify(self, hypo: str, text: str):
        score = []

        hypo = hypo.replace("'s", "")
        hypo = hypo.replace("'", "*")
        hypo = hypo.replace('"', "*")
        text = text.replace("'s", "")
        text = text.replace('"', '*')
        text = text.replace("'", '*')

        if not self.ne_match(hypo, text):
            return False, -1

        textDep = self.dep_parse(text)
        hypoDep = self.dep_parse(hypo)

        hypoRoot = self.construct_zss_tree(hypoDep)
        textRoot = self.construct_zss_tree(textDep)

        dist, ops = self.zss_distance(hypoRoot, textRoot)

        for op in ops:
            if op.type == 2 or op.type == 3:
                word1 = op.arg1.label
                word2 = op.arg2.label
            if op.type == 3:
                score.append(1)
            elif op.type == 2:
                if self.is_ne(word1):
                    s = 1
                else:
                    s = self.word_sim(word1.lower(), word2.lower())
                score.append(s)
        total = len(score)
        normScore = []
        for s in score:
            normScore.append(s / total)
        normScore = sum(normScore)
        if normScore < self.threshold:
            answer = True
        else:
            answer = False

        return answer, normScore


