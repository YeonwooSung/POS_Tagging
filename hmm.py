from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.tag.util import untag
import operator
import sys


class HMM:
    def __init__(self, corpus, tagset=""):
        self.corpus = corpus
        self.taggedSents, self.sents = self.getSentences(tagset)
        self.trainSize = int(sys.argv[2])
        self.testingSize = int(sys.argv[3])
        self.trainSents, self.testSents = self.splitTrainingTesting()
        self.words, self.tags = self.splitWordsTags()
        self.check_sents = self.taggedSents[self.trainSize:self.trainSize + self.testingSize]

        self.testingWords, self.testingTags = self.splitWordsTagsTesting()
        self.testingWordsNoDelim, self.testingTagsNoDelim = self.splitWordsTagsTestingNoDelim()

        self.tagsDistribution = FreqDist(self.tags)
        self.uniqueTags, self.uniqueTagsNoDelim = self.getUniqueTags()
        self.wordsDist, self.tagsDist = self.setProbDistributions()

        self.finalTags = self.viterbi()
        self.output()


    def viterbi(self):
        """
        Viterbi Algorithm
        """
        finalTags=[]
        probMatrix = []
        for s in self.testingWordsNoDelim:
            firstRun = True
            for word in s:
                col = []
                for t in self.uniqueTagsNoDelim:

                    if firstRun:
                        pT = self.tagsDist['<s>'].prob(t)
                        pW = self.wordsDist[t].prob(word)
                        col.append([pW*pT, "q0"])

                    else:
                        tagMap = {}
                        for pp in range(0, len(self.uniqueTagsNoDelim)):
                            pT = self.tagsDist[self.uniqueTagsNoDelim[pp]].prob(t)
                            pW = self.wordsDist[t].prob(word)

                            tagMap[self.uniqueTagsNoDelim[pp]] = pT * pW * probMatrix[-1][pp][0]

                        prevBestTag = max(tagMap.items(), key=operator.itemgetter(1))[0]
                        value = max(tagMap.items(), key=operator.itemgetter(1))[1]

                        col.append([value, prevBestTag])
                firstRun = False
                probMatrix.append(col)

            finalTags.append(self.getTagsFromMatrix(probMatrix, s))

        return finalTags

    def getSentences(self, selected_tagset):
        taggedSents = self.corpus.tagged_sents(tagset=selected_tagset)
        sents = self.corpus.sents()
        return taggedSents, sents

    def splitTrainingTesting(self):
        train_sents = self.taggedSents[:self.trainSize]
        test_sents = self.sents[self.trainSize:self.trainSize + self.testingSize]
        return train_sents, test_sents

    def splitWordsTags(self):
        words = []
        tags = []
        startDelimeter = ["<s>"]
        endDelimeter = ["</s>"]

        for s in self.trainSents:
            words += startDelimeter + [w for (w, _) in s] + endDelimeter
            tags += startDelimeter + [t for (_, t) in s] + endDelimeter
        return words, tags

    def splitWordsTagsTesting(self):
        words = []
        tags = []
        startDelimeter = ["<s>"]
        endDelimeter = ["</s>"]

        for s in self.check_sents:
            words += startDelimeter + [w for (w, _) in s] + endDelimeter
            tags += startDelimeter + [t for (_, t) in s] + endDelimeter
        return words, tags

    def splitWordsTagsTestingNoDelim(self):
        sentances = []
        tags = []

        for s in self.check_sents:
            sentances.append(untag(s))
            tags.append([t for (_, t) in s])
        return sentances, tags

    def getUniqueTags(self):
        tagSet = set(self.tags)
        tagList = list(tagSet)

        noDelim = tagList.copy()

        noDelim.remove('<s>')
        noDelim.remove('</s>')

        return tagList, noDelim

    def setProbDistributions(self):
        tagMap = {}
        wordMap = {}

        for t in self.uniqueTags:
            tagList = []
            wordList = []
            for i in range(len(self.tags)-1):
                if self.tags[i] == t:
                    wordList.append(self.words[i])
                    if i < (len(self.tags)-2):
                        tagList.append(self.tags[i+1])
            tagMap[t] = WittenBellProbDist(FreqDist(tagList), bins=1e5)
            wordMap[t] = WittenBellProbDist(FreqDist(wordList), bins=1e5)

        return wordMap, tagMap
    
    def getTagsFromMatrix(self, matrix, s):
        finalTags = []
        pointer = ""

        for i in range(1, len(s)+1):
            max = 0
            maxID = 0
            if i == 1:
                for j in range(0, len(self.uniqueTagsNoDelim)):

                    if matrix[-i][j][0] > max:
                        max = matrix[-i][j][0]
                        maxID = j
                finalTags.append(self.uniqueTagsNoDelim[maxID])
                pointer = matrix[-i][maxID][1]

            else:
                # print(pointer)
                pointerID = self.uniqueTagsNoDelim.index(pointer)
                finalTags.append(self.uniqueTagsNoDelim[pointerID])
                pointer = matrix[-i][pointerID][1]

        finalTags.reverse()

        return finalTags
    

    def output(self):
        print(self.testingTagsNoDelim)
        print("--------------------------")
        print(self.finalTags)
        print("--------------------------")

        correct = 0
        total = 0
        for s in range(0, len(self.testingTagsNoDelim)):
            for t in range(0, len(self.testingTagsNoDelim[s])):
                if self.testingTagsNoDelim[s][t] == self.finalTags[s][t]:
                    correct = correct+1
                total = total + 1

        percent = (correct/total)*100

        print("Training Data: " + str(self.trainSize) + " Sentences")
        print("Testing Data: " + str(self.testingSize) + " Sentences")
        print(str(percent)+"% Accuracy")



def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)

if __name__ == '__main__':
    main()
