import nltk
from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.tag.util import untag
import operator


class HMM:
    def __init__(self, corpus, tagset="", trainSize=10000, testSize=500):
        # initialise the basic attributes
        self.corpus = corpus
        self.taggedSents, self.sents = self.getSentences(tagset)
        self.trainSize = trainSize
        self.testingSize = testSize
        self.initialised = False


    def setup(self):
        """
        Initialise the basic attributes.
        """

        # split the set of all sentences into training set and testing set
        self.trainSents, self.testSents = self.splitTrainingTesting()

        # A transit table that is used for counting occurrences of one part of speech following another in a training corpus
        self.transitTable = {}

        # split training sentences into tags and words
        self.words, self.tags = self.splitWordsTagsTraining()

        # count occurrences of words together with parts of speech in a training corpus
        self.occurrenceMap_w, self.occurrenceMap_t = self.countOccurrences()

        # create transit table
        self.transitTable = self.createTransitTable(self.transitTable, self.occurrenceMap_t)

        # split testing sentences into tags and words
        self.check_sents = self.taggedSents[self.trainSize:self.trainSize + self.testingSize]
        self.testingWords, self.testingTags = self.splitWordsTagsTesting()

        # split testing sentences into tags and words - exclude delimiters
        self.testingWordsNoDelim, self.testingTagsNoDelim = self.splitWordsTagsTestingNoDelim()

        # Get unique tags
        self.uniqueTags, self.uniqueTagsNoDelim = self.getUniqueTags()
        # calculate smoothed probabilities
        self.wordsDist, self.tagsDist = self.setProbDistributions()
        # Mark as initialised
        self.initialised = True


    def viterbi(self, targetSentences:list=None):
        """
        Viterbi Algorithm
        """
        print('Start viterbi algorithm')
        finalTags=[]
        probMatrix = []
        distOfStartOfSentence = self.tagsDist['<s>']

        #TODO
        for w in self.occurrenceMap_w:
            if self.occurrenceMap_w[w] == 1:
                print(w)

        # check if the targetSentences is None
        if targetSentences is None:
            targetSentences = self.testingWordsNoDelim  # If None, use the testing sentences

        # use for loop to iterate targetSentences
        for s in targetSentences:
            firstRun = True
            for word in s:
                col = []
                for t in self.uniqueTagsNoDelim:

                    # check if this is the first run
                    if firstRun:
                        pT = distOfStartOfSentence.prob(t)  # use the distribution value of 'start-of-sentence' as tag probability
                        pW = self.wordsDist[t].prob(word)   # calculate the probability value for the current word with given tag
                        col.append([pW*pT, "q0"])

                    else:
                        tagMap = {}
                        for pp in range(0, len(self.uniqueTagsNoDelim)):
                            # P(t(i) | t(i-1))
                            pT = self.tagsDist[self.uniqueTagsNoDelim[pp]].prob(t)
                            # P(w(i) | t(i))
                            pW = self.wordsDist[t].prob(word)

                            tagMap[self.uniqueTagsNoDelim[pp]] = pT * pW * probMatrix[-1][pp][0]

                        prevBestTag = max(tagMap.items(), key=operator.itemgetter(1))[0]
                        value = max(tagMap.items(), key=operator.itemgetter(1))[1]

                        col.append([value, prevBestTag])
                firstRun = False
                probMatrix.append(col)

            finalTags.append(self.getTagsFromMatrix(probMatrix, s))

        print('Finish viterbi algorithm')

        return finalTags

    def getSentences(self, selected_tagset):
        tagged_sents = self.corpus.tagged_sents(tagset=selected_tagset)
        sents = self.corpus.sents()
        return tagged_sents, sents

    def countOccurrences(self):
        """
        Count the occurrences of words together with parts of speech in a training corpus.

        :return occurrenceMap_w: A dictionary for the occurrences of words
        :return occurrenceMap_t: A dictionary for the occurrences of tags
        """
        wordList = self.words
        tagList = self.tags

        def countOccurrencesForGivenList(targetList):
            """
            Count the occurrences of elements in the given list.

            :return targetMap: A dictionary for the calculated occurrences.
            """
            targetMap = {}
            for e in targetList:
                #TODO
                # skip the start-of-sentence and end-of-sentence
                #if e == '<s>' or e == '</s>':
                    #continue

                if targetMap.get(e) is None:
                    targetMap[e] = 1
                else:
                    targetMap[e] = targetMap[e] + 1

            return targetMap

        occurrenceMap_w = countOccurrencesForGivenList(wordList)
        occurrenceMap_t = countOccurrencesForGivenList(tagList)

        return occurrenceMap_w, occurrenceMap_t

    def countPrevTagToCurTag(self, sentence):
        """
        Count occurrences of one part of speech following another in a training corpus.
        """
        curTag = '<s>'

        for (_, t) in sentence:
            prevTag = curTag
            curTag = t

            # check if given sequence of tags is in the transitTable
            if prevTag not in self.transitTable:
                self.transitTable[prevTag] = {}
                self.transitTable[prevTag][curTag] = 1
            elif curTag not in self.transitTable[prevTag]:
                self.transitTable[prevTag][curTag] = 1
            else:
                self.transitTable[prevTag][curTag] += 1


    def splitTrainingTesting(self):
        """
        Split a list of all sentences into training sentences and testing sentences.

        :return train_sents: A list of training sentences
        :return test_sents: A list of testing sentences
        """
        train_sents = self.taggedSents[:self.trainSize]
        test_sents = self.sents[self.trainSize:self.trainSize + self.testingSize]
        return train_sents, test_sents

    def splitIntoWordsAndTags(self, sentences, isTraining=False):
        """
        This method splits the sentences into words and tags.

        :param sentences: A set of target sentences
        :return words: A list of words
        :return tags: A list of tags
        """
        words = []
        tags = []
        startDelimeter = ["<s>"]
        endDelimeter = ["</s>"]

        for s in sentences:
            if isTraining:
                self.countPrevTagToCurTag(s) # count occurrences of one part of speech following another in a training corpus

            words += startDelimeter + [w for (w, _) in s] + endDelimeter
            tags += startDelimeter + [t for (_, t) in s] + endDelimeter
        return words, tags


    def createTransitTable(self, transit_table, tags_counter):
        """
        Create transit table with Laplace smoothing.

        :param transit_table: A transit table that contains the occurrences of one part of speech following another in a training corpus.
        :param tags_counter: A list of occurrences of part of speech in a training corpus.
        """
        # One less because a start of sentence <S> is excluded
        total_tags = len(transit_table) - 1

        for i in transit_table:
            for j in transit_table:
                if j != '<s>':
                    if j not in transit_table[i]:
                        transit_table[i][j] = 0  # fill empty cells with zero value
                    else:
                        # apply Laplace smoothing
                        transit_table[i][j] = (transit_table[i][j] + 1.0) / (tags_counter[i] + total_tags)
        return transit_table

    def splitWordsTagsTraining(self):
        """
        Splitting the training sentences into words and tags.

        :return: Returns a list of words and a list of tags, which are returned by the splitToWordsByTags() method
        """
        return self.splitIntoWordsAndTags(self.trainSents, True)

    def splitWordsTagsTesting(self):
        """
        Splitting the testing sentences into words and tags.

        :return: Returns a list of words and a list of tags, which are returned by the splitToWordsByTags() method
        """
        return self.splitIntoWordsAndTags(self.check_sents)

    def splitWordsTagsTestingNoDelim(self):
        sentences = []
        tags = []

        for s in self.check_sents:
            # untag the sentence, and append it to the list of sentences
            sentences.append(untag(s))
            tags.append([t for (_, t) in s]) # add all tags in a sentence to a list of tags
        return sentences, tags


    def getUniqueTags(self):
        """
        Get lists of unique tags.

        :return uniqueTagList: A list of unique tags - include delimiters
        :return uniqueTagList_noDelim: A list of unique tags - exclude delimiters
        """
        uniqueTagList = list(set(self.tags))
        uniqueTagList_noDelim = uniqueTagList.copy()
        uniqueTagList_noDelim.remove('<s>')
        uniqueTagList_noDelim.remove('</s>')
        return uniqueTagList, uniqueTagList_noDelim

    def setProbDistributions(self):
        tag_dist = {}
        word_dist = {}

        for t in self.uniqueTags:
            tagList = []
            lenOfTags = len(self.tags)
            wordList = []
            for i in range(lenOfTags - 1):
                if self.tags[i] == t:
                    wordList.append(self.words[i])
                    if i < (lenOfTags - 2):
                        tagList.append(self.tags[i+1])
            tag_dist[t] = WittenBellProbDist(FreqDist(tagList), bins=1e5)
            word_dist[t] = WittenBellProbDist(FreqDist(wordList), bins=1e5)

        return word_dist, tag_dist

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
                pointerID = self.uniqueTagsNoDelim.index(pointer)
                finalTags.append(self.uniqueTagsNoDelim[pointerID])
                pointer = matrix[-i][pointerID][1]

        finalTags.reverse()

        return finalTags

    def getAccuracy(self):
        correct = 0
        total = 0
        for s in range(0, len(self.testingTagsNoDelim)):
            for t in range(0, len(self.testingTagsNoDelim[s])):
                if self.testingTagsNoDelim[s][t] == self.finalTags[s][t]:
                    correct = correct+1
                total = total + 1

        percent = (correct / total) * 100

        print("Training Data: " + str(self.trainSize) + " Sentences")
        print("Testing Data: " + str(self.testingSize) + " Sentences")
        print("Accuracy {}%".format(percent))

    def viterbi_test(self):
        if not self.initialised:
            self.setup()
        self.finalTags = self.viterbi()
        self.getAccuracy()



def downloadCorpus():
    nltk.download('brown')
    nltk.download('universal_tagset')

def main():
    # create HMM instance, and run the Viterbi test
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)
    hmm.setup()
    hmm.viterbi_test()

if __name__ == '__main__':
    downloadCorpus()
    main()
