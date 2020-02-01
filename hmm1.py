from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.tag.util import untag
import operator


class HMM:
    def __init__(self, corpus, tagset="", trainSize=10000, testSize=500):
        self.corpus = corpus
        self.taggedSents, self.sents = self.getSentences(tagset)
        self.trainSize = trainSize
        self.testingSize = testSize

        # split the set of all sentences into training set and testing set
        self.trainSents, self.testSents = self.splitTrainingTesting()
        self.countOccurences()

        # Gets the list of all unique tags
        self.uniqueTags = self.getListOfUniqueTags()

        #TODO relative frequency estimation with smoothing


    def countOccurences(self):
        self.words, self.tags, self.all_words, self.all_tags = self.splitToWordsByTags(self.trainSents)
        #TODO counting occurrences of one part of speech following another in a training corpus
        #TODO counting occurrences of words together with parts of speech in a training corpus

    def splitToWordsByTags(self, sentences):
        """
        This method that splits each sentence into words and tags.

        :param sentences: A set of target sentences
        :return words: A list of word lists
        :return tags: A list of tag lists
        :return all_words: A list of all words
        :return all_tags: A list of all tags
        """
        words = []
        tags = []
        all_words = []
        all_tags = []

        # use for loop to iterate the list of sentences
        for s in sentences:
            w, t = zip(*s) #split the sentence into tuples of all words an tags in the sentence
            list_w = list(w)
            list_t = list(t)

            all_words += list_w.copy()
            all_tags += list_t.copy()

            words.append(list(w))
            tags.append(list(t))

        return words, tags, all_words, all_tags

    def getSentences(self, selected_tagset):
        """
        Get the tagged sentences and sentences.

        :return tagged_sents: Tagged sentences, where each sentence contains both tags and words
        :return sents: Normal sentences
        """
        tagged_sents = self.corpus.tagged_sents(tagset=selected_tagset)
        sents = self.corpus.sents()
        return tagged_sents, sents

    def splitTrainingTesting(self):
        """
        Split a list of all sentences into training sentences and testing sentences.

        :return train_sents: A list of training sentences
        :return test_sents: A list of testing sentences
        """
        train_sents = self.taggedSents[:self.trainSize]
        test_sents = self.sents[self.trainSize:self.trainSize + self.testingSize]
        return train_sents, test_sents

    def getListOfUniqueTags(self):
        return list(set(self.all_tags))


    def smoothing(self):
        smoothed_tag = {}
        smoothed_words = {}
        wordMap = {}

        for t in self.uniqueTags:
            tagList = []
            wordList = []
            for i in range(len(self.tags)-1):
                if self.tags[i] == t:
                    wordList.append(self.words[i])
                    if i < (len(self.tags)-2):
                        tagList.append(self.tags[i+1])

            smoothed_tag[t] = WittenBellProbDist(FreqDist(tagList), bins=1e5)
            smoothed_words[t] = WittenBellProbDist(FreqDist(wordList), bins=1e5)

        return smoothed_words, smoothed_tag



def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM(corpus, tagset)

def downloadCorpus():
    import nltk
    nltk.download('brown')
    nltk.download('universal_tagset')

if __name__ == '__main__':
    downloadCorpus()
    main()
