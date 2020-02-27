import nltk
from hmm import HMM
from nltk.corpus import brown
import operator


class HMM_UNK(HMM):
    def __init__(self, corpus, tagset="", trainSize=10000, testSize=500, lang='en'):
        super().__init__(corpus, tagset, trainSize, testSize)
        self.initialised = False  # mark as 'not initialised'
        self.lang = lang
    
    def setup(self):
        # split the set of all sentences into training set and testing set
        self.trainSents, self.testSents = self.splitTrainingTesting()

        # A transit table that is used for counting occurrences of one part of speech following another in a training corpus
        self.transitTable = {}

        # split training sentences into tags and words
        self.words, self.tags = self.splitWordsTagsTraining()
        # count occurrences of words together with parts of speech in a training corpus
        self.occurrenceMap_w, self.occurrenceMap_t = self.countOccurrences()
        # find and replace the infrequent words with suitable UNK tag
        self.words = self.replaceInfrequentWords_UNK()
        # create transit table
        self.transitTable = self.createTransitTable(self.transitTable, self.occurrenceMap_t)

        # split testing sentences into tags and words
        self.check_sents = self.taggedSents[self.trainSize : self.trainSize + self.testingSize]
        self.testingWords, self.testingTags = self.splitWordsTagsTesting()

        # split testing sentences into tags and words - exclude delimiters
        self.testingWordsNoDelim, self.testingTagsNoDelim = self.splitWordsTagsTestingNoDelim()
        self.uniqueTags, self.uniqueTagsNoDelim = self.getUniqueTags()

        # calculate smoothed probabilities
        self.wordsDist, self.tagsDist = self.setProbDistributions()

        self.initialised = True  # mark as 'initialised'

    def convertWordToUNKTag_EN(self, word, tag):
        unk_tag = 'UNK'

        if word.endswith('ing'):
            unk_tag = 'UNK-ing'
        elif word.endswith("'s"):
            unk_tag = "UNK's"
        elif ',' in word:
            splitted = word.split(',')
            # use try-except to check if the string is a number
            try:
                for w in splitted:
                    val = int(w)
                unk_tag = 'UNK-Num'
            except:
                if '$' in word:
                    unk_tag = 'UNK-$'  # i.e. $25,000
                else:
                    unk_tag = 'UNK-unit'  # i.e. 75,000-ton , 25,000-man
        elif word.endswith('ion') or word.endswith('ions'):
            unk_tag = 'UNK-ion'
        elif word.endswith('ing'):
            unk_tag = 'UNK-ing'
        elif word.endswith('ial'):
            unk_tag = 'UNK-ial'
        elif word.endswith('al'):
            unk_tag = 'UNK-al'
        elif word.endswith("'t"):
            unk_tag = "UNK-'t"
        elif word.endswith('s'):
            unk_tag = 'UNK-s'
        elif word.endswith('ly'):
            unk_tag = 'UNK-ly'
        elif word.endswith('ful'):
            unk_tag = 'UNK-ful'
        elif word.endswith('est'):
            unk_tag = 'UNK-est'
        elif word.endswith('ed'):
            unk_tag = 'UNK-ed'
        elif word.endswith('ly'):
            unk_tag = 'UNK-ly'
        elif word.startswith('anti-'):
            unk_tag = 'anti-UNK'
        elif word.endswith('ism'):
            unk_tag = 'UNK-ism'
        elif word.endswith("'ll"):
            unk_tag = "UNK-'ll"
        elif word.endswith('%'):
            unk_tag = 'UNK-%'
        #TODO
            #print('word={} tag={}'.format(word, tag))
        else:
            try:
                val = int(word)
                unk_tag = 'UNK-Num'
            except ValueError:
                #TODO
                #print('word={} tag={}'.format(word, tag))
                unk_tag = word

        return unk_tag

    def convertWordToUNKTag_DU(self, word, tag):
        unk_tag = 'UNK'

        if word.endswith('te'):
            unk_tag = 'UNK-te'
        elif word.endswith('de'):
            unk_tag = 'UNK-de'
        elif word.startswith('ge') and (word.endswith('t') or word.endswith('d')):
            # pp -> ge + word + t/d
            unk_tag = 'ge-UNK-(t/d)'
        elif word.endswith('tien'):
            # tien is equal to teen in English
            unk_tag = 'UNK-tien'
        elif word.endswith('honderd'):
            #honderd = hundred
            unk_tag = 'UNK-honderd'
        elif word.endswith('thie'):
            # 'thie' endings are the equivalents of the English -thy (sympathy) and -thic.
            # 'thie' is used for nouns
            unk_tag = 'UNK-thie'
        elif word.endswith('thisch'):
            # 'thisch' endings are the equivalents of the English -thy (sympathy) and -thic.
            # 'thisch' is used for nouns
            unk_tag = 'UNK-thisch'
        elif word.endswith('achtig'):
            # '-achtig' is the Dutch translation for English '-like'
            unk_tag = 'UNK-achtig'
        elif word.endswith('ische'):
            # Many Dutch adjectives end in -isch or -ische (inflected).
            # It means something like English -ish.
            unk_tag = 'UNK-ische'
        elif word.endswith('isch'):
            unk_tag = 'UNK-isch'
        elif word.startswith('ont'):
            unk_tag = 'ont-UNK'
        elif word.startswith('er'):
            unk_tag = 'er-UNK'
        elif word.endswith('en'):
            unk_tag = 'UNK-en'
        elif word.endswith('s'):
            unk_tag = 'UNK-s'
        else:
            unk_tag = word

        return unk_tag

    def replaceInfrequentWords_UNK(self):
        """
        Find and replace the infrequent words with suitable UNK tag.
        :return newList: A new list of words, where all infrequent words are replaced with suitable UNK tags
        """
        newList = []
        # iterate both word list and tag list to find and replace the infrequent words with suitable UNK tag
        for w, t in zip(self.words, self.tags):
            word = w
            if self.occurrenceMap_w[w] == 1:
                if self.lang == 'en':
                    word = self.convertWordToUNKTag_EN(w, t)
                elif self.lang == 'du':
                    word = self.convertWordToUNKTag_DU(w, t)
            newList.append(word)
        return newList
    
    def viterbi(self, targetSentences:list=None):
        """
        Viterbi Algorithm
        """
        finalTags=[]
        probMatrix = []
        distOfStartOfSentence = self.tagsDist['<s>']

        # check if the targetSentences is None
        if targetSentences is None:
            targetSentences = self.testingWordsNoDelim  # If None, use the testing sentences

        # use for loop to iterate targetSentences
        for s in targetSentences:
            firstRun = True
            for init_word in s:
                col = []
                word = init_word

                # Check if a word did not occur in the training corpus or only infrequently (occurred once).
                if (word not in self.occurrenceMap_w) or (self.occurrenceMap_w[word] == 1):
                    if self.lang == 'en':
                        word = self.convertWordToUNKTag_EN(init_word, '')
                    elif self.lang == 'du':
                        word = self.convertWordToUNKTag_DU(init_word, '')

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

        return finalTags


def downloadCorpus():
    nltk.download('brown')
    nltk.download('universal_tagset')

def main():
    corpus = brown
    tagset = "universal"
    hmm = HMM_UNK(corpus, tagset)
    hmm.setup()
    hmm.viterbi_test()

if __name__ == '__main__':
    downloadCorpus()
    main()
