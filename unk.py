import nltk
from hmm import HMM
from nltk.corpus import brown
import operator


class HMM_UNK(HMM):
    def __init__(self, corpus, tagset="", trainSize=10000, testSize=500, lang='en', infrequent=1):
        super().__init__(corpus, tagset, trainSize, testSize)
        self.initialised = False  # mark as 'not initialised'
        self.lang = lang
        self.infrequent = infrequent

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

    def convertWordToUNKTag_EN(self, word):
        word = word.lower()
        unk_tag = word

        if word.endswith('ing'):
            unk_tag = 'UNK-ing'
        elif word.endswith("'s"):
            unk_tag = "UNK's"
        elif word.endswith('ion') or word.endswith('ions'):
            unk_tag = 'UNK-ion'
        elif word.endswith('ial'):
            unk_tag = 'UNK-ial'
        elif word.endswith('ble'):
            unk_tag = 'UNK-ble'
        elif word.endswith('er'):
            unk_tag = 'UNK-er'
        elif word.endswith('or'):
            unk_tag = 'UNK-or'
        elif word.endswith('al'):
            unk_tag = 'UNK-al'
        elif word.endswith('fy'):
            unk_tag = 'UNK-fy'
        elif word.endswith('ic'):
            unk_tag = 'UNK-ic'
        elif word.endswith('ful'):
            unk_tag = 'UNK-ful'
        elif word.endswith('est'):
            unk_tag = 'UNK-est'
        elif word.endswith('less'):
            unk_tag = 'UNK-less'
        elif word.endswith('ous'):
            unk_tag = 'UNK-ous'
        elif word.endswith('ed'):
            unk_tag = 'UNK-ed'
        elif word.endswith('ly'):
            unk_tag = 'UNK-ly'
        elif word.startswith('anti'):
            unk_tag = 'anti-UNK'
        elif word.startswith('pre'):
            unk_tag = 'pre-UNK'
        elif word.endswith('ism'):
            unk_tag = 'UNK-ism'
        elif word.endswith('wise'):
            # i.e. otherwise, likewise, clockwise
            unk_tag = 'UNK-wise'
        elif word.endswith('ward') or word.endswith('wards'):
            # i.e. forward, backward
            unk_tag = 'UNK-ward'
        elif word.endswith('ist'):
            unk_tag = 'UNK-ist'
        elif word.endswith('s'):
            unk_tag = 'UNK-s'

        return unk_tag

    def convertWordToUNKTag_DU(self, word):
        unk_tag = word

        if word.endswith('te'):
            # used for adjectives
            unk_tag = 'UNK-te'
        elif word.endswith('de'):
            # form ordinal numbers to cardinal numbers
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
            # prefix "ont-"
            unk_tag = 'ont-UNK'
        elif word.startswith('er'):
            # prefix "er-"
            unk_tag = 'er-UNK'
        elif word.endswith('en'):
            # A suffix “-en” forms verbs from nouns or adjectives.
            unk_tag = 'UNK-en'
        elif word.endswith('s'):
            # forms regular plurals of nouns that end in certain suffixes or syllables
            unk_tag = 'UNK-s'

        return unk_tag

    def convertWordToUNKTag_ES(self, word):
        unk_tag = word

        if word.endswith('achon'):
            # Adds negative connotations to a word
            unk_tag = 'UNK-achon'
        elif word.endswith('aco'):
            # It sometimes adds a despective sense, it is also seen to add a demonym
            unk_tag = 'UNK-aco'
        elif word.endswith('ado'):
            # Makes reference to names of associations or ensembles
            unk_tag = 'UNK-ado'
        elif word.endswith('ción'):
            # Expresses the idea of action on nouns that are derived from a verb
            unk_tag = 'UNK-ción'
        elif word.endswith('mente'):
            # Uses adjectives to form modal adverbs
            unk_tag = 'UNK-mente'
        elif word.endswith('génesis'):
            # Transmits the idea of origin or beginning
            unk_tag = 'UNK-génesis'

        # suffixes that makes the words to nouns or adjectives
        elif word.endswith('able'):
            unk_tag = 'UNK-able'
        elif word.endswith('ario'):
            unk_tag = 'UNK-ario'
        elif word.endswith('eña') or word.endswith('eño'):
            unk_tag = 'UNK-eñ(a/o)'
        elif word.endswith('iza') or word.endswith('izo'):
            unk_tag = 'UNK-iz(a/o)'
        elif word.endswith('oso') or word.endswith('osa'):
            unk_tag = 'UNK-os(a/o)'
        elif word.endswith('or'):
            unk_tag = 'UNK-or'

        return unk_tag

    def convertWordToUNKTag_PO(self, word):
        unk_tag = word

        if word.endswith('ário'):
            # ário is a suffix that is used for "professional" or "place"
            unk_tag = 'ário'
        elif word.endswith('eiro'):
            # eiro is a suffix that is used for "professional"
            unk_tag = 'UNK-eiro'
        elif word.endswith('inho') or word.endswith('ico') or word.endswith('isco'):
            # inho, ico, and isco are the diminutive suffixes
            # inho is much more used than the other diminutive suffixes
            unk_tag = 'UNK-diminutive'
        elif word.endswith('ão') or word.endswith('aço') or word.endswith('aréu'):
            # ão, aço, and aréu are the augmentative suffixes
            unk_tag = 'UNK-augmentative'
        elif word.endswith('mente'):
            # mente is a suffix that is generally used for adverbs
            # It does similar thing with "-ly" in English
            unk_tag = 'UNK-mente'
        elif word.startswith('Anfi') or word.startswith('anfi'):
            # prefix for dualty
            unk_tag = 'anfi-UNK'
        elif word.startswith('Anti') or word.startswith('anti'):
            # prefix for opposition
            unk_tag = 'anti-UNK'
        elif word.startswith('sim') or word.startswith('Sim') or word.startswith('sin') or word.startswith('Sin'):
            # "simultaneously"
            unk_tag = 'si(m/n)-UNK'
        elif word.startswith('peri') or word.startswith('Peri'):
            # "Around"
            unk_tag = 'peri-UNK'
        elif word.startswith('Hemi') or word.startswith('hemi'):
            # "Half"
            unk_tag = 'hemi-UNK'

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
            if self.occurrenceMap_w[w] <= self.infrequent:
                if self.lang == 'en':
                    word = self.convertWordToUNKTag_EN(w)
                elif self.lang == 'du':
                    word = self.convertWordToUNKTag_DU(w)
                elif self.lang == 'po':
                    word = self.convertWordToUNKTag_PO(w)
                elif self.lang == 'es':
                    word = self.convertWordToUNKTag_ES(w)
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
                if (word not in self.occurrenceMap_w) or (self.occurrenceMap_w[word] <= self.infrequent):
                    if self.lang == 'en':
                        word = self.convertWordToUNKTag_EN(word)
                    elif self.lang == 'du':
                        word = self.convertWordToUNKTag_DU(word)
                    elif self.lang == 'po':
                        word = self.convertWordToUNKTag_PO(word)
                    elif self.lang == 'es':
                        word = self.convertWordToUNKTag_ES(word)

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
