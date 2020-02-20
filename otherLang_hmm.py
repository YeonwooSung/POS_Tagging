import nltk
from hmm import HMM
from nltk.corpus import conll2000, alpino, floresta, gutenberg



def downloadCorpus():
    nltk.download('alpino')
    nltk.download('floresta')
    nltk.download('gutenberg')
    nltk.download('conll2000')

def main_otherLang(corpus, tagset):
    tagged_sents = list(corpus.tagged_sents())
    numOfSents = len(tagged_sents)
    print('The number of total sentences = {}'.format(numOfSents))

    if numOfSents > 15000:
        hmm = HMM(corpus, tagset)
        hmm.setup()
        hmm.viterbi_test()
    else:
        train_size = int(numOfSents / 10 * 9)
        test_size = numOfSents - train_size

        hmm = HMM(corpus, tagset, trainSize=train_size, testSize=test_size)
        hmm.setup()
        hmm.viterbi_test()


if __name__ == '__main__':
    #downloadCorpus()
    print('HMM for alpino')
    main_otherLang(alpino, "")
    print('\nHMM for floresta')
    main_otherLang(floresta, "")
    print('\nHMM for conll2000')
    main_otherLang(conll2000, "")
    print('\nHMM for gutenberg')
    main_otherLang(gutenberg, "")
