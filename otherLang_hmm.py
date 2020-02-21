import nltk
import sys
from hmm import HMM
from nltk.corpus import conll2000, conll2002, alpino, floresta



def downloadCorpus():
    nltk.download('alpino')
    nltk.download('floresta')
    nltk.download('conll2000')
    nltk.download('conll2002')

def main_otherLang(corpus, tagset):
    tagged_sents = list(corpus.tagged_sents())
    numOfSents = len(tagged_sents)
    print('The number of total sentences = {}'.format(numOfSents))

    if numOfSents > 10500:
        hmm = HMM(corpus, tagset)
        hmm.setup()
        hmm.viterbi_test()
    else:
        train_size = int(numOfSents / 100 * 95) #TODO 95% rather than 90%
        test_size = numOfSents - train_size

        hmm = HMM(corpus, tagset, trainSize=train_size, testSize=test_size)
        hmm.setup()
        hmm.viterbi_test()


if __name__ == '__main__':
    downloadCorpus()

    selected_corpus = ''
    try:
        selected_corpus = int(sys.argv[1])
    except ValueError:
        print('The first argument should be one of 1, 2, 3, and 4')
        exit(1)

    if selected_corpus == 1:
        print('\nHMM for alpino')
        main_otherLang(alpino, "")
    elif selected_corpus == 2:
        print('\nHMM for floresta')
        main_otherLang(floresta, "")
    elif selected_corpus == 3:
        print('\nHMM for conll2002')
        main_otherLang(conll2002, "")
    elif selected_corpus == 4:
        print('\nHMM for conll2000')
        main_otherLang(conll2000, "universal")
    else:
        print('The first argument should be one of 1, 2, 3, and 4')
