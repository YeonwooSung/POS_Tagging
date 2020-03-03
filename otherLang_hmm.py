import nltk
import sys
import time
from hmm import HMM
from unk import HMM_UNK
from nltk.corpus import conll2000, conll2002, alpino, floresta, cess_esp



def downloadCorpus():
    nltk.download('alpino')
    nltk.download('floresta')
    nltk.download('conll2000')
    nltk.download('conll2002')

def main_otherLang_UNK(corpus, tagset, lang):
    tagged_sents = list(corpus.tagged_sents())
    numOfSents = len(tagged_sents)
    print('The number of total sentences = {}'.format(numOfSents))

    if numOfSents > 10500:
        hmm = HMM_UNK(corpus, tagset, lang=lang)
        hmm.setup()
        hmm.viterbi_test()
    else:
        # train : test = 95 : 5
        train_size = int(numOfSents / 100 * 95)
        test_size = numOfSents - train_size

        hmm = HMM_UNK(corpus, tagset, trainSize=train_size, testSize=test_size, lang=lang)
        hmm.setup()
        hmm.viterbi_test()

def main_otherLang(corpus, tagset):
    tagged_sents = list(corpus.tagged_sents())
    numOfSents = len(tagged_sents)
    print('The number of total sentences = {}'.format(numOfSents))

    if numOfSents > 10500:
        hmm = HMM(corpus, tagset)
        hmm.setup()
        hmm.viterbi_test()
    else:
        # train : test = 95 : 5
        train_size = int(numOfSents / 100 * 95)
        test_size = numOfSents - train_size

        hmm = HMM(corpus, tagset, trainSize=train_size, testSize=test_size)
        hmm.setup()
        hmm.viterbi_test()


if __name__ == '__main__':
    downloadCorpus()

    selected_corpus = ''
    try:
        selected_corpus = int(sys.argv[1])
    except (ValueError, IndexError) as e:
        print('The first argument should be one of 1, 2, 3, 4, and 5')
        exit(1)

    # check if the user input 2nd command line argument
    unk = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'y':
            unk = True
        elif sys.argv[2] == 'n':
            unk = False
        else:
            print('The second argument should be either y or n')
            exit(1)

    start = time.time()

    if selected_corpus == 1:
        print('\nHMM for alpino')
        if unk:
            main_otherLang_UNK(alpino, "", 'du')
        else:
            main_otherLang(alpino, "")
    elif selected_corpus == 2:
        print('\nHMM for floresta')
        # floresta = 9k sentences, tagged and parsed (Portuguese)
        if unk:
            main_otherLang_UNK(floresta, "", 'po')
        else:
            main_otherLang(floresta, "")
    elif selected_corpus == 3:
        print('\nHMM for conll2002')
        if unk:
            main_otherLang_UNK(conll2002, "esp", 'es')
        else:
            main_otherLang(conll2002, "esp")
    elif selected_corpus == 4:
        print('\nHMM for conll2000')
        if unk:
            main_otherLang_UNK(conll2000, "universal", "en")
        else:
            main_otherLang(conll2000, "universal")
    elif selected_corpus == 5:
        if unk:
            main_otherLang_UNK(cess_esp, "", "es")
        else:
            main_otherLang(cess_esp, "")
    else:
        print('The first argument should be one of 1, 2, 3, 4, and 5')
        exit(1)

    costTime = time.time() - start
    print('Total cost time = {0:.2f}'.format(costTime))
