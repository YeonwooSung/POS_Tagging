How to run the program

1. hmm.py        ->  python3 hmm.py
- By running "python3 hmm.py", the Viterbi algorithm will be executed with the brown corpus.

2. unk.py        ->  python3 unk.py
 - By running "python3 unk.py", the UNK tagging method will be executed with the brown corpus.

3. otherLang.py  ->  python3 otherLang.py <1~5> <y | n>
- The otherLang.py requires at least 1 command line argument
- The first argument should be an integer between 1 to 5.
    1 = alpino corpus (Dutch)
    2 = floresta corpus (Portuguese)
    3 = conll2002 corpus (Spanish)
    4 = conll2000 corpus (English)
    5 = cess_esp corpus (Spanish)
- The second argument is an optional one
- If you want to use the second argument, you should put either y or n.
- If the second argument is y, then the HMM with UNK tagging method will be executed with the selected corpus.
- If the second argument is n, or there is no second argument, then the normal Viterbi algorithm will be executed.
- i.e.
    alpino with HMM     ->  python3 otherLang.py 1
    alpino with HMM     ->  python3 otherLang.py 1 n
    alpino with UNK     ->  python3 otherLang.py 1 y
    floresta with HMM   ->  python3 otherLang.py 2
    floresta with HMM   ->  python3 otherLang.py 2 n
    floresta with UNK   ->  python3 otherLang.py 2 y
    conll2002 with HMM  ->  python3 otherLang.py 3
    conll2002 with HMM  ->  python3 otherLang.py 3 n
    conll2002 with UNK  ->  python3 otherLang.py 3 y
    conll2000 with HMM  ->  python3 otherLang.py 4
    conll2000 with HMM  ->  python3 otherLang.py 4 n
    conll2000 with UNK  ->  python3 otherLang.py 4 y
    cess_esp with HMM   ->  python3 otherLang.py 5
    cess_esp with HMM   ->  python3 otherLang.py 5 n
    cess_esp with UNK   ->  python3 otherLang.py 5 y
