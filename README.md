## Files
- standard test set (or benchmark) files for evaluating
    - `documents.txt` contains collection of documents that record publications in the CACM, each records its title, author(s) and abstract, although some may be absent
    - `queries.txt` contains a set of IR queries for use against this collection
    - `cacm_gold_std.txt` is a gold standard identifying the documents that have been judged relevant to each query
- `eval_ir.py` calculates system performance scores, by comparing the gold standard to a system results file
    - `python eval_ir.py cacm_gold_std.txt results_file.txt`
    - execute script with its help option `-h` for instructions on use
- `IR_engine.py` is the ‘outer shell’ of a retrieval engine
    - `python IR_engine.py -s -p -w tfidf -o results_file.txt`
        - `-h` list command line options
        - `-s` use stoplist
        - `-p` use stemming
    - loads a (preprocessed) index and query set, from the file IR `data.pickle`
    - then ‘batch processes’ the queries to compute the 10 best-ranking documents for each, which it prints to a results file