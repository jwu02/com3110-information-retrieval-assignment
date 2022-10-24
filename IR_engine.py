"""\
------------------------------------------------------------
USE: python <PROGNAME> (options)
OPTIONS:
    -h : print this help message
    -s : use "with stoplist" configuration (default: without)
    -p : use "with stemming" configuration (default: without)
    -w LABEL : use weighting scheme "LABEL" (LABEL in {binary, tf, tfidf}, default: binary)
    -o FILE : output results to file FILE
------------------------------------------------------------\
"""

#==============================================================================
# Importing

import sys
import getopt
import pickle

from my_retriever import Retrieve

#==============================================================================
# Command line processing

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'hspw:o:')
        opts = dict(opts)
        self.exit = True

        if '-h' in opts:
            self.print_help()
            return

        if len(args) > 0:
            print("*** ERROR: no arg files - only options! ***", file=sys.stderr)
            self.print_help()
            return

        if '-w' in opts:
            if opts['-w'] in ('binary', 'tf', 'tfidf'):
                self.term_weighting = opts['-w']
            else:
                warning = (
                    "*** ERROR: term weighting label (opt: -w LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: binary / tf / tfidf"
                    )  % (opts['-w'])
                print(warning, file=sys.stderr)
                self.print_help()
                return
        else:
            self.term_weighting = 'binary'

        if '-o' in opts:
            self.outfile = opts['-o']
        else:
            print("*** ERROR: must specify output file (opt: -o FILE) ***",
                  file=sys.stderr)
            self.print_help()
            return

        if '-s' in opts:
            stoplist = 'yes'
        else:
            stoplist = 'no'
        
        if '-p' in opts:
            stemming = 'yes'
        else:
            stemming = 'no'

        with open('IR_data.pickle', 'rb') as data_in:
            all_data = pickle.load(data_in)

        choice = 'index_stoplist_%s_stemming_%s' % (stoplist, stemming)
        self.index = all_data[choice]
            
        choice = 'queries_stoplist_%s_stemming_%s' % (stoplist, stemming)
        self.queries = all_data[choice]
            
        self.exit = False

    def print_help(self):
        progname = sys.argv[0]
        progname = progname.split('/')[-1] # strip off extended path
        help = __doc__.replace('<PROGNAME>', progname, 1)
        print(help, file=sys.stderr)

#==============================================================================
# Store for Retrieval Results

class Result_Store:
    def __init__(self):
        self.results = []

    def store(self, qid, docids):
        if len(docids) > 10:
            docids = docids[:10]
        self.results.append((qid, docids))

    def output(self, outfile):
        with open(outfile, 'w') as out:
            for (qid, docids) in self.results:
                for docid in docids:
                    print(qid, docid, file=out)

#==============================================================================
# MAIN

if __name__ == '__main__':

    config = CommandLine()
    if config.exit:
        sys.exit(0)
    queries = config.queries
    retrieve = Retrieve(config.index, config.term_weighting)
    all_results = Result_Store()

    for (qid, query) in queries:
        results = retrieve.for_query(query)
        all_results.store(qid, results)

    all_results.output(config.outfile)

