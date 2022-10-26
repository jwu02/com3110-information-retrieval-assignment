from collections import Counter
import math


class Retrieve:
    
    def __init__(self, index, term_weighting):
        """
        Create new Retrieve object storing index and term weighting scheme.
        (You can extend this method, as required.)
        """
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()

        self.query = dict()
        self.relevant_doc_ids = set()
    

    def compute_number_of_documents(self):
        """
        Return total number of documents in collection
        """
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    
    def for_query(self, query: list) -> list:
        """
        Method performing retrieval for a single query (which is 
        represented as a list of preprocessed terms).
        Returns list of doc ids for relevant docs (in rank order).
        """
        self.query = self.process_query(query)
        self.update_relevant_doc_ids()

        # call respective retrieval methods for the different term weighting options
        if self.term_weighting == 'binary':
            return self.binary_retrieval()
        elif self.term_weighting == 'tf':
            return self.tf_retrieval()
        else: # self.term_weighting == 'tfidf'
            return self.tfidf_retrieval()


    def process_query(self, query: list) -> dict:
        """
        Remove terms from query which don't occur in index and
        return query list as a dictionary which maps query terms to their frequency,
        to take into account of multiple occurrences of terms, e.g. after stemming
        """
        # iterate through a copy of the query list, so we can modify original one
        # without running into issues
        for w in query[:]:
            if w not in self.index:
                query.remove(w)
        
        query_term_count = Counter(query)

        return dict(query_term_count)


    def update_relevant_doc_ids(self):
        """
        Clear and update the set of relevant document ids, which contains any query terms
        """
        self.relevant_doc_ids = set()
        for w in self.query:
            self.relevant_doc_ids.update(self.index[w])


    def binary_retrieval(self) -> list:
        """
        Only care about whether terms occur in a document, nothing more
        Count number of occurrences of query terms in each document
        Return top ten most occurrences
        """
        relevant_docs = {}
        # check if each term is in index
        for w in self.query:
            if w in self.index:
                # and record total number of occurrences of each term, for each document
                for doc_id in self.index[w]:
                    if doc_id in relevant_docs:
                        relevant_docs[doc_id] += self.index[w][doc_id]
                    else:
                        relevant_docs[doc_id] = self.index[w][doc_id]

        return self.get_top_relevant_doc_ids(relevant_docs)


    def tf_retrieval(self) -> list:
        """
        Term frequency retrieval ranks documents with most occurrrences of 
        query terms more highly
        """
        tf_wds = self.get_tf_wd_dict()
        
        for doc_id in tf_wds:
            # replace list of query term frequencies for each document with 
            # the sum of all query term occurrences in a document
            tf_wds[doc_id] = sum(tf_wds[doc_id].values())

        return self.get_top_relevant_doc_ids(tf_wds)


    def tfidf_retrieval(self) -> list:
        """
        Implements the principle of less frequent terms are more informative
        Multiplies the term frequency by the inverse document frequency 
        to provide a better term weighting scheme
        """
        tf_wds = self.get_tf_wd_dict() # term frequencies tf, of terms w, in a document d
        idfs = self.get_idf_dict() # inverse document frequencies

        # vector of representation of query: 1 x idfs = idfs (?) -> No
        query_tfidfs = self.get_query_tfidfs(idfs)
        document_tfidfs = self.get_document_tfidfs_dict(tf_wds, idfs)

        similarity_scores = self.get_similarty_score_dict(query_tfidfs, document_tfidfs)

        return self.get_top_relevant_doc_ids(similarity_scores)


    def get_tf_wd_dict(self) -> dict:
        """
        Return a dictionary which maps document ids, to terms, to frequency of occurrences
        """
        # tf_{w,d} = term frequency, occurrences of query term w in a document d
        tf_wds = {}
        for doc_id in self.relevant_doc_ids:
            tf_wd = {}
            for w in self.query:
                if doc_id in self.index[w]:
                    tf_wd[w] = self.index[w][doc_id]
                else: # if doc_id doesn't occur in index for a term
                    tf_wd[w] = 0

            tf_wds[doc_id] = tf_wd
        
        return tf_wds


    def get_idf_dict(self) -> dict:
        """
        Return a dictionary which maps query terms w to inverse document frequencies
        """
        # document frequency - number of documents containing each term w in query
        df_ws = {w:len(self.index[w]) for w in self.query}

        idfs = {w:math.log(self.num_docs/df_ws[w]) for w in df_ws}

        return idfs


    def get_query_tfidfs(self, idfs: dict) -> dict:
        """
        Return a dictionary which maps query terms to tfidfs (tf x idf)
        """
        return {w:self.query[w]*idfs[w] for w in self.query}


    def get_document_tfidfs_dict(self, tf_wds: dict, idfs: dict) -> dict:
        """
        Return a dictionary which maps document id to terms, to tfidfs (tf x idf)
        """
        tfidfs = {} # dictionary storing tfidfs for each document
        for doc_id in tf_wds:
            tf_wd = tf_wds[doc_id]
            tfidfs[doc_id] = {w:tf_wd[w]*idfs[w] for w in self.query}
        
        return tfidfs


    def get_similarty_score_dict(self, query_tfidfs: dict, document_tfidfs: dict) -> dict:
        """
        Return a dictionary which maps document id to a similarity score,
        the bigger the higher ranked the document relevance
        """
        similarity_scores = {} # between a query and each relevant document
        for doc_id in document_tfidfs:
            similarity_scores[doc_id] = (sum([q*d for q,d in zip(query_tfidfs.values(), document_tfidfs[doc_id].values())]) /
                (math.sqrt(sum([q*q for q in query_tfidfs.values()])) * math.sqrt(sum([d*d for d in document_tfidfs[doc_id].values()]))))
        
        return similarity_scores


    def get_top_relevant_doc_ids(self, doc_id_to_num_dict: dict) -> list:
        """
        Returns the top 10 most relevant documents given a dictionary of mapping
        from document id to some numeric score, with higher scores ranked more highly/relevant
        """
        top_relevant_doc_ids = []
        for i in range(10):
            doc_id = max(doc_id_to_num_dict, key=doc_id_to_num_dict.get)
            del doc_id_to_num_dict[doc_id]
            top_relevant_doc_ids.append(doc_id)

            if len(doc_id_to_num_dict) == 0:
                break
        
        return top_relevant_doc_ids
