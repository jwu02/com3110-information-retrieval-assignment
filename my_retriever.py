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

        self.query = list()
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
        for w in query[:]:
            if w not in self.index:
                query.pop(query.index(w))
        self.query = query
        self.update_relevant_doc_ids()

        if self.term_weighting == 'binary':
            return self.binary_retrieval()
        elif self.term_weighting == 'tf':
            return self.tf_retrieval()
        else: # self.term_weighting == 'tfidf'
            return self.tfidf_retrieval()


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
        tf_wds = self.get_tf_wd_dict()
        
        for tf_wd in tf_wds:
            # replace list of query term frequencies with 
            # the sum of all query term occurrences in a document
            tf_wds[tf_wd] = sum(tf_wds[tf_wd])

        return self.get_top_relevant_doc_ids(tf_wds)


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


    def tfidf_retrieval(self) -> list:
        """
        Implements the principle of less frequent terms are more informative
        Multiplies the term frequency by the inverse document frequency 
        to provide a better term weighting scheme
        """
        tf_wds = self.get_tf_wd_dict() # term frequencies tf, of terms w, in a document d
        idfs = self.get_idf_list() # inverse document frequencies

        # tfidf = tf x idf
        # vector of representation of query: 1 x idfs = idfs (?)
        # dictionary storing tfidfs for each document
        tfidfs = {}
        for doc_id in self.relevant_doc_ids:
            tf_wd = tf_wds[doc_id]
            tfidfs[doc_id] = [tf_wd[i] * idfs[i] for i in range(len(self.query))]

        similarity_scores = {} # between a query and each relevant document
        for doc_id in self.relevant_doc_ids:
            similarity_scores[doc_id] = (sum([q*d for q,d in zip(idfs, tfidfs[doc_id])]) /
                (math.sqrt(sum([q*q for q in idfs])) * math.sqrt(sum([d*d for d in tfidfs[doc_id]]))))

        return self.get_top_relevant_doc_ids(similarity_scores)


    def get_tf_wd_dict(self) -> dict:
        """
        Return a dictionary which maps document ids, to list of frequency of occurrences,
        corresponding to each query term w
        """
        # tf_{w,d} = term frequency, occurrences of query term w in a document d
        tf_wds = {}
        for doc_id in self.relevant_doc_ids:
            tf_wd = []
            for w in self.query:
                if doc_id in self.index[w]:
                    tf_wd.append(self.index[w][doc_id])
                else: # if doc_id doesn't occur in index for a term
                    tf_wd.append(0)

            tf_wds[doc_id] = tf_wd
        
        return tf_wds


    def get_idf_list(self) -> list:
        """
        Return a list of inverse document frequencies corresponding to each query term w
        """
        # document frequency - number of documents containing each term w in query
        df_ws = [len(self.index[w]) for w in self.query]
        idfs = [math.log(self.num_docs/dfw) for dfw in df_ws]

        return idfs
