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
        
        TF_MODERATION_METHODS = ['none', 'log_tf', 'max_tf_norm']
        self.tf_moderation_method = TF_MODERATION_METHODS[1]

        self.tf_wds = self.get_tfs() # term frequencies tf, of terms w, in a document d
        if self.term_weighting == 'tfidf':
            self.document_tfidfs = self.get_document_tfidfs() # {doc_id: {term: tfidf}}
    

    def compute_number_of_documents(self):
        """
        Return total number of documents in collection
        """
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    
    def get_tfs(self) -> dict:
        """
        Return a dictionary which maps document ids, to terms, to term frequency
        Raw term frequencies are transformed through the chosen moderation method
        """
        # tf_{w,d} = term frequency, occurrences of query term w in a document d
        tf_wds = {}
        for doc_id in self.doc_ids:
            tf_wd = {}
            for w in self.index:
                if doc_id in self.index[w]:
                    # record all term occurrences in a document
                    tf_wd[w] = self.index[w][doc_id]

            tf_wds[doc_id] = tf_wd
        
        # moderate term frequency
        if self.tf_moderation_method == 'log_tf':
            tf_wds = self.apply_log_tf(tf_wds)
        elif self.tf_moderation_method == 'max_tf_norm':
            tf_wds = self.apply_max_tf_normalization(tf_wds)

        return tf_wds


    def apply_log_tf(self, tf_wds: dict) -> dict:
        """
        Return a term frequency dict with the 1+log(raw frequency count) applied
        """
        for doc_id in tf_wds:
            for w in tf_wds[doc_id]:
                if tf_wds[doc_id][w] != 0:
                    tf_wds[doc_id][w] = 1+math.log(tf_wds[doc_id][w])
        
        return tf_wds


    def apply_max_tf_normalization(self, tf_wds: dict) -> dict:
        """
        Return a term frequency dict with maximum tf normalization applied\n
        Has the effect of normalising the raw frequency count by the freqeuncy
        of the most frequent term in each document
        """
        a = 0.4 # smoothing factor

        for doc_id in tf_wds:
            # max term frequency count for a document d
            tf_max_d = max(tf_wds[doc_id].values())

            for w in tf_wds[doc_id]:
                tf_wd = tf_wds[doc_id][w]
                tf_wds[doc_id][w] = a + (1-a)*(tf_wd/tf_max_d)
        
        return tf_wds


    def for_query(self, query: list) -> list:
        """
        Method performing retrieval for a single query (which is represented 
        as a list of preprocessed terms).\n
        Returns list of doc ids for relevant docs (in rank order).
        """
        self.query = self.process_query(query)
        self.update_relevant_doc_ids()

        # apply term frequency moderation to query
        if self.tf_moderation_method == 'log_tf':
            self.query = self.apply_log_tf({0:self.query})[0]
        elif self.tf_moderation_method == 'max_tf_norm':
            self.query = self.apply_max_tf_normalization({0:self.query})[0]

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
        return query list as a dictionary which maps query terms to their frequency
        in the query, to take into account of multiple occurrences of terms
        """
        # iterate through a copy of the query list, so we can modify original one
        # without running into issues
        for w in query[:]:
            if w not in self.index:
                query.remove(w)
        
        query_term_count = {}
        for w in query:
            if w in query_term_count:
                query_term_count[w] += 1
            else:
                query_term_count[w] = 1

        return query_term_count


    def update_relevant_doc_ids(self):
        """
        Clear and update set of relevant document ids, which contains any query terms
        """
        self.relevant_doc_ids = set()
        for w in self.query:
            self.relevant_doc_ids.update(self.index[w])


    def binary_retrieval(self) -> list:
        """
        Binary model\n
        Retrieves documents with any query term occurrences (disjunction of all query terms)\n
        Return top documents with most query term occurrences
        """
        relevant_tf_wds = self.get_relevant_tf_wds()
        for doc_id in self.relevant_doc_ids:
            # replace dict values with the sum of all query term occurrences in a document
            query_term_occurrences = []
            for w in self.query:
                if w in relevant_tf_wds[doc_id]:
                    query_term_occurrences.append(relevant_tf_wds[doc_id][w])
                else:
                    query_term_occurrences.append(0)
            
            relevant_tf_wds[doc_id] = sum(query_term_occurrences)

        return self.get_top_relevant_doc_ids(relevant_tf_wds)


    def get_relevant_tf_wds(self) -> dict:
        return {doc_id:self.tf_wds[doc_id] for doc_id in self.relevant_doc_ids}


    def tf_retrieval(self) -> list:
        """
        Vector space retrieval model: term frequency\n
        Return top most similar documents
        """
        relevant_tf_wds = self.get_relevant_tf_wds()
        similarity_scores = self.get_similarty_scores(self.query, relevant_tf_wds)

        return self.get_top_relevant_doc_ids(similarity_scores)


    def tfidf_retrieval(self) -> list:
        """
        Vector space retrieval model: term frequency x inverse document frequency\n
        Implements the principle of less frequent terms are more informative
        Multiplies the term frequency by the inverse document frequency 
        to provide an extension to the term weighting scheme
        """
        # 0 is a dummy document id, to make use of `get_document_tfidfs` function
        query_tfidfs = self.get_document_tfidfs({0:self.query})[0] # {query term: tfidf}
        relevant_document_tfdifs = {doc_id:self.document_tfidfs[doc_id] for doc_id in self.relevant_doc_ids}

        similarity_scores = self.get_similarty_scores(query_tfidfs, relevant_document_tfdifs)

        return self.get_top_relevant_doc_ids(similarity_scores)


    def get_document_tfidfs(self, query_tf=None) -> dict:
        """
        Return a dictionary which maps document id to terms, to tfidfs (tf x idf)
        """
        idfs = {} # {term: idf}
        for w in self.index:
            # document frequency - number of documents containing each term w
            df_w = len(self.index[w])
            idf = math.log(self.num_docs/df_w)
            idfs[w] = idf
        
        tfidfs = {} # {doc_id: {term: frequency}}

        tf_wds = query_tf if query_tf else self.tf_wds
        for doc_id in tf_wds:
            tf_wd = tf_wds[doc_id] # {term: frequency}
            # for each term multiply its tf a document by its idf
            tfidfs[doc_id] = {w:tf_wd[w]*idfs[w] for w in tf_wd}
        
        return tfidfs


    def get_similarty_scores(self, query_weights: dict, document_weights: dict) -> dict:
        """
        Return a dictionary which maps document id to a similarity/weighting score,
        the higher the score, the higher ranked the document relevance
        """
        similarity_scores = {} # between a query and each relevant document
        for doc_id in document_weights:
            # square root of the sum of each d_i term weight squared
            sqrt_sum_d_sq = math.sqrt(sum([d*d for d in document_weights[doc_id].values()]))

            single_document_weights = document_weights[doc_id]
            for w in query_weights:
                if w not in single_document_weights:
                    single_document_weights[w] = 0
            
            # sum of products of q_i and d_i term weights
            sum_prod_q_d = sum([query_weights[w]*single_document_weights[w] for w in query_weights])

            similarity_scores[doc_id] = sum_prod_q_d / sqrt_sum_d_sq
        
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
