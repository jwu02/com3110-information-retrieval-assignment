
class Retrieve:
    
    def __init__(self, index, term_weighting):
        """
        Create new Retrieve object storing index and term weighting scheme.
        (You can extend this method, as required.)
        """
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
    

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    
    def for_query(self, query):
        """
        Method performing retrieval for a single query (which is 
        represented as a list of preprocessed terms).
        Returns list of doc ids for relevant docs (in rank order).
        """
        if self.term_weighting == 'binary':
            return self.binary_retrieval(query)
        elif self.term_weighting == 'tf':
            return self.tf_retrieval(query)
        else: # self.term_weighting == 'tfidf'
            return self.tfidf_retrieval(query)
        
        # return list(range(1,11))


    def binary_retrieval(self, query: list) -> list:
        """
        Only care about whether terms occur in a document, nothing more
        Count number of occurrences of query terms in each document
        Return top ten most occurrences
        """
        top_10_relevant_doc_ids = []

        relevant_docs = {}
        # check if each term is in index
        for q in query:
            if q in self.index:
                # and record total number of occurrences of each term, for each document
                for doc_id in self.index[q]:
                    if doc_id in relevant_docs:
                        relevant_docs[doc_id] += self.index[q][doc_id]
                    else:
                        relevant_docs[doc_id] = self.index[q][doc_id]

        # print(query)
        # print(relevant_docs)
        # print("================================")

        for i in range(10):
            doc_id = max(relevant_docs, key=relevant_docs.get)
            del relevant_docs[doc_id]
            top_10_relevant_doc_ids.append(doc_id)

            if len(relevant_docs) == 0:
                break

        return top_10_relevant_doc_ids


    def tf_retrieval(self, query: list) -> list:
        return list(range(1,11))


    def tfidf_retrieval(self, query: list) -> list:
        return list(range(1,11))