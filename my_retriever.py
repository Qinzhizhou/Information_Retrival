import math

class Retrieve:
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index #
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents() # Totol number of documents in the  documents which is 3204
        self.doc_vec = self.doc_vec_size() # The vector of document size, the values different as the tw_scheme changed
        # log（D/number of documents contains the term） 

    def compute_number_of_documents(self): # Compute the total number of documents in the collection
        self.doc_ids = set() # doc_ids is a two_level dictionary
        for term in self.index:
            self.doc_ids.update(self.index[term]) # undate the id in for loop
        return len(self.doc_ids) # In this dcuments is 3204

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).

 
    def doc_vec_size(self): # A function get the dictionary contains {"qid": "Size(Vector)"}
        #|d| = sqrt(sum of (q)^2)
        docid_vec = dict().fromkeys(self.doc_ids, 0) # Make the directory with ids and satrt value as 0
        #print(docid_vec )
        if self.term_weighting == 'tf':
            for term, docid_dict in self.index.items(): # for loop for index(a 2-d dict{terms{docid : counts}})
                #print(term, docid_dict)
                for docid, freq in docid_dict.items():
                    docid_vec[docid] += freq * freq # accumlate frequency squre for each term in document corresponding the query
            
        elif self.term_weighting == 'binary':
            for term, docid_dict in self.index.items(): 
                for docid, freq in docid_dict.items():
                    docid_vec[docid] =+ 1 # Cumulative binary
            #print(docid_vec)
        elif self.term_weighting == 'tfidf':
            for term, docid_dict in self.index.items():
                idf = {term: math.log10(self.num_docs / len(docid_dict)) for term, docid_dict in self.index.items()}
                for docid, freq in docid_dict.items():
                    docid_vec [docid] += pow(freq * idf[term],2)  # Cumulative TF-IDF squared
        
        for docid, accu in docid_vec.items():
            docid_vec[docid] = math.sqrt(accu) # Squre root
        return docid_vec 
        # {"qid": "Size(Vector)"}   {qid: |d|}
        # |q|*|d|
        #eg {'D3672': 2.23606797749979, 'D1983': 3.0, 'D3021': 2.8284271247461903, 'D4650': 9.327379053088816, 'D0821': 2.449489742783178, 'D1886': 2.8284271247461903, 'D3609': 13.19090595827292}
    
    def similarity(self, query): # Builing vector q*d
        # Find the top 10 docid for each qid.  
        # document frequency
        term_index = {sentences: word for sentences, 
                      word in self.index.items() if sentences in query} # Subset of index dictionary related to query
        # print(term_index)
        # document frequency dfw which is q 
        #{term:{freq in document}}
        # eg. {'exist': {'D0969': 1, 'D2582': 1, 'D4639': 1, 'D2388': 1, 'D1723': 2, 'D2671': 1, 'D4214': 1, 'D3059': 1, 'D0944': 1, 'D3187': 1}, 'tss': {'D0733': 1}}
        
        dict_tem = dict()
        # Building the q vector
        if self.term_weighting == 'tf':
            for term, docid_dict in term_index.items(): # Term + index in documents  by + { D1911   }
                #print(term, docid_dict)
                for docid, freq in docid_dict.items():
                    dict_tem[docid] = dict_tem.get(docid, 0) +  freq
        
        # Divide the accumulated value by the size of the corresponding document

        if self.term_weighting == 'tfidf':
            for term, docid_dict in term_index.items(): # Term + index in documents  by + { D1911   }
                idf = {term: math.log10(self.num_docs / len(docid_dict)) for term, docid_dict in self.index.items()}
                #print(term, docid_dict)
                for docid, freq in docid_dict.items():
                    dict_tem[docid] = dict_tem.get(docid, 0) +  pow(idf[term],2) * freq  

        if self.term_weighting == 'binary':
            for term, docid_dict in term_index.items(): # Term + index in documents  by + { D1911   }
                #print(term, docid_dict)
                for docid, freq in docid_dict.items():
                    dict_tem[docid] = dict_tem.get(docid,0) + 1 
     
        score = dict_tem
        return score
        # A dictionary used to record q * d, whose key is the document 
        # {'D1911': 1, 'D1284': 1, 'D0642': 3, 'D1172': 1, 'D2457': 1, 'D2346': 3, 'D4875': 1, 'D2659': 1 .....} which is q * d

    
    def for_query(self, query):  
        score = self.similarity(query) # Builing the q vector
        doc_length = self.doc_vec # Building the d vector
        # Divide the accumulated value by the size of the corresponding docume
        for docid, accu in score.items():
            score[docid] = score[docid]/doc_length[docid] # q*d / |q||d|
            # The similarity score
        
        # Sort by dictionary value, that is, similarity order
        # Rank documents with respect to the query
        # Return the top 10 to the user

        results = sorted(score, key=lambda similarity: score[similarity], reverse=True) # QID + did rank top 10
        return results[:10]