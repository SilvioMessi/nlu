import numpy as np
from rdflib import Graph

class SentencesSimilarity():
    
    def __init__(self, words_similarity_obj, nlp_obj):
        self.words_similarity_obj = words_similarity_obj
        self.nlp_obj = nlp_obj

    def get_sentences_similarity(self, sentence_1=None, sentence_2=None):
        sentence_1_word_tokens, sentence_1_number_tokens = self.sentence_preprocessing(sentence_1)
        sentence_2_word_tokens, sentence_2_number_tokens = self.sentence_preprocessing(sentence_2)
        
        # lexical analysis
        # words
        num_tokens_s_1 = len(sentence_1_word_tokens)
        num_tokens_s_2 = len(sentence_2_word_tokens)
        if num_tokens_s_1 != 0 and  num_tokens_s_2 != 0 :
            # np.empty([num_rows, num_colums])
            matrix = np.empty([num_tokens_s_1, num_tokens_s_2]);
            for index_row , token_1 in enumerate(sentence_1_word_tokens):
                for index_column, token_2 in enumerate(sentence_2_word_tokens):
                    matrix[index_row, index_column] = self.words_similarity_obj.get_words_similarity(token_1, token_2)
            _, word_partial_similarity = self.matrix_similarity(matrix)
        else:
            _, word_partial_similarity = 1, 1
        word_SDPC = self.size_difference_penalization_coefficient(num_tokens_s_1, num_tokens_s_2, word_partial_similarity)
        # paper incoherence!!!
        # word_sim = word_total_similarity - word_SDPC
        word_sim = word_partial_similarity - word_SDPC
        # numbers
        num_tokens_s_1 = len(sentence_1_number_tokens)
        num_tokens_s_2 = len(sentence_2_number_tokens)
        if num_tokens_s_1 != 0 and num_tokens_s_2 != 0 :
            matrix = np.empty([num_tokens_s_1, num_tokens_s_2]);
            for index_row , token_1 in enumerate(sentence_1_number_tokens):
                for index_column, token_2 in enumerate(sentence_2_number_tokens):
                    matrix[index_row, index_column] = 1 if token_1 == token_2  else 0
            number_total_similarity, number_partial_similarity = self.matrix_similarity(matrix)
            number_SDPC = self.size_difference_penalization_coefficient(num_tokens_s_1, num_tokens_s_2, number_partial_similarity)
            number_sim = number_total_similarity - number_SDPC
        else:
            number_total_similarity = number_partial_similarity = 'Cannot create matrix for the numbers'
            number_SDPC = number_sim = 0
        # lexical final similarity
        # if number_sim == 1:
        if True:
            n_word = len(list(set(sentence_1_word_tokens) | set(sentence_2_word_tokens)))
            n_number = len(list(set(sentence_1_number_tokens) | set(sentence_2_number_tokens)))
            lexical_final_similarity = ((n_word * word_sim) + (n_number * number_sim)) / (n_word + n_number)
        else:
            lexical_final_similarity = word_sim - (1 - number_sim)    

        # syntactic analysis
        dependencies_graph_sentence_1 = self.nlp_obj.get_dependecies_graph(' '.join(self.nlp_obj.get_lemmas(sentence_1)[0]))[0]
        dependencies_graph_sentence_2 = self.nlp_obj.get_dependecies_graph(' '.join(self.nlp_obj.get_lemmas(sentence_2)[0]))[0]
        sentence_1_dep_graph = self.dependencies_graph_preprocessing(dependencies_graph_sentence_1,
                                                                     sentence_1_word_tokens)
        sentence_2_dep_graph = self.dependencies_graph_preprocessing(dependencies_graph_sentence_2,
                                                                     sentence_2_word_tokens)
        if len(sentence_1_dep_graph) != 0 and  len(sentence_2_dep_graph) != 0 :
            matrix = np.empty([len(sentence_1_dep_graph), len(sentence_2_dep_graph)]);
            for index_row , rdf_triple_sentence_1  in enumerate(sentence_1_dep_graph):
                for index_column, rdf_triple_sentence_2  in enumerate(sentence_2_dep_graph):
                    sentence_1_word_1, _, sentence_1_word_2 = rdf_triple_sentence_1
                    sentence_2_word_1, _, sentence_2_word_2 = rdf_triple_sentence_2
                    # step 1
                    x = self.words_similarity_obj.get_words_similarity(sentence_1_word_1, sentence_2_word_1)
                    y = self.words_similarity_obj.get_words_similarity(sentence_1_word_2, sentence_2_word_2)
                    step_1_similarity = (x + y) / 2
                    # step 2
                    x = self.words_similarity_obj.get_words_similarity(sentence_1_word_1, sentence_2_word_2)
                    y = self.words_similarity_obj.get_words_similarity(sentence_1_word_2, sentence_2_word_1)
                    step_2_similarity = (x + y) / 2
                    if step_1_similarity != 1 and step_2_similarity != 1:
                        matrix[index_row, index_column] = (step_1_similarity + step_2_similarity) / 2
                    else:
                        matrix[index_row, index_column] = 1
            _, syntactic_final_similarity = self.matrix_similarity(matrix)
        else:
            syntactic_final_similarity = 0
      
        # sentences similarity
        tot_num_words = len(sentence_1_word_tokens) + len(sentence_2_word_tokens)
        tot_num_rdf_triple = len(sentence_1_dep_graph) + len(sentence_2_dep_graph)
        sentences_similarity = (tot_num_words * lexical_final_similarity) + (tot_num_rdf_triple * syntactic_final_similarity)
        sentences_similarity = sentences_similarity / (tot_num_words + tot_num_rdf_triple)
        return  sentences_similarity
        
    def sentence_preprocessing(self, sentence):
        word_tokens = []
        number_tokens = []
        tokens = self.nlp_obj.get_tokens(sentence, remove_stop_words=True)[0]
        lemmas = self.nlp_obj.get_lemmas(' '.join(tokens))[0]
        pos_tags = self.nlp_obj.get_POS_tags(' '.join(lemmas))[0]
        for token, tag in pos_tags:
            if tag == 'CD':
                try:
                    number = float(token)
                    number_tokens.append(number)
                except ValueError:
                    word_tokens.append(token)
            else:
                word_tokens.append(token)
        return word_tokens, number_tokens
    
    def dependencies_graph_preprocessing(self, dependencies_graph, lexical_layer_tokes):
        final_dependencies_graph = Graph()
        for triple in dependencies_graph:
            triple_valid = True
            s, o, p = triple
#             lemmas_subject = self.nlp_obj.get_lemmas(s)[0]
#             lemmas_predicate = self.nlp_obj.get_lemmas(p)[0]
#             print (lemmas_subject)
#             assert len(lemmas_subject) == 1
#             print (lemmas_predicate)
#             assert len(lemmas_predicate) == 1
            if s.value not in lexical_layer_tokes:
                triple_valid = False
            if p.value  not in lexical_layer_tokes:
                triple_valid = False
            if triple_valid is True:
                final_dependencies_graph.add((s, o, p))
        return final_dependencies_graph

    def matrix_similarity(self, matrix):
        total_similarity = 0
        num_rows, num_colums, = matrix.shape
        iterations = 0
        while  num_rows != 0 and num_colums != 0:
            max_value = matrix.max()
            total_similarity += max_value
            # get indexes of max value in the matrix
            max_value_indexes = np.unravel_index(matrix.argmax(), matrix.shape)
            matrix = np.delete(matrix, max_value_indexes[0], 0) 
            matrix = np.delete(matrix, max_value_indexes[1], 1)
            num_rows, num_colums, = matrix.shape
            iterations += 1
        partial_similarity = total_similarity / iterations
        return total_similarity, partial_similarity
    
    def size_difference_penalization_coefficient(self, num_tokens_s_1, num_tokens_s_2, partial_similarity):
        if num_tokens_s_1 == num_tokens_s_2:
            return 0
        if num_tokens_s_1 > num_tokens_s_2:
            return (abs(num_tokens_s_1 - num_tokens_s_2) * partial_similarity) / num_tokens_s_1
        else: 
            return (abs(num_tokens_s_1 - num_tokens_s_2) * partial_similarity) / num_tokens_s_2
