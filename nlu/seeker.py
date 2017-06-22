import sys, os
import numpy as np

from nlu.utils import print_matrix

class Seeker():
    
    def __init__(self, words_similarity_obj, nlp_obj):
        self.words_similarity_obj = words_similarity_obj
        self.nlp_obj = nlp_obj

    def search_free_text(self, free_text=None, sentence=None, print_logs=False):
        # disable printing on console
        if print_logs is False:
            sys.stdout = open(os.devnull, 'w')
        # position (tokens index) of all occurrence found
        free_text_positions = []
        
        _, text_lemmas = self.preprocessing(free_text)
        sentence_tokens, sentence_lemmas = self.preprocessing(sentence)
        
        search = True
        while search:
            num_lemmas_text = len(text_lemmas)
            num_lemmas_s = len(sentence_lemmas)
            if num_lemmas_text != 0 and  num_lemmas_s != 0 :
                matrix = np.empty([num_lemmas_text, num_lemmas_s]);
                for index_row , lemma_1 in enumerate(text_lemmas):
                    for index_column, lemma_2 in enumerate(sentence_lemmas):
                        matrix[index_row, index_column] = self.words_similarity_obj.get_words_similarity(lemma_1, lemma_2)
                print_matrix(matrix, text_lemmas, sentence_lemmas)
                occurrence = self.matrix_similarity(matrix)
                if len(occurrence) > 0:
                    free_text_positions.append(occurrence)
                    # replace occurrence
                    for position in occurrence:
                        sentence_lemmas[position] = '###occurrence###'
                else:
                    search = False
            else:
                search = False       
        # enable printing on console
        if print_logs is False:
            sys.stdout = sys.__stdout__
        return  sentence_tokens, free_text_positions
        
    def preprocessing(self, sentence):
        tokens = self.nlp_obj.get_tokens(sentence, remove_stop_words=False)[0]
        lemmas = self.nlp_obj.get_lemmas(' '.join(tokens))[0]
        assert len(tokens) == len(lemmas)
        return tokens, lemmas
                
    def matrix_similarity(self, matrix):
        num_rows, num_colums, = matrix.shape
        free_text_tokens_len = num_rows
        free_text_position = []
        while  num_rows != 0 and num_colums != 0:
            max_value = matrix.max()
            # get indexes of max value in the matrix
            max_value_indexes = np.unravel_index(matrix.argmax(), matrix.shape)
            # delete row of best matching token of free text
            matrix = np.delete(matrix, max_value_indexes[0], 0) 
            num_rows, num_colums, = matrix.shape
            if max_value >= 0.9:
                free_text_position.append(max_value_indexes[1])
        if len(free_text_position) != free_text_tokens_len:
            return []
        if len(free_text_position) == 0 or len(free_text_position) == 1:
            return free_text_position
        # check if token are sequential 
        free_text_position.sort()
        if all(free_text_position[i] == free_text_position[i + 1] - 1 for i in range(len(free_text_position) - 1)):
            return free_text_position
        return []