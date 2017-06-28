import numpy as np

class Seeker():
    
    def __init__(self, words_similarity_obj, nlp_obj):
        self.words_similarity_obj = words_similarity_obj
        self.nlp_obj = nlp_obj

    def search_free_text(self, free_text=None, sentence=None):
        _, text_lemmas = self.data_preprocessing(free_text)
        sentence_tokens, sentence_lemmas = self.data_preprocessing(sentence)
        return self.search(text_lemmas, sentence_tokens, sentence_lemmas)
    
    def search(self,
               text_lemmas=[],
               sentence_tokens=[],
               sentence_lemmas=[]):
        # position (tokens index) of all occurrence found
        free_text_positions = []
        search = True
        while search:
            num_lemmas_text = len(text_lemmas)
            num_lemmas_s = len(sentence_lemmas)
            if num_lemmas_text != 0 and  num_lemmas_s != 0 :
                matrix = np.empty([num_lemmas_text, num_lemmas_s]);
                for index_row , lemma_1 in enumerate(text_lemmas):
                    for index_column, lemma_2 in enumerate(sentence_lemmas):
                        matrix[index_row, index_column] = self.words_similarity_obj.get_words_similarity(lemma_1, lemma_2)
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
        return  sentence_tokens, free_text_positions
        
    def data_preprocessing(self, sentence):
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