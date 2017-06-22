from nltk.corpus import wordnet as wn
import Levenshtein

class WordsNotValidException(Exception):
    pass

class BasicWordsSimilarity():
    
    def check_words(self, word_1, word_2):
        if word_1 is None or word_2 is None:
            raise WordsNotValidException
        
    def get_words_similarity(self, word_1=None, word_2=None):
        self.check_words(word_1, word_2)
        if word_1 == word_2:
            return 1
        return 0

class PathMeasureWordsSimilarity(BasicWordsSimilarity):
    
    def get_words_similarity(self, word_1=None, word_2=None):
        self.check_words(word_1, word_2)
        word_1_synsets = wn.synsets(word_1)
        word_2_synsets = wn.synsets(word_2)
        if (len(word_1_synsets) != 0 and len(word_2_synsets) != 0):
            path_similarity = wn.path_similarity(word_1_synsets[0], word_2_synsets[0])
            if path_similarity is not None and path_similarity >= 0.1:
                return round(path_similarity, 2)
        return round(1 - (Levenshtein.distance(word_1, word_2) / len(max([word_1, word_2], key=len))), 2)