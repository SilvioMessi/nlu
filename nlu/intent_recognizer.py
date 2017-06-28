from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.sentences_similarity import SentencesSimilarity

class IntentRecognizer():
    
    def __init__(self):
        nlp = StanfordCoreNLP()
        words_similarity = PathMeasureWordsSimilarity()
        self.sentences_similarity = SentencesSimilarity(words_similarity, nlp)
    
    def data_preprocessing(self, sentence):
        return self.sentences_similarity.data_preprocessing(sentence)
        
    def get_intents_probabilities(self, intents, sentence, data_preprocessed=False):
        intents_probabilities = {}
        for intent_id in intents:
            intents_probabilities[intent_id] = 0
            for example in intents [intent_id]:
                if data_preprocessed:
                    sentences_similarity = self.sentences_similarity.compute_similarity(example['word_tokens'],
                                                                                    example['number_tokens'],
                                                                                    sentence['word_tokens'],
                                                                                    sentence['number_tokens'],
                                                                                    example['dependencies_graph'],
                                                                                    sentence['dependencies_graph'])
                else:
                    sentences_similarity = self.sentences_similarity.get_sentences_similarity(example, sentence)
                if sentences_similarity > intents_probabilities[intent_id]:
                    intents_probabilities[intent_id] = sentences_similarity
        return intents_probabilities
