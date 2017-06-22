from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.sentences_similarity import SentencesSimilarity

class IntentRecognizer():
    
    def __init__(self):
        nlp = StanfordCoreNLP()
        words_similarity = PathMeasureWordsSimilarity()
        self.sentences_similarity = SentencesSimilarity(words_similarity, nlp)
    
    def get_intents_probabilities(self, intents, sentence):
        intents_probabilities = {}
        for intent_id in intents:
            intents_probabilities[intent_id] = 0
            for example in intents [intent_id]:
                sentences_similarity = self.sentences_similarity.get_sentences_similarity(example, sentence, False)
                if sentences_similarity > intents_probabilities[intent_id]:
                    intents_probabilities[intent_id] = sentences_similarity
        return intents_probabilities