from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.seeker import Seeker

class EntityRecognizer():
    
    def __init__(self):
        nlp = StanfordCoreNLP()
        words_similarity = PathMeasureWordsSimilarity()
        self.seeker = Seeker(words_similarity, nlp)
        
    def tag_sentence(self, entities, sentence):
        for entity_id in entities:
            for value_id in entities [entity_id]:
                for example in entities [entity_id][value_id]:
                    sentence_tokens, free_text_positions = self.seeker.search_free_text(example, sentence)
                    tmp = sentence_tokens
                    for position in free_text_positions:
                        first = True
                        for index in position :
                            if first == True:    
                                tmp[index] = entity_id
                                first = False
                            else:
                                # occurrence positions of free text are absolute. Tokens array length cannot change!
                                tmp[index] = '###placeholder###'
                    final_tokens = [x for x in tmp if x != '###placeholder###']
                    sentence = ' '.join(final_tokens)
        return sentence
    
    def get_entities(self, entities, sentence):
        entities_positions = []
        position_index = 0
        for entity_id in entities:
            for value_id in entities [entity_id]:
                for example in entities [entity_id][value_id]:
                    sentence_tokens, free_text_positions = self.seeker.search_free_text(example, sentence)
                    text_length = 0
                    for index, token in enumerate(sentence_tokens):
                        for position in free_text_positions:
                            if index in position:
                                try:
                                    entities_positions[position_index]
                                except IndexError:
                                    entities_positions.append({'entity_id': entity_id, 'value_id': value_id, 'start': text_length})
                                if index + 1 not in position:
                                    entities_positions[position_index]['end'] = text_length + len(token)
                                    position_index += 1
                            # space between tokens
                            text_length += len(token) + 1
        # remove duplicate
        final_entities_positions = [dict(t) for t in set([tuple(d.items()) for d in entities_positions])]
        return sentence_tokens, final_entities_positions
