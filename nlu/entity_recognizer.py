from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.seeker import Seeker

class EntityRecognizer():
    
    def __init__(self):
        nlp = StanfordCoreNLP()
        words_similarity = PathMeasureWordsSimilarity()
        self.seeker = Seeker(words_similarity, nlp)
        
    def data_preprocessing(self, sentence):
        return self.seeker.data_preprocessing(sentence)

    def get_entities(self, entities, sentence, data_preprocessed=False, tag_sentence=False):
        entities_positions = []
        position_index = 0
        for entity_id in entities:
            for value_id in entities [entity_id]:
                for example in entities [entity_id][value_id]:
                    if data_preprocessed:
                        sentence_tokens, free_text_positions = self.seeker.search(example['lemmas'], sentence['tokens'], sentence['lemmas'])
                    else:
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
        tagged_sentence = ''
        if tag_sentence:
            final_raw_sentence = ' '.join(sentence_tokens)
            ordered_positions = sorted(final_entities_positions, key=lambda k: k['start'])
            if len(ordered_positions) == 0:
                tagged_sentence = final_raw_sentence
            text_index = 0
            for index, position in enumerate(ordered_positions):
                start = position['start']
                end = position['end']
                entity_id = position['entity_id']
                tagged_sentence += final_raw_sentence[text_index:start]
                tagged_sentence += entity_id
                text_index = end
                if index == len(ordered_positions) - 1:
                    tagged_sentence += final_raw_sentence[end:]
        return sentence_tokens, final_entities_positions, tagged_sentence
