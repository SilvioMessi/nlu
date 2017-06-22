import os
import csv

from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.seeker import Seeker
from nlu.sentences_similarity import SentencesSimilarity

class Benchmark():
    
    def __init__(self):
        nlp = StanfordCoreNLP()
        words_similarity = PathMeasureWordsSimilarity()
        self.seeker = Seeker(words_similarity, nlp)
        self.sentences_similarity = SentencesSimilarity(words_similarity, nlp)
        self.test_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/test'
    
    def seek_entity_benchmark(self, entities, query, csv_name='seek_entity_benchmark.csv'):
        with open(self.test_folder_path + '/assets/' + csv_name + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['entity_id', 'value_id', 'entity_value', 'query', 'entity_recognized'])
            for entity_id in entities:
                for value_id in entities [entity_id]:
                    for example in entities [entity_id][value_id]:
                        query_tokens, free_text_positions = self.seeker.search_free_text(example, query)
                        tmp = query_tokens
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
                        csv_writer.writerow([entity_id, value_id,  example, query, ' '.join(final_tokens)])
    
    def sentences_similarity_benchmark(self, intents, query, csv_name='sentences_similarity_benchmark.csv'):
        with open(self.test_folder_path + '/assets/' + csv_name + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['intent_id', 'example', 'query', 'sentences_similarity'])
            for intent_id in intents:
                scores = []
                for example in intents [intent_id]:
                    sentences_similarity = self.sentences_similarity.get_sentences_similarity(example, query, False)
                    csv_writer.writerow([intent_id, example, query, sentences_similarity])
                    scores.append(sentences_similarity)
                csv_writer.writerow([intent_id, '', 'MEAN', sum(scores) / len(scores)])
                csv_writer.writerow([intent_id, '', 'MAX', max(scores)])