import os
import unittest
import json
from time import time

from nlu.nlp import StanfordCoreNLP
from nlu.words_similarity import PathMeasureWordsSimilarity
from nlu.sentences_similarity import SentencesSimilarity
from nlu.seeker import Seeker
from nlu.benchmark import Benchmark
from nlu.entity_recognizer import EntityRecognizer
from nlu.intent_recognizer import IntentRecognizer

class TestSuite(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestSuite, self).__init__(*args, **kwargs)
        words_similarity = PathMeasureWordsSimilarity()
        nlp = StanfordCoreNLP()
        self.sentences_similarity = SentencesSimilarity(words_similarity, nlp)
        self.seeker = Seeker(words_similarity, nlp)
        self.benchmark = Benchmark()
        self.entity_recognizer = EntityRecognizer()
        self.intent_recognizer = IntentRecognizer()
        self.test_folder_path = os.path.dirname(os.path.abspath(__file__))
        # entities data store
        self.entities = {}
        self.preprocessed_entities = {} 
        with open(self.test_folder_path + '/assets/entities.json') as json_file:    
            json_obj = json.load(json_file)
            for entity in json_obj:
                entity_id = entity['id']
                self.entities[entity_id] = {}
                self.preprocessed_entities[entity_id] = {}
                for entry in entity['entries']:
                    value_id = entry['value']
                    self.entities[entity_id][value_id] = [value_id]
                    preprocessed_data = {}
                    preprocessed_data['tokens'], preprocessed_data['lemmas'] = self.seeker.data_preprocessing(value_id)
                    self.preprocessed_entities[entity_id][value_id] = [preprocessed_data]
                    for synonymous in entry['synonyms']:
                        self.entities[entity_id][value_id].append(synonymous)
                        preprocessed_data = {}
                        preprocessed_data['tokens'], preprocessed_data['lemmas'] = self.seeker.data_preprocessing(synonymous)
                        self.preprocessed_entities[entity_id][value_id] = [preprocessed_data]
        
        # intents data store
        self.intents_raw = {}
        self.intents_raw_big = {}
        self.intents_tagged = {}
        self.preprocessed_intents = {}
        with open(self.test_folder_path + '/assets/intents.json') as json_file:    
            json_obj = json.load(json_file)
            for intent in json_obj:
                intent_id = intent['id']
                self.intents_raw[intent_id] = []
                self.intents_tagged[intent_id] = []
                self.preprocessed_intents[intent_id] = []
                for entry in intent['entries']:
                    sentence_raw = ''
                    sentence_entities_tagged = ''
                    for item in entry['value']:
                        if 'entity' in item:
                            sentence_entities_tagged += item['entity']
                        else:
                            sentence_entities_tagged += item['text']
                        sentence_raw += item['text']
                    self.intents_raw[intent_id].append(sentence_raw)
                    self.intents_tagged[intent_id].append(sentence_entities_tagged)
                    preprocessed_data = {}
                    preprocessed_data ['word_tokens'], preprocessed_data ['number_tokens'] , preprocessed_data ['dependencies_graph'] = self.intent_recognizer.data_preprocessing(sentence_raw)
                    self.preprocessed_intents[intent_id].append(preprocessed_data)
        
        with open(self.test_folder_path + '/assets/sentences.json') as json_file:    
            json_obj = json.load(json_file)
            for domain in json_obj['domains']:
                for intent in domain['intents']:
                    intent_id = intent['name']
                    self.intents_raw_big[intent_id] = []
                    queries = intent['queries']
                    for index in range (0, len(queries) - 1):
                        self.intents_raw_big[intent_id].append(queries[index]['text'])

    
    def test_sentences_similarity_simple(self):
        sentence_1 = 'Ruffner, 45, doesn\'t yet have an attorney in the murder charge, authorities said.'
        sentence_2 = 'Ruffner, 45, does not have a lawyer on the murder charge, authorities said.'
        assert self.sentences_similarity.get_sentences_similarity(sentence_1, sentence_2) > 0.89
        
        sentence_1 = 'The Commerce Commission would have been a significant hurdle for such a deal.'
        sentence_2 = 'The New Zealand Commerce Commission had given Westpac no indication whether it would have approved its deal.'
        assert self.sentences_similarity.get_sentences_similarity(sentence_1, sentence_2) != 1
    
    def test_sentences_similarity_form_json(self):
        with open(self.test_folder_path + '/assets/sentences.json') as json_file:    
            json_obj = json.load(json_file)
            for domain in json_obj['domains']:
                for intent in domain['intents']:
                    queries = intent['queries']
                    for index in range (0, len(queries) - 1):
                        sentence_1 = queries[index]['text']
                        sentence_2 = queries[index + 1]['text']
                        self.sentences_similarity.get_sentences_similarity(sentence_1, sentence_2)
    
    def test_seeker_simple(self):
        sentence = 'Can i have two pizzas margherita and a can of beer?'
        print(self.seeker.search_free_text('margherita', sentence))
        print(self.seeker.search_free_text('beer', sentence))
        print(self.seeker.search_free_text('beers', sentence))
        print(self.seeker.search_free_text('2', sentence))
        print(self.seeker.search_free_text('coca cola', sentence))
        sentence = 'Two cans of coke, tree beers, five pizzas margherita and more coca cola!'
        print(self.seeker.search_free_text('margherita', sentence))
        print(self.seeker.search_free_text('beer', sentence))
        print(self.seeker.search_free_text('beers', sentence))
        print(self.seeker.search_free_text('2', sentence))
        print(self.seeker.search_free_text('coca cola', sentence))
    
    def test_benchmark(self):
        self.benchmark.seek_entity_benchmark(self.entities, 'Can i have a pizza margherita, a pizza napoletan, two beer and a can of coca cola?', 'entities_benchmark')
        self.benchmark.sentences_similarity_benchmark(self.intents_raw, 'Can i have a pizza prosciutto?', 'intents_benchmark_raw')
        self.benchmark.sentences_similarity_benchmark(self.intents_tagged, 'Can i have a pizza pizza_type?', 'intents_benchmark_tagged')
        self.benchmark.sentences_similarity_benchmark(self.intents_raw_big, 'What is the best bar in London?', 'intents_benchmark_raw_big')        
    
    def test_entities_position(self):
        sentence = 'Can i have a pizza margherita, a pizza margherita, two beer and a can of coca cola?'
        sentence_tokens, entities_positions = self.entity_recognizer.get_entities(self.entities, sentence)
        print(sentence_tokens)
        print(entities_positions)
            
    def test_entities_intents_recognizer(self):
        sentences = ['Can i have a pizza margherita, a pizza napoletan, two beer and a can of coca cola?']
        for sentence in sentences :
            sentence_tagged = self.entity_recognizer.tag_sentence(self.entities, sentence)
            print (sentence_tagged)
            print (self.intent_recognizer.get_intents_probabilities(self.intents_tagged, sentence))
            print (self.intent_recognizer.get_intents_probabilities(self.intents_tagged, sentence_tagged))
            
    def test_cache(self):
        sentence = 'Can i have a pizza margherita, a pizza napoletan, two beer and a can of coca cola?'       
        preprocessed_sentence = {}
        preprocessed_sentence ['word_tokens'], preprocessed_sentence ['number_tokens'], preprocessed_sentence ['dependencies_graph'] = self.intent_recognizer.data_preprocessing(sentence)
        
        start = time()
        self.intent_recognizer.get_intents_probabilities(self.intents_tagged, sentence)
        elapsed = time() - start
        print ('execution time without cache %d' % elapsed)
        
        start = time()
        self.intent_recognizer.get_intents_probabilities(self.preprocessed_intents, preprocessed_sentence, data_preprocessed=True)
        elapsed = time() - start
        print ('execution time using cache %d' % elapsed)
        
        preprocessed_sentence = {}
        preprocessed_sentence ['tokens'], preprocessed_sentence ['lemmas'] = self.seeker.data_preprocessing(sentence)
        
        start = time()
        self.entity_recognizer.get_entities(self.entities, sentence)
        elapsed = time() - start
        print ('execution time without cache %d' % elapsed)
        
        start = time()
        self.entity_recognizer.get_entities(self.preprocessed_entities, preprocessed_sentence, data_preprocessed=True)
        elapsed = time() - start
        print ('execution time using cache %d' % elapsed)
        
if __name__ == "__main__":
    unittest.main()
