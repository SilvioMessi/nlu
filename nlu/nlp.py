from pycorenlp import StanfordCoreNLP as s_c_nlp
from rdflib import Literal, Graph

from nlu.parameters import s_c_nlp_web_service_url
from nlu.parameters import english_stop_words_list

class BasicNLP():
    
    def get_tokens(self, text, remove_stop_words=False):
        pass
    
    def get_lemmas(self, text):
        pass
    
    def get_POS_tags(self, text):
        pass
    
    def get_dependecies_graph(self, text):
        pass
    
    
class StanfordCoreNLP(BasicNLP):
    
    def __init__(self):
        self.nlp = s_c_nlp(s_c_nlp_web_service_url)
    
    def get_tokens(self, text, remove_stop_words=False):
        output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit',
            'outputFormat': 'json'
        })
        sentences = []
        for sentence in output['sentences']:
            tokens = []
            for token in sentence['tokens']:
                if remove_stop_words:
                    if token['word'].lower() not in english_stop_words_list:
                        tokens.append(token['word'])   
                else:
                    tokens.append(token['word'])
            sentences.append(tokens)
        return sentences
    
    def get_lemmas(self, text):
        output = self.nlp.annotate(text, properties={
            'annotators': 'lemma',
            'outputFormat': 'json'
        })
        sentences = []
        for sentence in output['sentences']:
            lemmas = []
            for token in sentence['tokens']:
                lemmas.append(token['lemma'])
            sentences.append(lemmas)
        return sentences
    
    def get_POS_tags(self, text):
        output = self.nlp.annotate(text, properties={
            'annotators': 'pos',
            'outputFormat': 'json'
        })
        sentences = []
        for sentence in output['sentences']:
            pos_tags = []
            for token in sentence['tokens']:
                pos_tags.append((token['originalText'], token['pos']))
            sentences.append(pos_tags)
        return sentences
    
    def get_dependecies_graph(self, text):
        output = self.nlp.annotate(text, properties={
            'annotators': 'depparse',
            'outputFormat': 'json'
        })
        sentences = []
        for sentence in output['sentences']:
            elements = sentence['basicDependencies']
            dependencies_graph = Graph()
            # create all vertices
            vertices = {}
            for element in elements:
                vertices[element['dependent']] = Literal(element['dependentGloss'])
            # create all edges
            for element in elements:
                if element['governor'] != 0:
                    dependencies_graph.add((vertices[element['governor']], Literal(element['dep']), vertices[element['dependent']]))
                sentences.append(dependencies_graph)
        return sentences