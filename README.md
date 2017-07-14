# NLU
Python library for NLU (Natural Language Understanding).
This python library allows to identify intents and entities present in a sentence, starting from a train set of examples. 

## Installation
``` 
pip install git+https://github.com/SilvioMessi/nlu.git 
```
### Dependencies
*  Stanford CoreNLP
At the moment all the NLP (Natural Language Processing) task are made by Stanford CoreNLP library, used like external [server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html).
Before use NLU library make sure that an instance of CoreNLP Server is correctly running on port 9000.
* Wordnet
The NLU library use [Wordnet](https://wordnet.princeton.edu/). 
Before use NLU library make sure that Wordnet corpora is correctly "intstalled" by [NTLK](http://www.nltk.org/data.html)

## Basic Usage
```python
from nlu.entity_recognizer import EntityRecognizer
from nlu.intent_recognizer import IntentRecognizer

entity_recongizer = EntityRecognizer()
intent_recognizer = IntentRecognizer()
    
# load entities definitions from datastore and put them in data structure
 entities = {
  'pizza_type': {
    'Margherita' : ['Margherita', 'margherita'],
    'Neapolitan' : ['Neapolitan', 'neapolitan'],
    'Sicilian' : ['Sicilian', 'sicilian']
  },
  'drink_type' : {
    'Coca-Cola' : ['Coca-Cola', 'coca cola', 'coke'],
    'Beer' : ['Beer', 'beer'],
    'Water' : ['Water', 'water'],
    'Wine' : ['Wine', 'wine']
  }
}

# load intents definitions from datastore and put them in data structure
intents = {
  'order_a_pizza': ['Can i have a pizza margherita?', 'A pizza margherita, please !', 'Two margherita please!'],
  'order_a_drink': ['Can i have a can of coke?', 'I\'ll have a glass of wine!']
}

# intents definitions can be pre tagged with entities
intents_pre_tagged = {
  'order_a_pizza': ['Can i have a pizza pizza_type?', 'A pizza pizza_type, please !', 'Two pizza_type please!'],
  'order_a_drink': ['Can i have a can of drink_type?', 'I\'ll have a glass of drink_type!']
}

new_sentence = 'I want order a pizza margherita and a can of coca cola!'

# find entities in new sentence
sentence_tokens, final_entities_positions, tagged_sentence = entity_recongizer.get_entities(entities, new_sentence, tag_sentence=True)

print(sentence_tokens)
['I', 'want', 'order', 'a', 'pizza', 'margherita', 'and', 'a', 'can', 'of', 'coca', 'cola', '!']

print(final_entities_positions)
[{'entity_id': 'pizza_type', 'value_id': 'Margherita', 'start': 21, 'end': 31}, {'entity_id': 'drink_type', 'value_id': 'Coca-Cola', 'start': 45, 'end': 54}]

# if parameter tag_sentence=True
print(tagged_sentence)
I want order a pizza pizza_type and a can of drink_type !

# find intents probabilities 
intents_probabilities = intent_recognizer.get_intents_probabilities(intents, new_sentence)

print(intents_probabilities)
{'order_a_pizza': 0.51000000000000001, 'order_a_drink': 0.161}

# better performance can be achieved used tagged sentences
intents_probabilities = intent_recognizer.get_intents_probabilities(intents_pre_tagged, tagged_sentence)

print(intents_probabilities)
{'order_a_pizza': 0.53653846153846152, 'order_a_drink': 0.23928846153846156}

```
