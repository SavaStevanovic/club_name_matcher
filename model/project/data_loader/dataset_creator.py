import os 
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader.iterator import StringDataset
from torchvision import transforms
from data_loader.data_provider import getNamingDictFromFile, getUniversalNames

class DatasetCreator:
    def __init__(self, root_dir, names_file):
        self.universal_names = getUniversalNames(names_file)
        self.data = self.parse_data(root_dir)
        self.corpus, self.ocurences = self.get_corpus(self.data)
        self.encoded_data = [self.string_vectorizer(x, self.corpus) for x in self.data]
        self.train_data, self.validation_data = train_test_split(self.encoded_data, test_size=0.2)

    def string_vectorizer(self, text_vector, alphabet):
        parsed_vector = [[[0 if char != letter else 1 for char in alphabet] for letter in text] for text in text_vector]
        return parsed_vector

    def get_corpus(self, data):
        corp_list = []
        for x in data:
            corp_list += x
        joined_text = "".join(corp_list)
        corpus = set(joined_text)
        ocurences = {x:joined_text.count(x) for x in corpus}
        # ocurences1 = sorted(ocurences.items(), key=lambda kv: kv[1])
        return corpus, ocurences

    def parse_data(self, root_dir):
        self.mapping_dict = {}
        for x in os.listdir(root_dir):
            getNamingDictFromFile(x, self.universal_names, self.mapping_dict) 
        for x in self.mapping_dict:
            self.mapping_dict[x] = list(set(self.mapping_dict[x]))
        mappings = list(self.mapping_dict.values())

        return mappings

    def get_train_iterator(self, transform=None):
        return StringDataset(self.train_data, transform)

    def get_validation_iterator(self, transform=None):
        return StringDataset(self.validation_data, transform) 