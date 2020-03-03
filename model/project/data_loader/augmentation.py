import random

class RandomShuffle(object):
    def __call__(self, sample):
        if(0.25 > random.random()):
            l = list(sample)
            random.shuffle(l)
            sample = ''.join(l)
        return sample

class RandomWordShuffle(object):
    def __call__(self, sample):
        if(0.25 > random.random()):
            l = sample.split(' ')
            random.shuffle(l)
            sample = ' '.join(l)
        return sample

class RandomCharDelete(object):
    def __call__(self, sample):
        l = list(sample)
        l = ['' if 0.05 > random.random() else x for x in l ]
        sample = ''.join(l)
        return sample

class RandomStarPlace(object):
    def __call__(self, sample):
        l = list(sample)
        l = [self.rand_star(x) for x in l]
        sample = ''.join(l)
        return sample
    
    def rand_star(self, x):
        r = random.random()
        if r < 0.05:
            return '*'
        return x

class StringVectorizer(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, text):
        parsed_vector = [[0 if char != letter else 1 for char in self.alphabet] for letter in text]
        return parsed_vector

class SequencePadder(object):
    def __init__(self, length, alphabet):
        self.length = length
        self.alphabet = alphabet

    def __call__(self, seq):
        seq += [[0]*len(self.alphabet) for _ in range(self.length-len(seq))]
        return seq