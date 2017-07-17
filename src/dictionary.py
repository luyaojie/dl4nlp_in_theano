from __init__ import OOV_KEY


class Dictionary(object):
    """
    A Class for Manage Word Dictionary by count/top setting.
    """
    def __init__(self):
        # Word -> Index, index start from 1
        self.word_index = dict()
        # Index -> Word, index start from 1
        self.index_word = dict()
        # Word -> Count
        self.word_count = dict()
        self.add_word(OOV_KEY)

    def __add__(self, d):
        """
        Merge two Dictionary
        :param d: 
        :return: 
        """
        assert type(d) == Dictionary
        word_set = set(self.word_index.keys()) | set(d.word_index.keys())
        new_d = Dictionary()
        for w in word_set:
            new_d.index_word[len(new_d.word_index) + 1] = w
            new_d.word_index[w] = len(new_d.word_index) + 1
            new_d.word_count[w] = 0
            if w in self.word_count:
                new_d.word_count[w] += self.word_count[w]
            if w in d.word_count:
                new_d.word_count[w] += d.word_count[w]
        return new_d

    def __getitem__(self, word):
        return self.word_index[word]

    def __contains__(self, word):
        return word in self.word_index

    def oov_index(self):
        return self.word_index[OOV_KEY]

    def add_word(self, word):
        """
        Add word to Dictionary
        :param word: 
        """
        if word not in self.word_index:
            self.index_word[len(self.word_index) + 1] = word
            self.word_index[word] = len(self.word_index) + 1
            self.word_count[word] = 0
        self.word_count[word] += 1

    def cut_by_top(self, top_k=30000):
        """
        Cut Dictionary by Top Count
        :param top_k: 
        """
        if len(self.word_index) <= top_k:
            print "Word number (%s) is smaller Top K (%s)" % (len(self.word_index), top_k)
        w_c = [(self.word_count[w], self.word_index[w]) for w in self.word_index.keys()]
        w_c.sort(reverse=True)
        new_word_index = dict()
        new_index_word = dict()
        new_word_count = dict()
        for _, index in w_c[:top_k]:
            w = self.index_word[index]
            new_index_word[len(new_word_index) + 1] = w
            new_word_index[w] = len(new_word_index) + 1
            new_word_count[w] = self.word_count[w]
        self.word_index = new_word_index
        self.index_word = new_index_word
        self.word_count = new_word_count
        if OOV_KEY not in self.word_index:
            self.add_word(OOV_KEY)


    def cut_by_count(self, min_count=1, max_count=None):
        """
        Cut Dictionary by Count
        :param min_count: 
        :param max_count: 
        """
        new_word_count = dict()
        for w, c in self.word_count.iteritems():
            if max_count is not None:
                if c > max_count:
                    continue
            if c >= min_count:
                new_word_count[w] = c
        new_word_index = dict()
        new_index_word = dict()
        for w in new_word_count.keys():
            new_index_word[len(new_word_index) + 1] = w
            new_word_index[w] = len(new_word_index) + 1
        self.word_index = new_word_index
        self.index_word = new_index_word
        self.word_count = new_word_count
        if OOV_KEY not in self.word_index:
            self.add_word(OOV_KEY)

    def write_to_file(self, filename):
        with open(filename, 'w') as out:
            for word, index in self.word_index.iteritems():
                write_str = "%s %s\n" % (word, index)
                out.write(write_str.encode('utf8'))
