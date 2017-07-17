import sys


def load_wordmap(encoding='utf8'):
    wordmap = dict()
    with sys.stdin as fin:
        for line in fin:
            att = line.decode(encoding).strip().split('\t')[1:]
            for word in ' '.join(att).split():
                if word not in wordmap:
                    wordmap[word] = 1
                else:
                    wordmap[word] += 1
    return wordmap


if __name__ == "__main__":
    word_map = load_wordmap()
    word_map = sorted(word_map.iteritems(), key=lambda d: d[1], reverse=True)
    with sys.stdout as out:
        for index, word_count in enumerate(word_map):
            write_str = "%s\t%s\t%s\n" % (word_count[0], index, word_count[1])
            out.write(write_str.encode('utf8'))
