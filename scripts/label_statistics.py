import sys


def load_label():
    label_count = dict()
    with sys.stdin as fin:
        for line in fin:
            y = line.strip().split('\t')[0]
            if y not in label_count:
                label_count[y] = 1
            else:
                label_count[y] += 1
    return label_count


if __name__ == "__main__":
    label_num = load_label()
    sum_label_num = sum(label_num.values())
    label_num = sorted(label_num.iteritems(), key=lambda d: d[1], reverse=True)
    count = 0
    for label, number in label_num:
        count += number
        sys.stdout.write("%s\t%s\t%.2f\t%.2f\n" % (label, number, float(number) / float(sum_label_num) * 100,
                                                   float(count) / float(sum_label_num) * 100))
