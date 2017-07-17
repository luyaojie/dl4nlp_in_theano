import sys

import numpy as np

a = """
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Log Loss Epoch 24, train log loss 0.144792297855, acc 0.948544566986
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Log Loss Epoch 24, dev   log loss 0.311203974554, acc 0.870412844037
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Log Loss Epoch 24, test  log loss 0.26929580265, acc 0.886326194399
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Acc Epoch 18, train log loss 0.166407832448, acc 0.937997950943
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Acc Epoch 18, dev   log loss 0.313489458188, acc 0.880733944954
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Acc Epoch 18, test  log loss 0.284396344657, acc 0.883580450302
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Log Loss Batch 22000, dev   log loss 0.308407036205, acc 0.875
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Log Loss Batch 22000, test  log loss 0.268638483941, acc 0.886326194399
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Acc Batch 28000, dev   log loss 0.324042030861, acc 0.884174311927
06-30 13:53 [MainProcess, 16860] [INFO ]  Best Dev Acc Batch 28000, test  log loss 0.27953320705, acc 0.887424492037
"""


def read_best_result(fname):
    def get_loss_acc(line):
        att = line.strip().split(',')
        iter_num = att[-3].split()[-1]
        loss = att[-2].split()[-1]
        acc = att[-1].split()[-1]
        return int(iter_num), float(loss), float(acc)

    def add_result(result_table, line):
        if 'train' in line:
            index = 0
        elif 'dev' in line:
            index = 1
        elif 'test' in line:
            index = 2
        else:
            raise NotImplementedError
        iter_num, loss, acc = get_loss_acc(line)
        result_table[0][index] = loss
        result_table[1][index] = acc
        result_table[2][index] = iter_num

    # loss, acc, iter
    # train, dev, test
    best_loss_epoch_result = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    best_acc_epoch_result = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    best_loss_batch_result = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    best_acc_batch_result = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    # for i in xrange(1):
    #     for line in a.split('\n'):
    with open(fname, 'r') as fin:
        for line in fin:
            if "Best Dev Log Loss Epoch" in line:
                add_result(best_loss_epoch_result, line)
            if "Best Dev Acc Epoch" in line:
                add_result(best_acc_epoch_result, line)
            if "Best Dev Log Loss Batch" in line:
                add_result(best_loss_batch_result, line)
            if "Best Dev Acc Batch" in line:
                add_result(best_acc_batch_result, line)
    return {'best_loss_epoch': best_loss_epoch_result,
            'best_acc_epoch': best_acc_epoch_result,
            'best_loss_batch': best_loss_batch_result,
            'best_acc_batch': best_acc_batch_result, }


def pprint_result(result, prefix=''):
    sys.stdout.write("[%s%strain]\t%s\t%s at iter %s\n" % (prefix, ' ' if prefix != "" else "",
                                                           result[0][0], result[1][0], result[2][0]))
    sys.stdout.write("[%s%sdev  ]\t%s\t%s\n" % (prefix, ' ' if prefix != "" else "",
                                                result[0][1], result[1][1]))
    sys.stdout.write("[%s%stest ]\t%s\t%s\n" % (prefix, ' ' if prefix != "" else "",
                                                result[0][2], result[1][2]))


def get_opt_param(filenames):
    select_type = 'best_acc_epoch'
    dev_loss = list()
    dev_acc = list()
    for fname in filenames:
        result = read_best_result(fname)[select_type]
        dev_loss.append(result[0][1])
        dev_acc.append(result[1][1])
    best_loss_index = np.argmin(dev_loss)
    best_acc_index = np.argmax(dev_acc)
    pprint_result(read_best_result(filenames[best_loss_index])[select_type], prefix='best loss')
    pprint_result(read_best_result(filenames[best_acc_index])[select_type], prefix='best acc')


if __name__ == "__main__":
    files = list()
    for filename in open(sys.argv[1]):
        files.append(filename.strip())
    get_opt_param(files)
