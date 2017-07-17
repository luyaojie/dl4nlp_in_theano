import itertools


def get_cmd(encoder='lstm', seed=1993, batch=25, hidden='100_50', dropout=0.2,
            pre='google', norm=9, optmizer='adagrad', lr=1, emblr=0.1, epoch=25):
    raw_cmd = ["python text_classify.py "]
    raw_cmd.append("--encoder %s" % encoder)
    raw_cmd.append("--seed %s" % seed)
    raw_cmd.append("--batch %s" % batch)
    raw_cmd.append("--hidden %s" % hidden)
    raw_cmd.append("--dropout %s" % dropout)
    raw_cmd.append("--pre %s" % pre)
    raw_cmd.append("--norm %s" % norm)
    raw_cmd.append("--optimizer %s" % optmizer)
    raw_cmd.append("--lr %s" % lr)
    raw_cmd.append("--emblr %s" % emblr)
    raw_cmd.append("--epoch %s" % epoch)
    return " ".join(raw_cmd)


def get_all_cmd():
    cmd_list = list()
    encoder_list = 'cbow lstm bilstm cnn'.split()
    seed_list = [1993, 2016, 3435]
    batch_list = [32, 64, 128]
    hidden_list = ['128', '256', '128_128', '256_256', '256_128']
    dropout_list = [0.2, 0.5, 0]
    pre_list = ['google']
    norm_list = [16]
    optimizer_list = ['adadelta']
    lr_list = [1]
    emblr_list = [0.01, 0.1, 0]
    epoch_list = [25]
    for x in itertools.product(encoder_list, seed_list, batch_list, hidden_list, dropout_list,
                               pre_list, norm_list, optimizer_list, lr_list, emblr_list, epoch_list):
        cmd_list.append(get_cmd(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]))
    return cmd_list


def split_to_gpus(cmd_list, gpu_list, float_type='float32', cnmem=0.45):
    each_gpu_num = len(cmd_list) / len(gpu_list)
    new_cmd_list = list()
    for gpu_i, gpu_name in enumerate(gpu_list):
        prefix = "THEANO_FLAGS='floatX=%s,device=%s,lib.cnmem=%s'" % (float_type, gpu_name, cnmem)
        if gpu_i == len(gpu_list) - 1:
            temp_cmd_list = cmd_list[gpu_i * each_gpu_num:]
        else:
            temp_cmd_list = cmd_list[gpu_i * each_gpu_num:(gpu_i + 1) * each_gpu_num]
        for cmd in temp_cmd_list:
            new_cmd_list.append("%s %s" % (prefix, cmd))
    return new_cmd_list


if __name__ == "__main__":
    all_cmd = get_all_cmd()
    for cmd in split_to_gpus(all_cmd, ['cuda3', 'cuda4', 'cuda5']):
        print cmd
