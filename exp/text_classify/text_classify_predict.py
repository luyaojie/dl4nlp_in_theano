import sys
from text_classify import TextClassifier, MAX_LEN, REMOVE_STOP, LOW_CASE
sys.path.append('../../')
from src.utils import load_model, generate_sentence_token, array2str
from src.segment_utils import ChineseWordSegmentor


def text_classify_predict_main(model_name, predict_file, output_file=None, seg=True,
                               encoding='utf8', language='english'):
    sys.stdout.write("Load model ...\n")
    model = load_model(model_name)
    sys.stdout.write("Loaded model from %s\n" % model_name)
    assert type(model) == TextClassifier
    texts = list()
    if seg:
        word_segmentor = ChineseWordSegmentor('ictclas')
    else:
        word_segmentor = None
    sys.stdout.write("Loaded Data from %s ...\n" % predict_file)
    with open(predict_file) as fin:
        for line in fin:
            line = line.decode(encoding=encoding).strip()
            if seg:
                token = ' '.join(word_segmentor.segment(line))
            else:
                token = line
            token = generate_sentence_token(token, max_len=MAX_LEN, remove_stop=REMOVE_STOP,
                                            low_case=LOW_CASE, language=language)
            texts.append(token)
    sys.stdout.write("Predict Data ...\n")
    prob_result = [model.predict_text_prob(' '.join(token)) for token in texts]
    if output_file is None:
        out = sys.stdout
    else:
        sys.stdout.write("Save Result to %s \n" % output_file)
        out = open(output_file, 'w')
    for prob in prob_result:
        write_str = model.prob_to_str(prob[0]) + '\n'
        out.write(write_str.encode('utf8'))
    if output_file is not None:
        out.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model File Name')
    parser.add_argument('--input', type=str, default=None, help='Input File Name')
    parser.add_argument('--output', type=str, default=None, help='Output File Name')
    parser.add_argument('--seg', action="store_true", dest='seg', default=False)
    parser.add_argument('--encoding', type=str, default='utf8', help='Input File Encoding, Default is UTF8')
    parser.add_argument('--language', type=str, default='en', help='Input File Language, Default is English.'
                                                                   '(english, chinese)')
    args = parser.parse_args()
    text_classify_predict_main(model_name=args.model, predict_file=args.input, output_file=args.output,
                               seg=args.seg, encoding=args.encoding, language=args.language)
