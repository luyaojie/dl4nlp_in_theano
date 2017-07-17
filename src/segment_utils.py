# -*- coding: utf-8 -*-


class ChineseWordSegmentor(object):
    def __init__(self, model='jieba'):
        self.model = model
        if model.lower() == 'jieba':
            import jieba.posseg as posseg
            posseg.initialize()
            self.segmentor = posseg.POSTokenizer(tokenizer=None)
        elif model.lower() == 'ictclas':
            import pynlpir
            pynlpir.open()
            self.segmentor = pynlpir
        else:
            raise NotImplementedError

    def segment(self, text, pos=False):
        """
        The segmented tokens are returned as a list. Each item of the list is a
        string if *pos* is `False`, e.g. ``['我们', '是', ...]``. If
        *pos* is `True`, then each item is a tuple (``(token, pos)``), e.g.
        ``[('我们', 'pronoun'), ('是', 'verb'), ...]``.
        :param text:
        :param pos:
        :return:
        """
        if self.model.lower() == 'jieba':
            if pos:
                return [token for token in self.segmentor.cut(text)]
            else:
                return [tuple(token)[0] for token in self.segmentor.cut(text)]
        elif self.model.lower() == 'ictclas':
            if pos:
                return self.segmentor.segment(text, pos_tagging=True)
            else:
                return self.segmentor.segment(text, pos_tagging=False)
        else:
            raise NotImplementedError
