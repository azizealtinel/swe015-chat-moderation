"""Dört modelin (artı kural tabanlı sözlüğün) yüklenmesi ve tek mesaj için tahmin.

Her model predict(text) -> (etiket, güven) döndürür. etiket: 0 = clean, 1 = offensive.
"""
import os
import pickle
import time

import joblib
import torch
import torch.nn as nn

from .textproc import normalize_full, normalize_light, lexicon_hits

MAX_LEN = 64


# ---------- Alan A: word2vec embedding'leri üzerinde feedforward ağ ----------
class FFN(nn.Module):
    def __init__(self, vocab_size, dim=100):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode='mean', padding_idx=0)
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, flat, offsets):
        return self.net(self.emb(flat, offsets))


class LexiconModel:
    name = 'Sözlük + edit distance (kural)'

    def __init__(self, path):
        self.lexicon = pickle.load(open(path, 'rb'))

    def predict(self, text):
        hits = lexicon_hits(normalize_full(text), self.lexicon)
        return int(bool(hits)), 1.0 if hits else 0.0


class TfidfModel:
    name = 'tf-idf + lojistik regresyon'

    def __init__(self, path):
        self.pipe = joblib.load(path)

    def predict(self, text):
        p = self.pipe.predict_proba([normalize_full(text)])[0]
        return int(p.argmax()), float(p.max())


class FFNModel:
    name = 'word2vec + feedforward ağ'

    def __init__(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.vocab = ckpt['vocab']
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        dim = ckpt['state']['emb.weight'].shape[1]
        self.model = FFN(len(self.vocab), dim)
        self.model.load_state_dict(ckpt['state'])
        self.model.eval()

    def predict(self, text):
        ids = [self.w2i.get(w, 1) for w in normalize_full(text).split()] or [1]
        with torch.no_grad():
            p = self.model(torch.tensor(ids), torch.tensor([0])).softmax(-1)[0]
        return int(p.argmax()), float(p.max())


class BertModel:
    def __init__(self, model_dir, quantize):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        if quantize:   # Alan D: Linear katmanlar INT8 (dinamik quantization), CPU'da çalışır
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.name = 'BERTurk INT8 (quantized)' if quantize else 'BERTurk FP32'

    def predict(self, text):
        enc = self.tok([normalize_light(text)], truncation=True, max_length=MAX_LEN, return_tensors='pt')
        with torch.no_grad():
            p = self.model(**enc).logits.softmax(-1)[0]
        return int(p.argmax()), float(p.max())


def load_all(models_dir, include_fp32=False):
    """Uygulamanın kullandığı modelleri yükler. Varsayılan model: BERTurk INT8."""
    models = {}
    bert_dir = os.path.join(models_dir, 'berturk_offensive')
    models['BERTurk INT8 (quantized)'] = BertModel(bert_dir, quantize=True)
    if include_fp32:
        models['BERTurk FP32'] = BertModel(bert_dir, quantize=False)
    models['word2vec + feedforward ağ'] = FFNModel(os.path.join(models_dir, 'w2v_ffn.pt'))
    models['tf-idf + lojistik regresyon'] = TfidfModel(os.path.join(models_dir, 'tfidf_lr.joblib'))
    models['Sözlük + edit distance (kural)'] = LexiconModel(os.path.join(models_dir, 'lexicon.pkl'))
    for m in models.values():          # ilk çağrıdaki yavaşlık ölçümleri bozmasın diye ısınma
        m.predict('merhaba')
    return models


def timed_predict(model, text):
    t0 = time.perf_counter()
    label, conf = model.predict(text)
    return label, conf, (time.perf_counter() - t0) * 1000
