"""Metin ön işleme (Alan B): regex normalizasyonu ve edit-distance sözlük eşleştirici.

Bu dosyadaki fonksiyonlar notebooks/train_all.ipynb'deki eğitimde kullanılanlarla birebir aynıdır.
"""
import re

TR_LOWER = str.maketrans({'I': 'ı', 'İ': 'i'})            # Türkçe'ye uygun küçük harf
TR_ASCII = str.maketrans('çğıöşü', 'cgiosu')               # "çok" ve "cok" aynı kelime olsun
LEET = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'}
LEET_RE = re.compile(r'(?<=[a-zçğıöşü])[013457@$]|[013457@$](?=[a-zçğıöşü])')


def normalize_light(text: str) -> str:
    """BERT için hafif temizlik: @USER/URL etiketleri ve uzun harf tekrarları."""
    text = re.sub(r'@USER', ' ', text)
    text = re.sub(r'\bURL\b', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)            # "çooook" -> "çook"
    return re.sub(r'\s+', ' ', text).strip()


def normalize_full(text: str) -> str:
    """Klasik modeller için tam normalizasyon: küçük harf, leetspeak, ASCII, tekrarlar."""
    text = normalize_light(text).translate(TR_LOWER).lower()
    for _ in range(2):                                        # "s1kt1r" gibi ardışık leet'ler
        text = LEET_RE.sub(lambda m: LEET[m.group()], text)
    text = text.translate(TR_ASCII)
    text = re.sub(r'(.)\1+', r'\1', text)                     # tüm tekrarları teke indir
    text = re.sub(r'[^a-z\s]', ' ', text)                     # harf olmayanları at
    return re.sub(r'\s+', ' ', text).strip()


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein), dinamik programlama ile."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lexicon_hits(clean_text: str, lexicon: list) -> list:
    """Sözlükteki bir kelimeye yeterince yakın olan kelimeleri döndürür (yazım hatalarını yakalar)."""
    hits = []
    for tok in clean_text.split():
        for w in lexicon:
            max_d = 0 if len(w) <= 4 else 1                   # kısa kelimelerde sadece birebir eşleşme
            if abs(len(tok) - len(w)) <= max_d and edit_distance(tok, w) <= max_d:
                hits.append(tok)
                break
    return hits
