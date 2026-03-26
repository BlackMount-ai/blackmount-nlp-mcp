"""Language and content detection — language ID, encoding, text statistics."""

import re
from collections import Counter

from .tokenize import sentence_tokenize, word_tokenize


# Common words per language for detection heuristic
_LANGUAGE_MARKERS: dict[str, set[str]] = {
    "english": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "her", "she",
        "or", "an", "will", "my", "would", "there", "their", "what",
        "about", "which", "when", "make", "can", "like", "time", "just",
        "him", "know", "take", "people", "into", "could", "your",
    },
    "spanish": {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
        "las", "por", "un", "para", "con", "no", "una", "su", "al",
        "es", "lo", "como", "mas", "pero", "sus", "le", "ya", "o",
        "fue", "este", "ha", "si", "porque", "esta", "son", "entre",
        "cuando", "muy", "sin", "sobre", "ser", "tambien", "me",
        "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
    },
    "french": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
        "que", "est", "dans", "qui", "ce", "au", "pas", "sur", "ne",
        "se", "par", "plus", "pour", "avec", "il", "son", "cette",
        "sont", "mais", "comme", "on", "tout", "nous", "sa", "fait",
        "ete", "aussi", "leur", "bien", "peut", "ces", "elle", "entre",
        "faire", "ont", "meme", "apres", "avoir", "donc", "tous",
    },
    "german": {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit",
        "sich", "des", "auf", "fur", "ist", "im", "dem", "nicht", "ein",
        "eine", "als", "auch", "es", "an", "werden", "aus", "er",
        "hat", "dass", "sie", "nach", "wird", "bei", "einer", "um",
        "am", "sind", "noch", "wie", "einem", "uber", "so", "zum",
        "war", "haben", "nur", "oder", "aber", "vor", "zur", "bis",
    },
    "italian": {
        "di", "che", "e", "la", "il", "un", "a", "per", "in", "una",
        "mi", "sono", "ho", "ma", "lo", "ha", "le", "si", "no", "al",
        "da", "non", "ci", "questo", "del", "come", "io", "con", "se",
        "cosa", "tutto", "bene", "molto", "anche", "qui", "ti", "hai",
        "fatto", "chi", "era", "dei", "nella", "essere", "lei", "lui",
        "sta", "gli", "piu", "quello", "va", "suo",
    },
    "portuguese": {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
        "com", "nao", "uma", "os", "no", "se", "na", "por", "mais",
        "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem",
        "seu", "sua", "ou", "ser", "quando", "muito", "ha", "nos",
        "ja", "esta", "eu", "tambem", "so", "pelo", "pela", "ate",
        "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
    },
    "dutch": {
        "de", "het", "een", "van", "en", "in", "is", "dat", "op",
        "te", "zijn", "voor", "met", "die", "niet", "aan", "er",
        "maar", "om", "ook", "als", "dan", "bij", "nog", "uit",
        "naar", "heeft", "ze", "worden", "was", "over", "werd",
        "hun", "meer", "kan", "deze", "wel", "geen", "door", "tot",
    },
    "swedish": {
        "och", "i", "att", "det", "som", "en", "pa", "ar", "av",
        "for", "med", "har", "den", "till", "inte", "var", "jag",
        "ett", "om", "han", "hade", "de", "sa", "vi", "kan", "men",
        "ska", "sin", "efter", "sig", "alla", "nu", "da", "min",
        "mot", "skulle", "fran", "vara", "upp", "nar",
    },
    "norwegian": {
        "og", "i", "det", "er", "en", "til", "som", "pa", "for",
        "med", "har", "av", "at", "den", "ikke", "var", "jeg",
        "men", "et", "han", "om", "sa", "vi", "kan", "de", "ble",
        "fra", "skal", "sin", "etter", "seg", "alle", "na", "da",
    },
    "danish": {
        "og", "i", "at", "er", "en", "den", "til", "det", "de",
        "som", "pa", "med", "for", "var", "har", "af", "et", "ikke",
        "der", "han", "jeg", "men", "blev", "fra", "kan", "vi",
        "vil", "sin", "sa", "efter", "sig", "alle", "nu", "da",
    },
    "finnish": {
        "ja", "on", "ei", "se", "oli", "han", "etta", "kun", "mutta",
        "niin", "kuin", "tai", "olen", "ovat", "vain", "nyt", "jo",
        "olla", "tama", "myos", "mina", "sitten", "yli", "hin",
        "tuo", "voi", "muu", "tehda", "hyvaa", "kanssa", "ennen",
    },
    "polish": {
        "i", "w", "nie", "na", "z", "do", "to", "sie", "co", "jest",
        "jak", "ale", "za", "od", "ze", "po", "tak", "ja", "go",
        "czy", "tego", "ten", "juz", "tym", "tylko", "jego", "by",
        "je", "o", "pan", "bardzo", "tu", "mi", "pan", "pani",
    },
    "russian": {
        "и", "в", "не", "на", "я", "что", "он", "с", "это", "а",
        "как", "но", "по", "она", "все", "они", "было", "к", "у",
        "его", "от", "за", "так", "же", "быть", "бы", "мне", "ее",
        "из", "вы", "то", "мы", "до", "да", "нет", "еще", "уже",
    },
    "turkish": {
        "bir", "bu", "da", "de", "ve", "ne", "ben", "mi", "o",
        "ile", "ama", "icin", "gibi", "var", "daha", "cok", "mu",
        "her", "sen", "benim", "onun", "kadar", "sonra", "bana",
        "ki", "ya", "onu", "hem", "siz", "simdi", "bunu",
    },
    "arabic": {
        "في", "من", "على", "إلى", "عن", "أن", "هذا", "كان",
        "هو", "هي", "ما", "لا", "التي", "الذي", "مع", "هذه",
        "بين", "ذلك", "كل", "ان", "بعد", "قد", "حتى", "إذا",
    },
    "japanese": {
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
        "れ", "さ", "ある", "いる", "も", "する", "から", "な",
        "こと", "として", "い", "や", "など", "なっ", "ない", "この",
    },
    "chinese": {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这",
    },
    "korean": {
        "이", "는", "의", "을", "를", "에", "와", "한", "하", "가",
        "다", "은", "로", "에서", "그", "있", "것", "으로", "도",
    },
}


def detect_language(text: str) -> list[dict[str, str | float]]:
    """Detect language using word frequency heuristics.

    Checks text against common word lists for 20 languages.

    Returns:
        List of dicts with 'language' and 'confidence' keys,
        sorted by confidence descending. Top result is best guess.
    """
    if not text or not text.strip():
        return [{"language": "unknown", "confidence": 0.0}]

    # Tokenize
    words = set(re.findall(r"\S+", text.lower()))
    if not words:
        return [{"language": "unknown", "confidence": 0.0}]

    scores: list[tuple[str, float]] = []

    for lang, markers in _LANGUAGE_MARKERS.items():
        overlap = words & markers
        if markers:
            score = len(overlap) / len(markers)
            if overlap:
                scores.append((lang, score))

    if not scores:
        return [{"language": "unknown", "confidence": 0.0}]

    scores.sort(key=lambda x: x[1], reverse=True)

    # Normalize confidence relative to top score
    top_score = scores[0][1]
    results = []
    for lang, score in scores[:5]:
        confidence = round(score / top_score if top_score > 0 else 0.0, 4)
        results.append({"language": lang, "confidence": confidence})

    return results


def detect_encoding_type(text: str) -> str:
    """Detect the primary character encoding type in text.

    Returns one of: 'ASCII', 'Latin', 'Cyrillic', 'CJK', 'Arabic',
    'Devanagari', 'Greek', 'Hebrew', 'Thai', 'Korean', 'Mixed', 'Unknown'.
    """
    if not text:
        return "Unknown"

    counts = Counter[str]()

    for char in text:
        cp = ord(char)
        if cp < 128:
            counts["ASCII"] += 1
        elif 0x00C0 <= cp <= 0x024F:
            counts["Latin"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["Cyrillic"] += 1
        elif (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            counts["CJK"] += 1
        elif 0x3040 <= cp <= 0x30FF:
            counts["CJK"] += 1  # Japanese kana
        elif 0x0600 <= cp <= 0x06FF:
            counts["Arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["Greek"] += 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["Hebrew"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["Thai"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["Korean"] += 1

    if not counts:
        return "Unknown"

    # Remove ASCII from consideration for the primary type
    non_ascii = {k: v for k, v in counts.items() if k != "ASCII"}

    if not non_ascii:
        return "ASCII"

    # If non-ASCII chars exist, check if there are multiple types
    if len(non_ascii) > 1:
        top = max(non_ascii.values())
        second = sorted(non_ascii.values(), reverse=True)[1]
        if second > top * 0.3:
            return "Mixed"

    return max(non_ascii, key=lambda k: non_ascii[k])


def is_english(text: str) -> float:
    """Estimate confidence that text is English (0-1 scale)."""
    results = detect_language(text)
    for r in results:
        if r["language"] == "english":
            return float(r["confidence"])
    return 0.0


def word_count(text: str) -> int:
    """Count words in text."""
    tokens = word_tokenize(text)
    return len([t for t in tokens if re.match(r"[A-Za-z]", t)])


def sentence_count(text: str) -> int:
    """Count sentences in text."""
    return len(sentence_tokenize(text))


def paragraph_count(text: str) -> int:
    """Count paragraphs in text (separated by blank lines)."""
    if not text or not text.strip():
        return 0
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return len([p for p in paragraphs if p.strip()])


def avg_word_length(text: str) -> float:
    """Average word length in characters."""
    tokens = word_tokenize(text)
    words = [t for t in tokens if re.match(r"[A-Za-z]", t)]
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 2)


def avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    sentences = sentence_tokenize(text)
    if not sentences:
        return 0.0
    total_words = sum(
        len([w for w in word_tokenize(s) if re.match(r"[A-Za-z]", w)])
        for s in sentences
    )
    return round(total_words / len(sentences), 2)
