# preprocess.py

import re

class ArabicPreprocessor:
    """
    Nettoyage et normalisation du texte arabe :
      - Suppression des diacritiques
      - Normalisation des alifs (إ أ آ ا → ا)
      - Normalisation de la yāʾ et wāw (ى → ي, ؤ ئ → ي)
      - Normalisation de tāʾ marbūṭa (ة → ه)
      - Suppression de la ponctuation et caractères non-arabes
      - Réduction des espaces multiples
    """

    def __init__(self):
        # Pattern pour diacritiques arabes
        self.diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
        # Tout ce qui n'est pas lettre arabe ou espace
        self.non_arabic = re.compile(r'[^\u0600-\u06FF\s]')

    def clean(self, text: str) -> str:
        # 1) Supprimer les diacritiques
        text = re.sub(self.diacritics, '', text)
        # 2) Normaliser alifs
        text = re.sub(r'[إأآا]', 'ا', text)
        # 3) Normaliser yāʾ et wāw
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'[ؤئ]', 'ي', text)
        # 4) Normaliser tāʾ marbūṭa
        text = re.sub(r'ة', 'ه', text)
        # 5) Supprimer tout caractère non-arabe
        text = re.sub(self.non_arabic, ' ', text)
        # 6) Élaguer et réduire les espaces multiples
        return re.sub(r'\s+', ' ', text).strip()

def extract_fragment(path: str, offset: int, length: int) -> str:
    """
    Lit le fichier UTF-8 à l'emplacement `path`, extrait la sous-chaîne
    [offset:offset+length], puis la nettoie via ArabicPreprocessor.

    Args:
        path (str): chemin du fichier .txt
        offset (int): position de début du fragment
        length (int): longueur du fragment en caractères

    Returns:
        str: fragment nettoyé, ou chaîne vide si hors bornes.
    """
    try:
        with open(path, encoding='utf-8') as f:
            raw = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    # Si l'offset est en dehors du texte, retourner une chaîne vide
    if offset < 0 or offset >= len(raw):
        return ""

    fragment = raw[offset: offset + length]
    return ArabicPreprocessor().clean(fragment)
