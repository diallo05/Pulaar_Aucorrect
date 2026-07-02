"""
Module de correction orthographique basé sur la distance d'édition
et les probabilités d'un corpus.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Set

# Alphabet utilisé pour les remplacements et insertions
# Inclut les lettres latines standard + les caractères spécifiques au pulaar
pulaar_letters = "ɓɗƴŋɲñ"
letters = string.ascii_lowercase + pulaar_letters


# ---------------------------------------------------------------------------
# Construction du corpus de probabilités
# ---------------------------------------------------------------------------

def process_text(file_path: str) -> List[str]:
    """Lit un fichier texte et retourne la liste des mots en minuscules.

    Le texte est nettoyé : seuls les mots alphabétiques sont conservés,
    la ponctuation et les chiffres sont ignorés.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = re.findall(
        r"[a-zA-ZɓɗƴŋɲñƁƊƳŊƝ']+",
        text,
    )
    return [word.lower() for word in words]


def get_word_count(words: List[str]) -> Dict[str, int]:
    """Compte les occurrences de chaque mot du corpus."""
    return Counter(words)


def get_probabilities(word_count: Dict[str, int]) -> Dict[str, float]:
    """Calcule la probabilité d'apparition de chaque mot dans le corpus.

    probabilité(mot) = nombre d'occurrences du mot / nombre total de mots
    """
    total_count = sum(word_count.values())
    return {
        word: count / total_count
        for word, count in word_count.items()
    }


def build_probabilities_from_corpus(file_path: str) -> Dict[str, float]:
    """Pipeline complet : fichier texte -> dictionnaire de probabilités."""
    words = process_text(file_path)
    word_count = get_word_count(words)
    return get_probabilities(word_count)


# ---------------------------------------------------------------------------
# Génération des mots à une distance d'édition de 1
# ---------------------------------------------------------------------------

def delete_letter(word: str) -> List[str]:
    """Supprime une lettre du mot (à chaque position possible)."""
    return [
        word[:i] + word[i + 1:]
        for i in range(len(word))
    ]


def swap_letter(word: str) -> List[str]:
    """Échange deux lettres voisines."""
    return [
        word[:i] + word[i + 1] + word[i] + word[i + 2:]
        for i in range(len(word) - 1)
    ]


def replace_letter(word: str) -> List[str]:
    """Remplace une lettre par chaque autre lettre de l'alphabet."""
    return [
        word[:i] + letter + word[i + 1:]
        for i in range(len(word))
        for letter in letters
        if letter != word[i]
    ]


def insert_letter(word: str) -> List[str]:
    """Insère une lettre à chaque position possible du mot."""
    return [
        word[:i] + letter + word[i:]
        for i in range(len(word) + 1)
        for letter in letters
    ]


def edit_one_letter(word: str) -> Set[str]:
    """Génère tous les mots possibles avec une seule modification
    (suppression, échange, remplacement ou insertion)."""
    return set(
        delete_letter(word)
        + swap_letter(word)
        + replace_letter(word)
        + insert_letter(word)
    )


def edit_two_letters(word: str) -> Set[str]:
    """Génère tous les mots possibles avec deux modifications
    (utile quand une seule modification ne suffit pas à retrouver
    un mot correct, ex. deux fautes de frappe)."""
    return set(
        e2
        for e1 in edit_one_letter(word)
        for e2 in edit_one_letter(e1)
    )


def levenshtein_distance(word1: str, word2: str) -> int:
    """Calcule la distance de Levenshtein (nombre minimal d'insertions,
    suppressions ou substitutions) entre deux mots."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # suppression
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


# ---------------------------------------------------------------------------
# Correction du mot
# ---------------------------------------------------------------------------

def correct_word(word: str, probabilities: Dict[str, float]) -> str:
    """Retourne la correction la plus probable selon le corpus.

    Si le mot existe déjà dans le corpus, il est retourné tel quel.
    Sinon, on cherche parmi les mots à une modification de distance
    celui qui a la plus haute probabilité dans le corpus.
    Si aucun candidat n'est connu, le mot original est retourné.
    """
    word = word.lower()

    if word in probabilities:
        return word

    candidates = edit_one_letter(word)
    known_candidates = [
        candidate for candidate in candidates
        if candidate in probabilities
    ]

    if not known_candidates:
        return word

    return max(known_candidates, key=lambda candidate: probabilities[candidate])


def get_best_suggestions(
    word: str,
    probabilities: Dict[str, float],
    n: int = 3,
) -> List[str]:
    """Retourne les n mots du corpus les plus probables pour corriger `word`.

    Stratégie en cascade :
      1. Mots à distance d'édition 1 (une modification).
      2. Si aucun, mots à distance d'édition 2 (deux modifications).
      3. Si toujours aucun, recherche approximative par distance de
         Levenshtein sur tout le vocabulaire du corpus.

    Le mot saisi n'est jamais renvoyé seul comme "suggestion" : on
    retourne toujours une liste de vrais mots issus du corpus, triés
    par proximité (distance d'édition) puis par probabilité décroissante.
    """
    word = word.lower()

    # Étape 1 : distance 1
    candidates = edit_one_letter(word)
    known_candidates = [c for c in candidates if c in probabilities]

    # Étape 2 : distance 2 si rien trouvé
    if not known_candidates:
        candidates = edit_two_letters(word)
        known_candidates = [c for c in candidates if c in probabilities]

    if known_candidates:
        known_candidates.sort(key=lambda c: probabilities[c], reverse=True)
        return known_candidates[:n]

    # Étape 3 : recherche approximative sur tout le vocabulaire
    scored = [
        (vocab_word, levenshtein_distance(word, vocab_word), probabilities[vocab_word])
        for vocab_word in probabilities
    ]
    # Tri par distance croissante, puis probabilité décroissante
    scored.sort(key=lambda item: (item[1], -item[2]))

    return [vocab_word for vocab_word, _, _ in scored[:n]]


# ---------------------------------------------------------------------------
# Correction d'un texte complet
# ---------------------------------------------------------------------------

# Pattern qui identifie un "mot" (lettres latines + caractères pulaar)
WORD_PATTERN = re.compile(
    r"[a-zA-ZàâäéèêëïîôöùûüçÀÂÄÉÈÊËÏÎÔÖÙÛÜÇɓɗƴŋɲƁƊƳŊƝ']+"
)


def _match_case(original: str, corrected: str) -> str:
    """Applique la casse du mot original au mot corrigé.

    - TOUT MAJUSCULE -> reste tout majuscule
    - Première lettre majuscule -> reste première lettre majuscule
    - Sinon -> minuscules
    """
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def correct_text(
    text: str,
    probabilities: Dict[str, float],
):
    """Corrige un texte complet : chaque mot mal orthographié (absent du
    corpus) est remplacé par la meilleure suggestion trouvée par
    get_best_suggestions. La ponctuation, les espaces et la casse
    d'origine sont préservés.

    Retourne un tuple (texte_corrige, liste_des_corrections) où
    liste_des_corrections contient des tuples (mot_original, mot_corrige)
    pour chaque mot effectivement modifié.
    """
    corrections: List[tuple] = []

    def replace_match(match: "re.Match") -> str:
        original_word = match.group(0)
        lower_word = original_word.lower()

        # Mot déjà correct : on ne touche à rien
        if lower_word in probabilities:
            return original_word

        suggestions = get_best_suggestions(lower_word, probabilities, n=1)
        if not suggestions:
            return original_word

        best = suggestions[0]
        corrected_word = _match_case(original_word, best)

        if corrected_word != original_word:
            corrections.append((original_word, corrected_word))

        return corrected_word

    corrected_text = WORD_PATTERN.sub(replace_match, text)
    return corrected_text, corrections


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    file_path = "corpus_pulaar.txt"
    words = process_text(file_path)
    word_count = get_word_count(words)
    probabilities = get_probabilities(word_count)

    user_text = input("Entrez un texte : ")
    corrected_text, corrections = correct_text(user_text, probabilities)

    print("\nTexte corrigé :", corrected_text)

    
