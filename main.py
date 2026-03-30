from __future__ import annotations

from typing import Optional, Tuple


class PorterStemmer:
    # Множество “обычных” гласных в английском. Y отдельно.
    _vowels = set("aeiou")

    def stem(self, word: str) -> str:
        """
        принимает слово и возвращает его основу(sten).
        """
        if not word:
            return word

        w = word.lower()

        if any(not ("a" <= ch <= "z") for ch in w):
            return w

        if len(w) <= 2:
            return w

        w = self._step1a(w)
        w = self._step1b(w)
        w = self._step1c(w)
        w = self._step2(w)
        w = self._step3(w)
        w = self._step4(w)
        w = self._step5a(w)
        w = self._step5b(w)

        return w

    # (C/V, мера m, *v*, *d, *o)
    def _is_consonant(self, w: str, i: int) -> bool:
        """
        True если буква w[i].
        Особый случай: буква 'y' зависит от предыдущей буквы.
        """
        ch = w[i]

        if ch in self._vowels:
            return False

        # 'y'
        if ch == "y":
            # В начале слова 'y' считается согласной
            if i == 0:
                return True
            return not self._is_consonant(w, i - 1)

        return True

    def _measure(self, w: str) -> int:
        """
        Считает меру m: количество блоков VC в [C](VC)^m[V].
        """
        n = 0
        i = 0
        L = len(w)

        while i < L and self._is_consonant(w, i):
            i += 1

        while i < L:
            while i < L and not self._is_consonant(w, i):
                i += 1

            if i == L:
                break

            while i < L and self._is_consonant(w, i):
                i += 1

            n += 1

        return n

    def _contains_vowel(self, w: str) -> bool:
        """
        Проверка условия *v*: есть ли хотя бы одна гласная в строке.
        """
        return any(not self._is_consonant(w, i) for i in range(len(w)))

    def _ends_with_double_consonant(self, w: str) -> bool:
        """
        Проверка условия *d: tt, dd и т.д..
        """
        if len(w) < 2:
            return False
        if w[-1] != w[-2]:
            return False
        return self._is_consonant(w, len(w) - 1)

    def _cvc(self, w: str) -> bool:
        """
        Проверка условия *o:
        слово оканчивается на CVC, и последняя C не равна w/x/y.
        """
        if len(w) < 3:
            return False

        i = len(w) - 1

        if not self._is_consonant(w, i):
            return False
        if self._is_consonant(w, i - 1):
            return False
        if not self._is_consonant(w, i - 2):
            return False

        return w[i] not in ("w", "x", "y")

    def _replace_if(
        self,
        w: str,
        suffix: str,
        repl: str,
        *,
        min_m: Optional[int] = None,
        eq_m: Optional[int] = None,
        require_vowel: bool = False
    ) -> Tuple[str, bool]:
        """
        если w заканчивается на suffix, отрезать suffix, получить stem,
        проверить условия (m>k / m=k / *v*), и заменить suffix на repl.

        min_m=k <===> m > k

        Возвращает:
        - новую строку
        - флаг, true или false
        """
        if not w.endswith(suffix):
            return w, False

        stem = w[: -len(suffix)]  # основа до суффикса

        if require_vowel and not self._contains_vowel(stem):
            return w, False

        m = self._measure(stem)

        if min_m is not None and not (m > min_m):
            return w, False

        # eq_m <===> m == k
        if eq_m is not None and not (m == eq_m):
            return w, False

        return stem + repl, True

    # основные шаги
    def _step1a(self, w: str) -> str:
        """
        Шаг 1a (упрощение множественного числа):
        SSES -> SS
        IES -> I
        SS -> SS (не меняем)
        S -> (удалить)
        """
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        """
        (m>0) EED -> EE
        (*v*) ED -> (удалить)
        (*v*) ING -> (удалить)
        После удаления ED/ING делаем post-processing.
        """
        w2, ok = self._replace_if(w, "eed", "ee", min_m=0)
        if ok:
            return w2

        if w.endswith("ed"):
            stem = w[:-2]
            if self._contains_vowel(stem):
                w = stem
                return self._step1b_post(w)
            return w

        if w.endswith("ing"):
            stem = w[:-3]
            if self._contains_vowel(stem):
                w = stem
                return self._step1b_post(w)
            return w

        return w

    def _step1b_post(self, w: str) -> str:
        """
        после удаления -ed или -ing:
        AT -> ATE
        BL -> BLE
        IZ -> IZE
        (*d и не (*L or *S or *Z)) -> удаляем последнюю согласную
        (m=1 and *o) -> + E
        """
        for suf, repl in (("at", "ate"), ("bl", "ble"), ("iz", "ize")):
            if w.endswith(suf):
                return w[:-len(suf)] + repl

        if self._ends_with_double_consonant(w) and w[-1] not in ("l", "s", "z"):
            return w[:-1]

        if self._measure(w) == 1 and self._cvc(w):
            return w + "e"

        return w

    def _step1c(self, w: str) -> str:
        """
        (*v*) Y -> I
        """
        if w.endswith("y"):
            stem = w[:-1]
            if self._contains_vowel(stem):
                return stem + "i"
        return w

    def _step2(self, w: str) -> str:
        rules = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ]

        for suf, repl in sorted(rules, key=lambda x: len(x[0]), reverse=True):
            w2, ok = self._replace_if(w, suf, repl, min_m=0)
            if ok:
                return w2

        return w

    def _step3(self, w: str) -> str:
        rules = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]

        for suf, repl in sorted(rules, key=lambda x: len(x[0]), reverse=True):
            w2, ok = self._replace_if(w, suf, repl, min_m=0)
            if ok:
                return w2

        return w

    def _step4(self, w: str) -> str:
        """
        -ion удаляем если перед ним s или t.
        """
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]

        if w.endswith("ion") and len(w) > 3:
            stem = w[:-3]
            if stem and stem[-1] in ("s", "t") and self._measure(stem) > 1:
                return stem

        for suf in sorted(suffixes, key=len, reverse=True):
            w2, ok = self._replace_if(w, suf, "", min_m=1)
            if ok:
                return w2

        return w

    def _step5a(self, w: str) -> str:
        """
        (m>1) E -> удалить
        (m=1 + не *o) E -> удалить
        """
        if w.endswith("e"):
            stem = w[:-1]
            m = self._measure(stem)

            if m > 1:
                return stem
            if m == 1 and not self._cvc(stem):
                return stem

        return w

    def _step5b(self, w: str) -> str:
        """
        (m>1 + *d + *L) -> -l
        """
        if self._measure(w) > 1 and w.endswith("ll"):
            return w[:-1]
        return w