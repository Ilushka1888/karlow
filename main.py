from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ВНИМАНИЕ:
# Этот dataclass в текущей реализации не используется.
# Я оставляю его как “заготовку” на случай, если ты захочешь хранить правила
# как объекты (суффикс, замена, условия). Можно смело удалить.
@dataclass(frozen=True)
class _Rule:
    suffix: str
    replacement: str
    min_m: Optional[int] = None
    eq_m: Optional[int] = None
    require_vowel_in_stem: bool = False


class PorterStemmer:
    """
    Реализация Porter Stemmer (1980): шаги 1a..5b.
    """

    # Множество “обычных” гласных в английском. Y обрабатываем отдельно.
    _vowels = set("aeiou")

    def stem(self, word: str) -> str:
        """
        Главная функция: принимает слово и возвращает его stem.
        Здесь только “обвязка”: нормализация + последовательный запуск шагов.
        """
        if not word:
            # Пустая строка -> пустая строка
            return word

        # Porter предполагает работу со строчными буквами.
        w = word.lower()

        # Учебная защита: если встретили не латинские буквы a-z,
        # можно вернуть как есть (или фильтровать иначе — зависит от требований).
        if any(not ("a" <= ch <= "z") for ch in w):
            return w

        # На очень коротких словах правила дают мусор, поэтому обычно их не трогают.
        if len(w) <= 2:
            return w

        # Дальше — строго по шагам алгоритма:
        w = self._step1a(w)
        w = self._step1b(w)
        w = self._step1c(w)
        w = self._step2(w)
        w = self._step3(w)
        w = self._step4(w)
        w = self._step5a(w)
        w = self._step5b(w)

        return w

    # ─────────────────────────────
    # БАЗОВЫЕ ПРЕДИКАТЫ (C/V, мера m, *v*, *d, *o)
    # ─────────────────────────────

    def _is_consonant(self, w: str, i: int) -> bool:
        """
        True если буква w[i] — согласная по определению Портера.
        Особый случай: буква 'y' зависит от предыдущей буквы.
        """
        ch = w[i]

        # Обычные гласные
        if ch in self._vowels:
            return False

        # Особое правило для 'y'
        if ch == "y":
            # В начале слова 'y' считается согласной
            if i == 0:
                return True
            # Иначе 'y' согласная тогда и только тогда,
            # когда предыдущая буква НЕ согласная (то есть гласная).
            # Это реализует “переключатель” V<->C.
            return not self._is_consonant(w, i - 1)

        # Все остальные буквы a-z, не являющиеся гласными, считаем согласными
        return True

    def _measure(self, w: str) -> int:
        """
        Считает меру m: количество блоков VC в [C](VC)^m[V].

        Алгоритм:
        1) пропускаем стартовые согласные C*
        2) затем многократно:
           - пропускаем блок гласных V*
           - если слово кончилось -> стоп (VC больше не образуется)
           - пропускаем блок согласных C* и увеличиваем счетчик (это был один VC)
        """
        n = 0
        i = 0
        L = len(w)

        # 1) пропустить начальные согласные
        while i < L and self._is_consonant(w, i):
            i += 1

        # 2) считать повторяющиеся “V* затем C*”
        while i < L:
            # пропустить гласные
            while i < L and not self._is_consonant(w, i):
                i += 1

            # если дошли до конца на гласных, VC не завершился -> выходим
            if i == L:
                break

            # пропустить согласные (это конец очередного VC-блока)
            while i < L and self._is_consonant(w, i):
                i += 1

            # мы нашли один блок VC
            n += 1

        return n

    def _contains_vowel(self, w: str) -> bool:
        """
        Проверка условия *v*: есть ли хотя бы одна гласная в строке.
        """
        return any(not self._is_consonant(w, i) for i in range(len(w)))

    def _ends_with_double_consonant(self, w: str) -> bool:
        """
        Проверка условия *d: заканчивается ли строка на двойную согласную.
        Например: falling -> 'll' (да), fizzed -> 'zz' (да), but: feed -> 'ee' (нет).
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
        Пример: hop (h-o-p) -> CVC и p не w/x/y -> True.
                box -> b-o-x: CVC, но x запрещен -> False.
        """
        if len(w) < 3:
            return False

        i = len(w) - 1

        # последняя буква должна быть согласной
        if not self._is_consonant(w, i):
            return False
        # предпоследняя — гласной
        if self._is_consonant(w, i - 1):
            return False
        # третья с конца — согласной
        if not self._is_consonant(w, i - 2):
            return False

        # последняя согласная не должна быть w, x, y
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
        Универсальная “машинка” для правил вида:
        если w заканчивается на suffix, отрезать suffix, получить stem,
        проверить условия (m>k / m=k / *v*), и заменить suffix на repl.

        Возвращает:
        - новую строку
        - флаг, применилось правило или нет
        """
        if not w.endswith(suffix):
            return w, False

        stem = w[: -len(suffix)]  # основа до суффикса

        # условие *v* на основе (в некоторых правилах)
        if require_vowel and not self._contains_vowel(stem):
            return w, False

        m = self._measure(stem)

        # min_m=k означает условие m > k (так в статье обычно пишут m>0, m>1)
        if min_m is not None and not (m > min_m):
            return w, False

        # eq_m означает m == k
        if eq_m is not None and not (m == eq_m):
            return w, False

        return stem + repl, True

    # ─────────────────────────────
    # ШАГИ 1a..5b
    # ─────────────────────────────

    def _step1a(self, w: str) -> str:
        """
        Шаг 1a (упрощение множественного числа):
        SSES -> SS
        IES  -> I
        SS   -> SS (не меняем)
        S    -> (удалить)
        """
        if w.endswith("sses"):
            return w[:-2]  # sses -> ss
        if w.endswith("ies"):
            return w[:-2]  # ies -> i
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        """
        Шаг 1b:
        (m>0) EED -> EE
        (*v*) ED  -> (удалить)
        (*v*) ING -> (удалить)
        После удаления ED/ING делаем post-processing (см. _step1b_post).
        """
        # правило EED -> EE при m>0 на основе
        w2, ok = self._replace_if(w, "eed", "ee", min_m=0)
        if ok:
            return w2

        # ED
        if w.endswith("ed"):
            stem = w[:-2]
            # *v*: в основе должна быть гласная
            if self._contains_vowel(stem):
                w = stem
                return self._step1b_post(w)
            return w

        # ING
        if w.endswith("ing"):
            stem = w[:-3]
            if self._contains_vowel(stem):
                w = stem
                return self._step1b_post(w)
            return w

        return w

    def _step1b_post(self, w: str) -> str:
        """
        Post-processing после удаления -ed или -ing:
        AT -> ATE
        BL -> BLE
        IZ -> IZE
        (*d and not (*L or *S or *Z)) -> remove last consonant
        (m=1 and *o) -> add E
        """
        # короткие “достройки” окончания
        for suf, repl in (("at", "ate"), ("bl", "ble"), ("iz", "ize")):
            if w.endswith(suf):
                return w[:-len(suf)] + repl

        # если слово кончается на двойную согласную,
        # убираем одну, кроме случаев l/s/z
        if self._ends_with_double_consonant(w) and w[-1] not in ("l", "s", "z"):
            return w[:-1]

        # если m=1 и шаблон cvc, добавляем 'e'
        if self._measure(w) == 1 and self._cvc(w):
            return w + "e"

        return w

    def _step1c(self, w: str) -> str:
        """
        Шаг 1c:
        (*v*) Y -> I
        То есть: если слово кончается на y, и в основе есть гласная, меняем y на i.
        happy -> happi, но sky -> sky (в sk нет гласной по определению).
        """
        if w.endswith("y"):
            stem = w[:-1]
            if self._contains_vowel(stem):
                return stem + "i"
        return w

    def _step2(self, w: str) -> str:
        """
        Шаг 2: “тяжелые” суффиксы, все с условием (m>0).
        Важно: выбирать самый длинный подходящий суффикс (поэтому сортировка).
        """
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

        # сортируем по длине суффикса убыванию,
        # чтобы “ization” проверилось раньше “ation”, и т.д.
        for suf, repl in sorted(rules, key=lambda x: len(x[0]), reverse=True):
            w2, ok = self._replace_if(w, suf, repl, min_m=0)  # m>0
            if ok:
                return w2

        return w

    def _step3(self, w: str) -> str:
        """
        Шаг 3: еще одна группа суффиксов, тоже с условием (m>0).
        """
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
        Шаг 4: удаление суффиксов при (m>1).
        Особое правило: -ion удаляем только если перед ним s или t.
        """
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]

        # special case для ion:
        # (m>1 and (*S or *T)) ION ->
        if w.endswith("ion") and len(w) > 3:
            stem = w[:-3]
            if stem and stem[-1] in ("s", "t") and self._measure(stem) > 1:
                return stem

        for suf in sorted(suffixes, key=len, reverse=True):
            w2, ok = self._replace_if(w, suf, "", min_m=1)  # m>1
            if ok:
                return w2

        return w

    def _step5a(self, w: str) -> str:
        """
        Шаг 5a: работа с конечной 'e'
        (m>1) E -> удалить
        (m=1 and not *o) E -> удалить
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
        Шаг 5b:
        (m>1 and *d and *L) -> remove one L
        Практически: если слово заканчивается на 'll' и m>1, делаем 'l'.
        """
        if self._measure(w) > 1 and w.endswith("ll"):
            return w[:-1]
        return w