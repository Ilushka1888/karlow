from dataclasses import dataclass
from collections import deque
import os

os.system("")


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[31m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"


@dataclass(frozen=True)
class Situation:
    lhs: str
    rhs: tuple[str, ...]
    dot: int

    def is_complete(self) -> bool:
        return self.dot == len(self.rhs)

    def next_symbol(self):
        if self.is_complete():
            return None
        return self.rhs[self.dot]

    def advance(self):
        return Situation(self.lhs, self.rhs, self.dot + 1)


def default_grammar():
    grammar = {
        "S": [
            ("S", "T"),
            ("T",)
        ],
        "T": [
            ("a", "T"),
            ("b",),
        ]
    }

    start_symbol = "S"

    return grammar, start_symbol


def format_situation(situation: Situation) -> str:
    rhs = list(situation.rhs)
    rhs.insert(situation.dot, "·")

    if rhs:
        rhs_text = " ".join(rhs)
    else:
        rhs_text = "·"

    return f"{situation.lhs} -> {rhs_text}"


def parse_rhs_variant(text: str) -> tuple[str, ...]:
    text = text.strip()

    if text in {"", "ε", "eps", "epsilon"}:
        return tuple()

    if " " in text:
        return tuple(text.split())

    return tuple(text)


def parse_rule_line(line: str):
    if "->" not in line:
        raise ValueError("В правиле должен быть символ ->")

    lhs, rhs_text = line.split("->", 1)
    lhs = lhs.strip()

    if not lhs:
        raise ValueError("Пустая левая часть правила")

    variants = rhs_text.split("|")
    productions = [parse_rhs_variant(variant) for variant in variants]

    return lhs, productions


def input_custom_grammar():
    print()
    print("Ввод новой грамматики")
    print("Формат:")
    print("  S -> ST | T")
    print("  T -> aT | b")
    print()
    print("Или через пробелы:")
    print("  S -> S T | T")
    print("  T -> a T | b")
    print()
    print("Для пустой строки используйте ε:")
    print("  A -> ε")
    print()
    print("Вводите правила по одному.")
    print("Пустая строка — закончить ввод.")
    print()

    grammar = {}

    while True:
        line = input("Правило: ").strip()

        if line == "":
            break

        try:
            lhs, productions = parse_rule_line(line)

            if lhs not in grammar:
                grammar[lhs] = []

            grammar[lhs].extend(productions)

        except ValueError as error:
            print(f"Ошибка: {error}")
            print("Попробуйте ещё раз.")
            print()

    if not grammar:
        print("Грамматика не была введена. Используется грамматика по умолчанию.")
        return default_grammar()

    start_symbol = input("Стартовый символ, по умолчанию S: ").strip()

    if not start_symbol:
        start_symbol = "S"
        print("Стартовый символ не указан. Используется: S")

    if start_symbol not in grammar:
        print(f"Для стартового символа {start_symbol} нет правил.")
        print("Используется грамматика по умолчанию.")
        return default_grammar()

    return grammar, start_symbol


def print_grammar(grammar, start_symbol):
    print()
    print("Текущая грамматика:")

    for lhs, productions in grammar.items():
        variants = []

        for rhs in productions:
            if rhs:
                variants.append(" ".join(rhs))
            else:
                variants.append("ε")

        print(f"  {lhs} -> {' | '.join(variants)}")

    print(f"Стартовый символ: {start_symbol}")
    print()


def parse_word(line: str) -> list[str]:
    line = line.strip()

    if line in {"ε", "eps", "epsilon"}:
        return []

    if " " in line:
        return line.split()

    return list(line)


def earley_algorithm(grammar, start_symbol, word):
    n = len(word)

    M = [[set() for _ in range(n + 1)] for _ in range(n + 1)]

    queue = deque()
    nonterminals = set(grammar.keys())

    def is_nonterminal(symbol: str) -> bool:
        return symbol in nonterminals

    def add(i: int, j: int, situation: Situation):
        if situation not in M[i][j]:
            M[i][j].add(situation)
            queue.append((i, j, situation))

    # A. Инициализация
    for rhs in grammar[start_symbol]:
        add(0, 0, Situation(start_symbol, rhs, 0))

    while queue:
        i, j, situation = queue.popleft()

        if not situation.is_complete():
            symbol = situation.next_symbol()

            if is_nonterminal(symbol):
                for production in grammar[symbol]:
                    add(j, j, Situation(symbol, production, 0))

                for k in range(j, n + 1):
                    for completed in list(M[j][k]):
                        if completed.lhs == symbol and completed.is_complete():
                            add(i, k, situation.advance())

            else:
                if j < n and word[j] == symbol:
                    add(i, j + 1, situation.advance())

        else:
            completed_nonterminal = situation.lhs

            for h in range(0, i + 1):
                for previous in list(M[h][i]):
                    if previous.next_symbol() == completed_nonterminal:
                        add(h, j, previous.advance())

    accepted = False

    for rhs in grammar[start_symbol]:
        final_situation = Situation(start_symbol, rhs, len(rhs))

        if final_situation in M[0][n]:
            accepted = True
            break

    return accepted, M


def color_situation(situation, i, j, n, start_symbol):
    text = format_situation(situation)

    if i == 0 and j == n and situation.lhs == start_symbol and situation.is_complete():
        return Color.BOLD + Color.GREEN + text + Color.RESET

    if situation.is_complete():
        return Color.CYAN + text + Color.RESET

    return Color.YELLOW + text + Color.RESET


def sorted_cell(cell):
    return sorted(cell, key=lambda s: (s.lhs, s.rhs, s.dot))


def print_table(M, start_symbol):
    n = len(M) - 1

    plain_cells = []
    color_cells = []

    for i in range(n + 1):
        plain_row = []
        color_row = []

        for j in range(n + 1):
            situations = sorted_cell(M[i][j])

            if not situations:
                plain_lines = [""]
                color_lines = [""]
            else:
                plain_lines = [format_situation(s) for s in situations]
                color_lines = [
                    color_situation(s, i, j, n, start_symbol)
                    for s in situations
                ]

            plain_row.append(plain_lines)
            color_row.append(color_lines)

        plain_cells.append(plain_row)
        color_cells.append(color_row)

    col_widths = []

    for j in range(n + 1):
        width = len(f"j={j}")

        for i in range(n + 1):
            for line in plain_cells[i][j]:
                width = max(width, len(line))

        col_widths.append(width + 2)

    row_header_width = 6

    def horizontal_line():
        result = "+"
        result += "-" * row_header_width
        result += "+"

        for width in col_widths:
            result += "-" * width
            result += "+"

        return result

    print()
    print("ТАБЛИЦА M[i][j]")
    print(horizontal_line())

    header = "|"
    header += f"{'':^{row_header_width}}"
    header += "|"

    for j in range(n + 1):
        header += f"{f'j={j}':^{col_widths[j]}}"
        header += "|"

    print(header)
    print(horizontal_line())

    for i in range(n + 1):
        max_lines = max(len(plain_cells[i][j]) for j in range(n + 1))

        for line_index in range(max_lines):
            row = "|"

            if line_index == 0:
                row += f"{f'i={i}':^{row_header_width}}"
            else:
                row += " " * row_header_width

            row += "|"

            for j in range(n + 1):
                plain_lines = plain_cells[i][j]
                color_lines = color_cells[i][j]

                if line_index < len(plain_lines):
                    plain_text = plain_lines[line_index]
                    color_text = color_lines[line_index]
                else:
                    plain_text = ""
                    color_text = ""

                padding = " " * (col_widths[j] - len(plain_text))
                row += color_text + padding + "|"

            print(row)

        print(horizontal_line())

    print()


def print_result(word, accepted):
    word_text = "".join(word) if word else "ε"

    if accepted:
        print(Color.BOLD + Color.GREEN + f"{word_text} принимается" + Color.RESET)
    else:
        print(Color.BOLD + Color.RED + f"{word_text} не принимается" + Color.RESET)

    print()


def check_word(grammar, start_symbol):
    line = input("Введите слово: \n").strip()

    if not line:
        print("Слово не введено.\n")
        return

    word = parse_word(line)

    accepted, M = earley_algorithm(grammar, start_symbol, word)

    print(f"Слово: {''.join(word) if word else 'ε'}")

    print_table(M, start_symbol)
    print_result(word, accepted)


def print_menu():
    print("1. Проверить слово")
    print("2. Показать грамматику")
    print("3. Изменить грамматику")
    print("4. Вернуть грамматику по умолчанию")
    print("0. Выход\n")


def main():
    grammar, start_symbol = default_grammar()

    while True:
        print_menu()

        choice = input("Выберите пункт: \n").strip()

        if choice == "1":
            check_word(grammar, start_symbol)

        elif choice == "2":
            print_grammar(grammar, start_symbol)

        elif choice == "3":
            grammar, start_symbol = input_custom_grammar()
            print_grammar(grammar, start_symbol)

        elif choice == "4":
            grammar, start_symbol = default_grammar()
            print("Грамматика по умолчанию восстановлена.")
            print_grammar(grammar, start_symbol)


if __name__ == "__main__":
    main()
