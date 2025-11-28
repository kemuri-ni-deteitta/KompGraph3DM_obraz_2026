"""
Набор R‑функций, с помощью которых мы собираем сложные фигуры из простых F(x,y,z).

Это аккуратная портировка исходного C++‑кода:
  * линейные R‑функции при alpha = 1 (пересечение и объединение);
  * обобщённые R‑функции для произвольного alpha;
  * инверсия (логическое НЕ для функции F);
  * квадратичные варианты при alpha = 0.

Идеология:
  * Каждая фигура описывается функцией F(x,y,z).
  * Знак F интерпретируем как:
        F > 0  — точка принадлежит фигуре (внутри);
        F = 0  — точка лежит на границе;
        F < 0  — точка снаружи.
  * R‑функции позволяют выразить логические операции над такими «неявными множествами»
    (AND, OR, NOT) так, чтобы результат снова был гладкой функцией, а не просто маской.

Практически это значит, что мы можем написать:
  F_total = r_intersection(F_body, r_inversion(F_hole))
и тем самым «вычесть» отверстие из тела фигуры, как это сделано для свечи в main.py.
"""

from __future__ import annotations

import math


# -----------------------------
# 1. Линейные R-функции (alpha = 1)
# -----------------------------
def r_intersection_alpha1(w1: float, w2: float) -> float:
    """
    Линейное R-пересечение (alpha = 1).

    Аналог C++:
        return (w1 + w2 - abs(w1 - w2)) * 0.5;
    """
    return (w1 + w2 - abs(w1 - w2)) * 0.5


def r_union_alpha1(w1: float, w2: float) -> float:
    """
    Линейное R-объединение (alpha = 1).

    Аналог C++:
        return (w1 + w2 + abs(w1 - w2)) * 0.5;
    """
    return (w1 + w2 + abs(w1 - w2)) * 0.5


# ----------------------------------------------------
# 2. Общие R-функции для произвольного параметра alpha
# ----------------------------------------------------
def _normalize_alpha(alpha: float) -> float:
    """
    Нормализация параметра alpha в том же духе, что и в C++:
    если alpha <= -1 или alpha > 1, используем alpha = 1.
    """
    if alpha <= -1.0 or alpha > 1.0:
        return 1.0
    return alpha


def r_intersection(w1: float, w2: float, alpha: float = 1.0) -> float:
    """
    Универсальное R-пересечение.

    Формула из методички:
        (w1 + w2 - sqrt(w1^2 + w2^2 - 2 alpha w1 w2)) / (alpha + 1)
    """
    alpha = _normalize_alpha(alpha)

    denominator = alpha + 1.0
    if abs(denominator) < 1e-10:
        denominator = 1.0

    sqrt_term = math.sqrt(w1 * w1 + w2 * w2 - 2.0 * alpha * w1 * w2)
    return (w1 + w2 - sqrt_term) / denominator


def r_union(w1: float, w2: float, alpha: float = 1.0) -> float:
    """
    Универсальное R-объединение.

    Формула:
        (w1 + w2 + sqrt(w1^2 + w2^2 - 2 alpha w1 w2)) / (alpha + 1)
    """
    alpha = _normalize_alpha(alpha)

    denominator = alpha + 1.0
    if abs(denominator) < 1e-10:
        denominator = 1.0

    sqrt_term = math.sqrt(w1 * w1 + w2 * w2 - 2.0 * alpha * w1 * w2)
    return (w1 + w2 + sqrt_term) / denominator


# -----------------------
# 3. Инверсия (NOT)
# -----------------------
def r_inversion(w: float) -> float:
    """
    Инверсия (логическое НЕ для F(x,y,z)).

    Аналог C++:
        return -w;
    """
    return -w


# ------------------------------------------------------------
# 4. Квадратичные R-функции (альтернативы при alpha = 0)
# ------------------------------------------------------------
def r_intersection_alpha0(w1: float, w2: float) -> float:
    """
    R-пересечение при alpha = 0.

    Формула:
        w1 + w2 - sqrt(w1^2 + w2^2)
    """
    return w1 + w2 - math.sqrt(w1 * w1 + w2 * w2)


def r_union_alpha0(w1: float, w2: float) -> float:
    """
    R-объединение при alpha = 0:

        w1 + w2 + sqrt(w1^2 + w2^2)
    """
    return w1 + w2 + math.sqrt(w1 * w1 + w2 * w2)



