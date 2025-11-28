"""
Модуль, который превращает математическую функцию F(x,y,z) в набор M‑образов (BMP‑файлов).

На вход мы даём:
  * неявную функцию F(x,y,z), задающую фигуру;
  * границы области по X/Y/Z и размеры сетки;
  * путь для сохранения результатов.

Модуль делает несколько вещей:
  1. Строит регулярную 3D‑сетку и вычисляет значения F во всех её узлах;
  2. Для каждой маленькой ячейки проверяет, пересекает ли её поверхность F=0;
  3. Если пересекает — считает нормаль N = (n1..n5) через определители и нормирует её;
  4. Кодирует компоненты нормали в пять M‑образов (по одному BMP на компоненту);
  5. В первую строку каждого M‑образа записывает «шапку» с параметрами сетки и диапазоном F.

Таким образом, на выходе мы получаем пять синхронных изображений Mi, которых достаточно,
чтобы визуализатор мог восстановить поверхность и ориентировку нормалей.
"""

import os
from PIL import Image
from typing import Callable

from mimage_common import normal_to_mimage_value, encode_color_to_rgb
from mimage_math import compute_n12345, normalize_n12345
from mimage_header import write_header_to_images


# ============================================================================
# ВЫЧИСЛЕНИЕ ПОЛЯ F(x,y,z) ПО ВСЕЙ 3D-СЕТКЕ
# ============================================================================

def compute_function_field(F: Callable, xmin: float, xmax: float,
                          ymin: float, ymax: float, zmin: float, zmax: float,
                          nx: int, ny: int, nz: int) -> tuple:
    """
    Вычисляет значения функции F(x,y,z) во всех узлах 3D-сетки.
    
    Args:
        F: Неявная функция фигуры F(x, y, z)
        xmin, xmax, ymin, ymax, zmin, zmax: Границы области моделирования
        nx, ny, nz: Размеры сетки
    
    Returns:
        Кортеж (X, Y, Z, u, umin, umax) где:
        - X, Y, Z: Координатные сетки
        - u: Трёхмерный массив значений функции F
        - umin, umax: Минимальное и максимальное значение функции F
    """
    print("Шаг 1: вычисляем u = F(x,y,z) во всех узлах и находим диапазон [UMIN; UMAX]...")
    
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)
    
    # Строим одномерные координатные массивы X, Y, Z,
    # чтобы к ним можно было обращаться по индексу везде одинаково.
    X = [xmin + j * dx for j in range(nx)]
    Y = [ymin + i * dy for i in range(ny)]
    Z = [zmin + k * dz for k in range(nz)]
    
    # Вычисляем значения функции F в каждом узле трёхмерной сетки.
    u = [[[0.0 for _ in range(nx)] for _ in range(ny)] for _ in range(nz)]
    
    UMIN = float("inf")
    UMAX = float("-inf")
    
    for k in range(nz):
        for i in range(ny):
            for j in range(nx):
                val = F(X[j], Y[i], Z[k])
                u[k][i][j] = val
                
                if val < UMIN:
                    UMIN = val
                if val > UMAX:
                    UMAX = val
    
    print(f"Диапазон F: UMIN = {UMIN:.4f}, UMAX = {UMAX:.4f}")
    
    return X, Y, Z, u, UMIN, UMAX


# ============================================================================
# ГЕНЕРАЦИЯ M-ОБРАЗОВ ДЛЯ ОДНОГО СЛОЯ
# ============================================================================

def generate_layer_mimages(k: int, X: list, Y: list, Z: list, u: list, 
                           nx: int, ny: int, nz: int) -> tuple:
    """
    Генерирует M-образы для одного слоя k.
    
    Args:
        k: Индекс слоя (0..nz-1)
        X, Y, Z: Координатные сетки
        u: Трёхмерный массив значений функции F
        nx, ny, nz: Размеры сетки
    
    Returns:
        Кортеж (imgs, pix) где:
        - imgs: Список из 5 объектов Image
        - pix: Список из 5 объектов pix (результат img.load())
    """
    # Создаём 5 изображений для текущего слоя.
    # По умолчанию заполняем их «нулевой» нормалью (n = 0),
    # чтобы даже там, где поверхность не проходит через ячейку,
    # в M‑образе лежали корректные (пусть и неинформативные) значения.
    imgs = [Image.new("RGB", (nx, ny), (0, 0, 0)) for _ in range(5)]
    pix = [img.load() for img in imgs]

    # Значение палитры, соответствующее нормали n = 0 по всем компонентам.
    zero_M = normal_to_mimage_value(0.0)
    zero_r, zero_g, zero_b = encode_color_to_rgb(zero_M)

    # Заполняем всю область (кроме заголовка y=0, который будет записан позже)
    for i in range(1, ny):
        for j in range(nx):
            for m in range(5):
                pix[m][j, i] = (zero_r, zero_g, zero_b)

    # Обрабатываем каждую ячейку объёма (параллелепипед между четырьмя узлами в этом слое
    # и их «соседом» по оси Z). В этой ячейке мы решаем: есть ли пересечение с F=0 или нет.
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Берём 4 соседних узла (точки, как в методичке).
            # P1 = (j, i, k) - текущая точка
            # P2 = (j+1, i, k) - сосед по X
            # P3 = (j, i+1, k) - сосед по Y
            # P4 = (j, i, k+1) или (j, i, k) если последний слой
            x1, y1, z1, u1 = X[j],   Y[i],   Z[k],   u[k][i][j]
            x2, y2, z2, u2 = X[j+1], Y[i],   Z[k],   u[k][i][j+1]
            x3, y3, z3, u3 = X[j],   Y[i+1], Z[k],   u[k][i+1][j]

            if k + 1 < nz:
                x4, y4, z4, u4 = X[j], Y[i], Z[k+1], u[k+1][i][j]
            else:
                x4, y4, z4, u4 = X[j], Y[i], Z[k],   u[k][i][j]

            # Проверяем, пересекает ли поверхность F=0 данную ячейку.
            # Если все четыре значения одного знака, значит и вся ячейка
            # лежит полностью внутри или полностью снаружи тела – нормаль нам там не нужна.
            all_pos = (u1 > 0.0 and u2 > 0.0 and u3 > 0.0 and u4 > 0.0)
            all_neg = (u1 < 0.0 and u2 < 0.0 and u3 < 0.0 and u4 < 0.0)
            if all_pos or all_neg:
                # Оставляем «нулевую» нормаль, записанную выше.
                continue

            # Вычисляем нормали n1..n5 через определители (см. mimage_math.compute_n12345)
            n1, n2, n3, n4, n5 = compute_n12345(
                x1, y1, z1, u1,
                x2, y2, z2, u2,
                x3, y3, z3, u3,
                x4, y4, z4, u4
            )

            # Нормализуем нормали, чтобы длина вектора N была равна 1.
            normalized = normalize_n12345(n1, n2, n3, n4, n5)
            if normalized is None:
                continue  # Пропускаем, если длина равна нулю
            
            n1, n2, n3, n4, n5 = normalized
            N = [n1, n2, n3, n4, n5]

            # Формируем пиксели для 5 M‑образов: каждая компонента нормали в своём BMP.
            for m in range(5):
                M_val = normal_to_mimage_value(N[m])
                r, g, b = encode_color_to_rgb(M_val)
                pix[m][j, i] = (r, g, b)

    return imgs, pix


def save_layer_mimages(imgs: list, pix: list, k: int, nz: int,
                       xmin: float, xmax: float, ymin: float, ymax: float,
                       zmin: float, zmax: float, umin: float, umax: float,
                       output_dir: str = "images"):
    """
    Записывает заголовок и сохраняет M-образы слоя в файлы.
    
    Args:
        imgs: Список из 5 объектов Image
        pix: Список из 5 объектов pix
        k: Индекс слоя (0..nz-1)
        nz: Количество слоёв
        xmin, xmax, ymin, ymax, zmin, zmax: Границы области моделирования
        umin, umax: Минимальное и максимальное значение функции F
        output_dir: Директория для сохранения файлов
    """
    # Записываем заголовок в строку y=0 всех 5 M-образов
    write_header_to_images(pix, nz, xmin, xmax, ymin, ymax, zmin, zmax, umin, umax)

    # Сохраняем 5 M-образов слоя
    for m in range(5):
        filename = f"{output_dir}/{k+1} (M{m+1}).bmp"
        imgs[m].save(filename)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ M-ОБРАЗОВ
# ============================================================================

def generate_mimages(F: Callable,
                     xmin: float = -2.0, xmax: float = 2.0,
                     ymin: float = -2.0, ymax: float = 2.0,
                     zmin: float = -2.0, zmax: float = 2.0,
                     nx: int = 128, ny: int = 128, nz: int = 256,
                     output_dir: str = "images"):
    """
    Генерирует M-образы для заданной неявной функции F(x,y,z).
    
    Args:
        F: Неявная функция фигуры F(x, y, z)
            F > 0  – внутри тела
            F = 0  – поверхность
            F < 0  – снаружи
        xmin, xmax, ymin, ymax, zmin, zmax: Границы области моделирования
        nx, ny, nz: Размеры сетки (число узлов по осям)
        output_dir: Директория для сохранения M-образов
    
    Returns:
        None (сохраняет файлы в output_dir)
    """
    # Создаём директорию для выходных файлов
    os.makedirs(output_dir, exist_ok=True)
    
    # Вычисляем поле функции F(x,y,z)
    X, Y, Z, u, umin, umax = compute_function_field(
        F, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz
    )
    
    # Генерируем M-образы для каждого слоя
    print("Шаг 2: генерация M-образов...")
    
    for k in range(nz):
        # Генерируем M-образы для слоя k
        imgs, pix = generate_layer_mimages(k, X, Y, Z, u, nx, ny, nz)
        
        # Сохраняем M-образы слоя
        save_layer_mimages(imgs, pix, k, nz, xmin, xmax, ymin, ymax, 
                          zmin, zmax, umin, umax, output_dir)
    
    print(f"Готово! M-образы сохранены в папку {output_dir}/")

