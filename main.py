"""
Главная точка входа для генерации M-образов.

В этом файле собраны:
  * простые неявные функции F(x,y,z) для базовых примитивов (сфера, эллипсоид, цилиндр);
  * вспомогательные геометрические примитивы, которые используются совместно с R‑функциями;
  * составная фигура «свеча с подставкой» – хороший пример того, как комбинировать
    несколько примитивов через R‑функции пересечения / объединения / вычитания;
  * функция `start(...)`, которая запускает полный цикл генерации M‑образов для
    выбранной функции или набора функций.

Идея такая: мы описываем фигуру одной скалярной функцией F(x,y,z).
Дальше всё остальное (построение сетки, вычисление нормалей, запись M‑образов)
делает модуль `mimage_generator`.
"""

from typing import Callable, Union, List
from mimage_generator import generate_mimages
from r_functions import (
    r_intersection_alpha0,
    r_union_alpha0,
    r_inversion,
)


# ============================================================================
# ПРОСТЫЕ ПРИМИТИВЫ F(x,y,z): СФЕРА, ЭЛЛИПСОИД, ПАРАБОЛОИД, ЦИЛИНДР
# ============================================================================

def sphere(x: float, y: float, z: float) -> float:
    """
    Сфера радиуса 1 с центром в начале координат.

    Классическая формула:
        F(x,y,z) = 1 - x^2 - y^2 - z^2

    По нашему соглашению:
        F > 0  — внутри сферы
        F = 0  — поверхность
        F < 0  — снаружи
    """
    return 1.0 - x*x - y*y - z*z


def ellipsoid(x: float, y: float, z: float) -> float:
    """
    Эллипсоид с полуосями a=1.5, b=1.0, c=0.8.
    Оси ориентированы вдоль X, Y, Z, центр — в начале координат.
    """
    return 1.0 - (x/1.5)**2 - (y/1.0)**2 - (z/0.8)**2


def paraboloid(x: float, y: float, z: float) -> float:
    """
    Параболоид вращения, «открытый вверх».

    Формула:
        F(x,y,z) = z - x^2 - y^2

    При F = 0 получаем поверхность z = x^2 + y^2.
    """
    return z - x*x - y*y


def cylinder(x: float, y: float, z: float) -> float:
    """
    Неограниченный по высоте цилиндр радиуса 1 вокруг оси Oz.

    Формула:
        F(x,y,z) = 1 - x^2 - y^2
    """
    return 1.0 - x*x - y*y


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ПРИМИТИВЫ ДЛЯ СЛОЖНЫХ ФИГУР (ИСПОЛЬЗУЮТСЯ С R-ФУНКЦИЯМИ)
# ============================================================================

def _cyl_R(x: float, y: float, R: float) -> float:
    """
    Круглый цилиндр радиуса R вокруг оси Oz.

    Удобная «заготовка» для дальнейших пересечений с плоскостями по z.
    """
    return R * R - x * x - y * y


def _plane_z_ge(z: float, z0: float) -> float:
    """Полупространство z >= z0: F > 0 выше (или на) плоскости z = z0."""
    return z - z0


def _plane_z_le(z: float, z1: float) -> float:
    """Полупространство z <= z1: F > 0 ниже (или на) плоскости z = z1."""
    return z1 - z


def _sphere_R_center_z(x: float, y: float, z: float, R: float, cz: float) -> float:
    """Сфера радиуса R с центром (0,0,cz)."""
    return R * R - x * x - y * y - (z - cz) * (z - cz)


# ============================================================================
# КОМПОЗИТНАЯ ФИГУРА: СВЕЧА С ПОДСТАВКОЙ
# ============================================================================

# Геометрические параметры свечи.
# Отдельные константы помогают быстро менять форму без переписывания формул.
# Подставка (нижняя «тарелочка»)
CANDLE_BASE_R = 1.2      # радиус подставки по X/Y – определяет, насколько она шире свечи
CANDLE_BASE_Z0 = 0.0     # высота нижней плоскости подставки (условный «стол»)
CANDLE_BASE_Z1 = 0.3     # высота верхней плоскости подставки

# Тело свечи – основной цилиндр из воска
CANDLE_BODY_R = 0.5      # радиус цилиндра свечи
CANDLE_BODY_Z0 = 0.3     # нижняя грань тела (стоит на подставке)
CANDLE_BODY_Z1 = 2.5     # верхняя грань тела (до начала углубления и фитиля)

# Отверстие под фитиль – вычитается из воска
WICK_R = 0.1             # радиус отверстия под фитиль
WICK_Z0 = 2.5            # начало отверстия по высоте (у поверхности свечи)
WICK_Z1 = 3.0            # конец отверстия по высоте (чуть выше поверхности)

# Параметры вмятины сверху (углубление под фитиль, чашеобразная выемка)
DIP_R = 0.45             # радиус углубления (часть верхнего сечения свечи)
DIP_DEPTH = 0.3          # глубина углубления относительно плоского верха свечи

# Параметры фитиля (палочка, которая торчит вверх в центре углубления)
WICK_STICK_R = 0.05      # радиус фитиля – тонкая вертикальная палочка
WICK_STICK_Z0 = CANDLE_BODY_Z1 - DIP_DEPTH  # стартовая высота фитиля: дно выемки
WICK_STICK_Z1 = 3.2      # конечная высота фитиля: насколько он выступает над свечой


def candle_base(x: float, y: float, z: float) -> float:
    """
    Круглая подставка под свечу.
    Широкий цилиндр по радиусу, ограниченный по z в [CANDLE_BASE_Z0; CANDLE_BASE_Z1].
    """
    f_cyl = _cyl_R(x, y, CANDLE_BASE_R)
    f_z_ge = _plane_z_ge(z, CANDLE_BASE_Z0)
    f_z_le = _plane_z_le(z, CANDLE_BASE_Z1)
    return r_intersection_alpha0(r_intersection_alpha0(f_cyl, f_z_ge), f_z_le)


def candle_body(x: float, y: float, z: float) -> float:
    """
    Основной цилиндр тела свечи.
    Радиус CANDLE_BODY_R, высота [CANDLE_BODY_Z0; CANDLE_BODY_Z1].
    """
    f_cyl = _cyl_R(x, y, CANDLE_BODY_R)
    f_z_ge = _plane_z_ge(z, CANDLE_BODY_Z0)
    f_z_le = _plane_z_le(z, CANDLE_BODY_Z1)
    return r_intersection_alpha0(r_intersection_alpha0(f_cyl, f_z_ge), f_z_le)


def candle_wick(x: float, y: float, z: float) -> float:
    """
    Отверстие под фитиль — тонкий вертикальный цилиндр, который будет
    ВЫЧИТАТЬСЯ из воска.
    """
    f_cyl = _cyl_R(x, y, WICK_R)
    f_z_ge = _plane_z_ge(z, WICK_Z0)
    f_z_le = _plane_z_le(z, WICK_Z1)
    return r_intersection_alpha0(r_intersection_alpha0(f_cyl, f_z_ge), f_z_le)


def candle_wick_stick(x: float, y: float, z: float) -> float:
    """
    Фитиль — тонкая палочка, которая торчит вверх из углубления.
    Это ДОБАВЛЯЕТСЯ к свече (не вычитается).
    """
    f_cyl = _cyl_R(x, y, WICK_STICK_R)
    f_z_ge = _plane_z_ge(z, WICK_STICK_Z0)
    f_z_le = _plane_z_le(z, WICK_STICK_Z1)
    return r_intersection_alpha0(r_intersection_alpha0(f_cyl, f_z_ge), f_z_le)


def candle_wax(x: float, y: float, z: float) -> float:
    """
    Объединённый воск + подставка (без отверстия под фитиль).
    Верх свечи — плоский, без сферического купола.
    """
    f_base = candle_base(x, y, z)
    f_body = candle_body(x, y, z)

    return r_union_alpha0(f_base, f_body)


def candle_dip(x: float, y: float, z: float) -> float:
    """
    Углубление (вмятина) в верхней части свечи.

    Модель: параболоид, вогнутый вниз.
      - в центре (r = 0) глубина DIP_DEPTH ниже плоского верха (CANDLE_BODY_Z1)
      - на радиусе r = DIP_R совпадает с плоским верхом

    z_par(r) = CANDLE_BODY_Z1 - DIP_DEPTH + (DIP_DEPTH / DIP_R^2) * r^2
    F_par    = z - z_par(r)

    Углубление (область "воздуха") задаём как пересечение:
      F_par > 0 (выше параболоида)
      z <= CANDLE_BODY_Z1 (ниже плоского верха)
      r <= DIP_R (внутри маленького цилиндра)
    """
    r2 = x * x + y * y
    a = DIP_DEPTH / (DIP_R * DIP_R)
    z_par = CANDLE_BODY_Z1 - DIP_DEPTH + a * r2

    f_par = z - z_par                       # выше параболоида
    f_z_le_top = _plane_z_le(z, CANDLE_BODY_Z1)  # ниже верха свечи
    f_cyl = _cyl_R(x, y, DIP_R)            # внутри малого радиуса

    return r_intersection_alpha0(
        r_intersection_alpha0(f_par, f_z_le_top),
        f_cyl,
    )


def candle(x: float, y: float, z: float) -> float:
    """
    Полная фигура свечи с подставкой, углублением и фитилем.

    Собирается так:
      F_wax  = base ∪ body
      F      = F_wax ∧ NOT(F_wick) ∧ NOT(F_dip)  (вычитаем отверстие и углубление)
      F      = F ∪ F_wick_stick  (добавляем фитиль-палочку)
    """
    f_wax = candle_wax(x, y, z)

    # Вычитаем отверстие под фитиль
    f_wick = candle_wick(x, y, z)
    f_outside_wick = r_inversion(f_wick)
    f = r_intersection_alpha0(f_wax, f_outside_wick)

    # Вычитаем углубление сверху
    f_dip = candle_dip(x, y, z)
    f_outside_dip = r_inversion(f_dip)
    f = r_intersection_alpha0(f, f_outside_dip)

    # Добавляем фитиль-палочку, которая торчит вверх
    f_wick_stick = candle_wick_stick(x, y, z)
    f = r_union_alpha0(f, f_wick_stick)

    return f


# ============================================================================
# ПАРАМЕТРЫ ГЕНЕРАЦИИ (опционально)
# ============================================================================

# Границы области моделирования
xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0
zmin, zmax = 0.0, 3.5  # подняли область вверх под свечу

# Размеры сетки
nx = 128  # число узлов по X
ny = 128  # число узлов по Y
nz = 256  # число слоёв по Z

# Директория для сохранения
output_dir = "images"


# ============================================================================
# ФУНКЦИЯ СТАРТА ГЕНЕРАЦИИ
# ============================================================================

def start(F: Union[Callable, List[Callable]],
          xmin: float = -2.0, xmax: float = 2.0,
          ymin: float = -2.0, ymax: float = 2.0,
          zmin: float = -2.0, zmax: float = 2.0,
          nx: int = 128, ny: int = 128, nz: int = 256,
          output_dir: str = "images"):
    """
    Запускает генерацию M-образов для заданной функции или списка функций.
    
    Args:
        F: Функция F(x,y,z) или список функций [F1, F2, ...]
            Если передана одна функция - генерирует M-образы для неё.
            Если передан список - генерирует M-образы для каждой функции отдельно.
        xmin, xmax, ymin, ymax, zmin, zmax: Границы области моделирования
        nx, ny, nz: Размеры сетки
        output_dir: Базовая директория для сохранения (для каждой функции создаётся подпапка)
    
    Примеры:
        # Одна функция:
        start(sphere)
        
        # Несколько функций:
        start([sphere, ellipsoid, paraboloid])
        
        # С параметрами:
        start(sphere, nx=256, nz=512, output_dir="my_images")
    """
    # Преобразуем одну функцию в список для единообразной обработки
    if not isinstance(F, list):
        functions = [F]
    else:
        functions = F
    
    # Генерируем M-образы для каждой функции
    for idx, func in enumerate(functions):
        # Определяем имя функции для создания подпапки
        func_name = func.__name__ if hasattr(func, '__name__') else f"function_{idx+1}"
        
        # Создаём отдельную директорию для каждой функции
        if len(functions) > 1:
            current_output_dir = f"{output_dir}/{func_name}"
        else:
            current_output_dir = output_dir
        
        print(f"\n{'='*60}")
        print(f"Генерация M-образов для функции: {func_name}")
        print(f"{'='*60}\n")
        
        generate_mimages(
            F=func,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            zmin=zmin, zmax=zmax,
            nx=nx, ny=ny, nz=nz,
            output_dir=current_output_dir
        )
    
    print(f"\n{'='*60}")
    print(f"Все M-образы успешно сгенерированы!")
    print(f"{'='*60}")


# ============================================================================
# ЗАПУСК ГЕНЕРАЦИИ
# ============================================================================

if __name__ == "__main__":
    # Пример 1: Свеча с подставкой
    start(candle, zmin=0.0, zmax=3.5)
    
    # Пример 2: Другие фигуры (раскомментируйте при необходимости)
    # start([sphere, ellipsoid, paraboloid, cylinder])

