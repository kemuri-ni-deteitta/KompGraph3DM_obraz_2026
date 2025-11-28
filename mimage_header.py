"""
Небольшой модуль, отвечающий только за «шапку» M‑образов.

Мы договорились, что строка с индексом y = 0 в каждом BMP не участвует в хранении нормалей,
а целиком отведена под служебную информацию: размеры сетки, диапазон координат и диапазон F.

Здесь две функции:
  * `read_header_from_mimage`  — аккуратно читает эти параметры из любой из картинок Mi;
  * `write_header_to_images`   — записывает один и тот же заголовок во все пять M‑образов.

Так визуализатору не нужно ничего «догадываться» – он просто читает первую строку
и восстанавливает параметры объёма.
"""

from PIL import Image
from mimage_common import HEADER_SHIFT, decode_header_param


def read_header_from_mimage(path: str):
    """
    Читает параметры заголовка из M-образа.
    
    Формат заголовка (9 параметров в пикселях [0..8, 0]):
    0: SIZE_Z (nz)
    1: xmin
    2: xmax
    3: ymin
    4: ymax
    5: zmin
    6: zmax
    7: umin
    8: umax
    
    Args:
        path: Путь к M-образу (любому из M1..M5)
    
    Returns:
        Кортеж: (nx, ny, SIZE_Z, xmin, xmax, ymin, ymax, zmin, zmax, umin, umax)
    """
    img = Image.open(path)
    nx, ny = img.size
    pix = img.load()

    params = []
    for idx in range(9):   # 9 параметров: 0..8 пиксели верхней строки
        r, g, b = pix[idx, 0]
        value = decode_header_param(r, g, b)
        params.append(value)

    SIZE_Z  = int(params[0])
    xmin    = float(params[1])
    xmax    = float(params[2])
    ymin    = float(params[3])
    ymax    = float(params[4])
    zmin    = float(params[5])
    zmax    = float(params[6])
    umin    = float(params[7])
    umax    = float(params[8])

    return nx, ny, SIZE_Z, xmin, xmax, ymin, ymax, zmin, zmax, umin, umax


def write_header_to_images(pix_list, nz, xmin, xmax, ymin, ymax, zmin, zmax, umin, umax):
    """
    Записывает параметры заголовка в строку y=0 всех 5 M-образов.
    
    Args:
        pix_list: Список из 5 объектов pix (результат img.load() для каждого M-образа)
        nz: Количество слоёв по Z
        xmin, xmax, ymin, ymax, zmin, zmax: Границы области моделирования
        umin, umax: Минимальное и максимальное значение функции F
    """
    from mimage_common import encode_header_param
    
    params = [
        nz,          # 0: SIZE_Z
        xmin, xmax,  # 1–2: xmin/xmax
        ymin, ymax,  # 3–4: ymin/ymax
        zmin, zmax,  # 5–6: zmin/zmax
        int(umin),   # 7: umin (по методичке привели к int)
        int(umax)    # 8: umax (int)
    ]

    for m in range(5):
        for idx, prm in enumerate(params):
            r, g, b = encode_header_param(prm)
            pix_list[m][idx, 0] = (r, g, b)

