"""
Визуализатор M‑образов с использованием PyVista (обёртка над VTK).

Этот скрипт не знает ничего о том, какая именно фигура генерировалась.
Он делает ровно следующее:
  1. Берёт один из файлов Mi из папки с M‑образами и считывает из него заголовок
     (размеры сетки, диапазон координат и значения F);
  2. Для каждого слоя и для всех пяти файлов Mi декодирует компоненты нормали;
  3. Строит облако точек по центрам тех ячеек, где нормаль ненулевая — именно там
     проходит поверхность F=0;
  4. Отображает это облако точек (и, при желании, стрелки нормалей) в интерактивном окне.

Так визуализатор остаётся универсальным: он не «угадывает» форму, а честно читает то,
что подготовил генератор.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import pyvista as pv
except ImportError:
    print("PyVista не установлен. Установите: pip install pyvista")
    exit(1)

from mimage_header import read_header_from_mimage
from mimage_common import P_MAX


# -------------------------------------------------------------
# Чтение нормалей из M-образов (тот же алгоритм)
# -------------------------------------------------------------
def _decode_normals_from_layer(img: Image.Image, nx: int, ny: int) -> np.ndarray:
    """
    Преобразует один BMP‑слой (один из файлов Mi) в числовую компоненту нормали.

    На выходе получаем массив shape (ny-1, nx-1), потому что:
      * строка y = 0 отведена под заголовок и не содержит нормалей;
      * последняя колонка по X не используется для ячеек (она граничная).
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.uint32)
    # Первую строку (y=0) используем под заголовок, поэтому её игнорируем.
    # Берём строки 1..ny-1 и столбцы 0..nx-2.
    arr = arr[1:ny, : nx - 1, :]
    color = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    normals = 2.0 * color.astype(np.float32) / P_MAX - 1.0
    return normals


def read_normals_from_mimages(mimage_dir: str, nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Читает все компоненты нормалей из набора M-образов.

    Returns:
        np.ndarray shape (nz-1, ny-1, nx-1, 5)
    """
    normals = np.zeros((nz - 1, ny - 1, nx - 1, 5), dtype=np.float32)

    print("Читаем нормали из M-образов...")
    for k in range(nz - 1):
        for m in range(5):
            path = os.path.join(mimage_dir, f"{k + 1} (M{m + 1}).bmp")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден файл {path}")

            img = Image.open(path)
            normals[k, :, :, m] = _decode_normals_from_layer(img, nx, ny)

    return normals


# -------------------------------------------------------------
# Построение облака точек
# -------------------------------------------------------------
def build_point_cloud_from_normals(
    normals: np.ndarray,
    header: Tuple[int, int, int, float, float, float, float, float, float, float, float],
    magnitude_threshold: float = 0.05,
    downsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Превращает нормали в облако точек (центры ячеек, где поверхность пересекает ячейку).

    Args:
        normals: Массив нормалей shape (nz-1, ny-1, nx-1, 5)
        header: Параметры из заголовка
        magnitude_threshold: Порог для фильтрации ячеек
        downsample_factor: Фактор прореживания (1 = все точки, 2 = каждая вторая, и т.д.)

    Returns:
        points: shape (N, 3)
        point_normals: shape (N, 3)
    """
    nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, _, _ = header

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)

    cell_normals = normals  # shape (nz-1, ny-1, nx-1, 5)
    gradients = cell_normals[..., :3]
    magnitudes = np.linalg.norm(gradients, axis=-1)
    mask = magnitudes > magnitude_threshold

    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3))

    # Прореживание для ускорения
    if downsample_factor > 1:
        mask = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        gradients = gradients[::downsample_factor, ::downsample_factor, ::downsample_factor]
        nz_sub, ny_sub, nx_sub = mask.shape
        cell_x = xmin + (np.arange(nx_sub) * downsample_factor + 0.5) * dx
        cell_y = ymin + (np.arange(ny_sub) * downsample_factor + 0.5) * dy
        cell_z = zmin + (np.arange(nz_sub) * downsample_factor + 0.5) * dz
    else:
        cell_x = xmin + (np.arange(nx - 1) + 0.5) * dx
        cell_y = ymin + (np.arange(ny - 1) + 0.5) * dy
        cell_z = zmin + (np.arange(nz - 1) + 0.5) * dz

    grid_z, grid_y, grid_x = np.meshgrid(cell_z, cell_y, cell_x, indexing="ij")

    points = np.column_stack((grid_x[mask], grid_y[mask], grid_z[mask]))
    point_normals = gradients[mask]

    # Нормализуем нормали
    norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
    point_normals = np.where(norms > 1e-10, point_normals / norms, point_normals)

    return points, point_normals


# -------------------------------------------------------------
# Визуализация через PyVista
# -------------------------------------------------------------
def visualize_with_pyvista(
    points: np.ndarray,
    normals: np.ndarray,
    show_normals: bool = False,
    normal_scale: float = 0.1,
    point_size: float = 3.0,
    max_points: int = 500_000,
) -> None:
    """
    Визуализирует облако точек через PyVista.

    Args:
        points: Массив точек shape (N, 3)
        normals: Массив нормалей shape (N, 3)
        show_normals: Показывать ли стрелки нормалей
        normal_scale: Масштаб стрелок нормалей
        point_size: Размер точек
        max_points: Максимальное количество точек для отображения
    """
    print(f"\nВизуализация {len(points)} точек...")

    # Ограничиваем количество точек для производительности
    if len(points) > max_points:
        print(f"  Ограничиваем до {max_points} точек для производительности...")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        normals = normals[indices]

    # Создаём PyVista point cloud
    point_cloud = pv.PolyData(points)
    point_cloud["normals"] = normals

    # Цвет по Z-координате для наглядности
    point_cloud["Z"] = points[:, 2]

    # Создаём plotter
    plotter = pv.Plotter()
    plotter.add_mesh(
        point_cloud,
        scalars="Z",
        point_size=point_size,
        render_points_as_spheres=True,
        cmap="coolwarm",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Z координата"},
    )

    # Опционально показываем нормали
    if show_normals and len(points) < 50_000:  # Только для небольшого количества точек
        print("  Добавляем стрелки нормалей...")
        plotter.add_arrows(
            points,
            normals * normal_scale,
            mag=normal_scale,
            color="black",
        )

    # Настройки камеры
    plotter.camera_position = "iso"
    plotter.show_axes()

    # Заголовок
    plotter.add_text(
        "Поверхность из M-образов (PyVista)",
        font_size=12,
        position="upper_left",
    )

    print("\nОткрывается окно визуализации...")
    print("Управление:")
    print("  - ЛКМ + движение: вращение")
    print("  - ПКМ + движение: панорамирование")
    print("  - Колесо мыши: масштабирование")
    print("  - 'q' или закрыть окно: выход")

    plotter.show()


# -------------------------------------------------------------
# Точка входа
# -------------------------------------------------------------
if __name__ == "__main__":
    m_image_path = "images/128 (M1).bmp"
    mimage_dir = os.path.dirname(m_image_path) or "images"

    print("Читаем заголовок из M-образа...")
    header = read_header_from_mimage(m_image_path)
    nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, umin, umax = header

    print("Параметры из шапки M-образа:")
    print(f"  SIZE_Z (nz) = {nz}")
    print(f"  X ∈ [{xmin}; {xmax}]")
    print(f"  Y ∈ [{ymin}; {ymax}]")
    print(f"  Z ∈ [{zmin}; {zmax}]")
    print(f"  u ∈ [{umin}; {umax}]")
    print(f"  nx = {nx}, ny = {ny}")

    # Читаем нормали
    normals = read_normals_from_mimages(mimage_dir, nx, ny, nz)

    # Строим облако точек
    # Можно настроить downsample_factor для ускорения (1 = все точки, 2 = каждая вторая)
    points, point_normals = build_point_cloud_from_normals(
        normals,
        header,
        magnitude_threshold=0.05,
        downsample_factor=1,  # Измените на 2 или 3 для ускорения
    )

    print(f"\nПостроено {len(points)} точек поверхности")

    # Визуализируем
    visualize_with_pyvista(
        points,
        point_normals,
        show_normals=False,  # Установите True для показа стрелок (медленнее)
        normal_scale=0.1,
        point_size=3.0,
        max_points=500_000,  # Ограничение для производительности
    )

