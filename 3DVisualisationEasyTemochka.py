import tkinter as tk
from math import cos, sin, radians

# Класс для трехмерных векторов
class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Сложение векторов
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    # Вычитание векторов
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    # Умножение вектора на скаляр
    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    # Скалярное произведение
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # Векторное произведение
    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    # Нормализация вектора
    def normalize(self):
        length = (self.x**2 + self.y**2 + self.z**2) ** 0.5
        if length == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / length, self.y / length, self.z / length)

    # Преобразование вектора в список
    def to_list(self):
        return [self.x, self.y, self.z, 1]

# Класс для 4x4 матриц
class Matrix4x4:
    def __init__(self, data):
        self.data = data  # data - двумерный список 4x4

    # Умножение матрицы на матрицу
    def __matmul__(self, other):
        result = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix4x4(result)

    # Умножение матрицы на вектор
    def transform_vector(self, vec):
        res = [0]*4
        vec_list = vec.to_list()
        for i in range(4):
            for j in range(4):
                res[i] += self.data[i][j] * vec_list[j]
        # Нормализация координаты w
        if res[3] != 0:
            return Vector3D(res[0]/res[3], res[1]/res[3], res[2]/res[3])
        else:
            return Vector3D(res[0], res[1], res[2])

# Функции для создания матриц преобразований
def translation_matrix(dx, dy, dz):
    return Matrix4x4([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def scaling_matrix(sx, sy, sz):
    return Matrix4x4([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ])

def rotation_matrix_x(angle):
    angle = radians(angle)
    return Matrix4x4([
        [1, 0,          0,         0],
        [0, cos(angle), -sin(angle), 0],
        [0, sin(angle), cos(angle),  0],
        [0, 0,          0,         1]
    ])

def rotation_matrix_y(angle):
    angle = radians(angle)
    return Matrix4x4([
        [cos(angle),  0, sin(angle), 0],
        [0,           1, 0,          0],
        [-sin(angle), 0, cos(angle), 0],
        [0,           0, 0,          1]
    ])

def rotation_matrix_z(angle):
    angle = radians(angle)
    return Matrix4x4([
        [cos(angle), -sin(angle), 0, 0],
        [sin(angle), cos(angle),  0, 0],
        [0,          0,           1, 0],
        [0,          0,           0, 1]
    ])

# Класс камеры
class Camera:
    def __init__(self, position, tilt_x=0, tilt_y=0, tilt_z=0, screen_distance=1):
        self.position = position
        self.tilt_x = tilt_x  # Угол наклона по оси X
        self.tilt_y = tilt_y  # Угол наклона по оси Y
        self.tilt_z = tilt_z  # Угол наклона по оси Z
        self.screen_distance = screen_distance
        self.update_view_matrix()

    def update_view_matrix(self):
        # Создаем матрицы поворота
        rx = rotation_matrix_x(self.tilt_x)
        ry = rotation_matrix_y(self.tilt_y)
        rz = rotation_matrix_z(self.tilt_z)
        # Последовательное применение поворотов
        rotation = rz @ ry @ rx
        # Создаем матрицу переноса камеры
        translation = translation_matrix(-self.position.x, -self.position.y, -self.position.z)
        # Полная матрица вида
        self.view_matrix = rotation @ translation

    def set_tilt(self, tilt_x, tilt_y, tilt_z):
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.tilt_z = tilt_z
        self.update_view_matrix()

    # Проекция точки на экран
    def project_vertex(self, vertex):
        # Применяем матрицу вида к вершине
        transformed = self.view_matrix.transform_vector(vertex)
        # Простая перспективная проекция
        if transformed.z == 0:
            return None
        factor = self.screen_distance / transformed.z
        x = transformed.x * factor
        y = transformed.y * factor
        return (x, y)

# Класс модели
class Model:
    def __init__(self, vertices_file, faces_file):
        self.vertices = []  # Список Vector3D
        self.faces = []     # Список кортежей с индексами вершин
        self.load_vertices(vertices_file)
        self.load_faces(faces_file)
        self.normalize_vertices()

    def load_vertices(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x_str, y_str, z_str = parts[:3]
                x, y, z = float(x_str), float(y_str), float(z_str)
                self.vertices.append(Vector3D(x, y, z))

    def load_faces(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                indices = tuple(map(int, line.strip().split()))
                if len(indices) < 2:
                    continue
                self.faces.append(indices)

    def normalize_vertices(self):
        # Нормализация координат
        max_coord = max(
            max(abs(vertex.x), abs(vertex.y), abs(vertex.z)) for vertex in self.vertices
        )
        if max_coord == 0:
            return
        for vertex in self.vertices:
            vertex.x /= max_coord
            vertex.y /= max_coord
            vertex.z /= max_coord

    def apply_transformation(self, matrix):
        transformed_vertices = []
        for vertex in self.vertices:
            transformed_vertex = matrix.transform_vector(vertex)
            transformed_vertices.append(transformed_vertex)
        self.vertices = transformed_vertices

# Класс сцены
class Scene:
    def __init__(self, model, camera):
        self.model = model
        self.camera = camera

    def render(self):
        # Создаем окно для отображения
        window = tk.Tk()
        window.title("3D Model Visualization")
        canvas_width = 800
        canvas_height = 600
        canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
        canvas.pack()

        # Центр экрана
        center_x = canvas_width // 2
        center_y = canvas_height // 2

        # Проецируем и отрисовываем каждую грань модели
        for face in self.model.faces:
            points = []
            for vertex_index in face:
                if vertex_index < 0 or vertex_index >= len(self.model.vertices):
                    continue
                vertex = self.model.vertices[vertex_index]
                screen_coords = self.camera.project_vertex(vertex)
                if screen_coords:
                    # Масштабирование и сдвиг в центр экрана
                    x_screen = center_x + screen_coords[0] * 100
                    y_screen = center_y - screen_coords[1] * 100  # Инвертируем Y для экранных координат
                    points.append((x_screen, y_screen))
            if len(points) >= 2:
                # Отрисовываем ребра грани
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % len(points)]
                    canvas.create_line(
                        x1, y1,
                        x2, y2,
                        fill="black"
                    )

        # Функция для обновления сцены при изменении параметров камеры
        def update_scene():
            # Очистка канваса
            canvas.delete("all")
            # Повторная отрисовка
            for face in self.model.faces:
                points = []
                for vertex_index in face:
                    if vertex_index < 0 or vertex_index >= len(self.model.vertices):
                        continue
                    vertex = self.model.vertices[vertex_index]
                    screen_coords = self.camera.project_vertex(vertex)
                    if screen_coords:
                        x_screen = center_x + screen_coords[0] * 100
                        y_screen = center_y - screen_coords[1] * 100
                        points.append((x_screen, y_screen))
                if len(points) >= 2:
                    for i in range(len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[(i + 1) % len(points)]
                        canvas.create_line(
                            x1, y1,
                            x2, y2,
                            fill="black"
                        )

        # Интерфейс для изменения наклона камеры
        control_frame = tk.Frame(window)
        control_frame.pack()

        tk.Label(control_frame, text="Наклон X:").grid(row=0, column=0)
        tilt_x_var = tk.DoubleVar(value=self.camera.tilt_x)
        tilt_x_entry = tk.Entry(control_frame, textvariable=tilt_x_var)
        tilt_x_entry.grid(row=0, column=1)

        tk.Label(control_frame, text="Наклон Y:").grid(row=1, column=0)
        tilt_y_var = tk.DoubleVar(value=self.camera.tilt_y)
        tilt_y_entry = tk.Entry(control_frame, textvariable=tilt_y_var)
        tilt_y_entry.grid(row=1, column=1)

        tk.Label(control_frame, text="Наклон Z:").grid(row=2, column=0)
        tilt_z_var = tk.DoubleVar(value=self.camera.tilt_z)
        tilt_z_entry = tk.Entry(control_frame, textvariable=tilt_z_var)
        tilt_z_entry.grid(row=2, column=1)

        def apply_tilt():
            tilt_x = tilt_x_var.get()
            tilt_y = tilt_y_var.get()
            tilt_z = tilt_z_var.get()
            self.camera.set_tilt(tilt_x, tilt_y, tilt_z)
            update_scene()

        apply_button = tk.Button(control_frame, text="Применить наклон", command=apply_tilt)
        apply_button.grid(row=3, column=0, columnspan=2)

        window.mainloop()

# Пример использования
if __name__ == "__main__":
    # Создаем модель из файлов
    model = Model("vertices_pyramide.txt", "faces_pyramide.txt")

    # Параметры камеры
    camera_position = Vector3D(0, 0, -5)
    camera = Camera(position=camera_position, tilt_x=0, tilt_y=0, tilt_z=0, screen_distance=1)

    # Создаем и отображаем сцену
    scene = Scene(model, camera)
    scene.render()
