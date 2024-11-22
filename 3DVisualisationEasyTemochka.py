import tkinter as tk
from math import cos, sin, radians

# Класс для представления трехмерного вектора
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Операции сложения
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    # Операции вычитания
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    # Умножение на скаляр
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    # Скалярное произведение
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # Векторное произведение
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    # Нормализация
    def normalize(self):
        magnitude = (self.x**2 + self.y**2 + self.z**2) ** 0.5
        if magnitude == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / magnitude, self.y / magnitude, self.z / magnitude)

    # Преобразование в список для матричных операций
    def to_list(self):
        return [self.x, self.y, self.z, 1]

# Класс для 4x4 матрицы
class Mat4x4:
    def __init__(self, matrix):
        self.matrix = matrix  # Матрица 4x4

    # Перегрузка оператора @ для умножения матриц
    def __matmul__(self, other):
        result = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return Mat4x4(result)

    # Преобразование вектора с помощью матрицы
    def apply(self, vec):
        transformed = [0]*4
        v = vec.to_list()
        for i in range(4):
            for j in range(4):
                transformed[i] += self.matrix[i][j] * v[j]
        if transformed[3] != 0:
            return Vec3(transformed[0]/transformed[3], transformed[1]/transformed[3], transformed[2]/transformed[3])
        return Vec3(transformed[0], transformed[1], transformed[2])

# Функции для создания трансформационных матриц
def create_translation(dx, dy, dz):
    return Mat4x4([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def create_scaling(sx, sy, sz):
    return Mat4x4([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ])

def create_rotation_x(degrees):
    rad = radians(degrees)
    return Mat4x4([
        [1, 0,          0,         0],
        [0, cos(rad), -sin(rad), 0],
        [0, sin(rad), cos(rad),  0],
        [0, 0,          0,         1]
    ])

def create_rotation_y(degrees):
    rad = radians(degrees)
    return Mat4x4([
        [cos(rad),  0, sin(rad), 0],
        [0,         1, 0,        0],
        [-sin(rad), 0, cos(rad), 0],
        [0,         0, 0,        1]
    ])

def create_rotation_z(degrees):
    rad = radians(degrees)
    return Mat4x4([
        [cos(rad), -sin(rad), 0, 0],
        [sin(rad), cos(rad),  0, 0],
        [0,        0,         1, 0],
        [0,        0,         0, 1]
    ])

# Класс камеры
class Camera:
    def __init__(self, pos, tilt_x=0, tilt_y=0, tilt_z=0, screen_dist=1):
        self.position = pos
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.tilt_z = tilt_z
        self.screen_distance = screen_dist
        self.view_mat = None
        self.update_view_matrix()

    def update_view_matrix(self):
        rot_x = create_rotation_x(self.tilt_x)
        rot_y = create_rotation_y(self.tilt_y)
        rot_z = create_rotation_z(self.tilt_z)
        rotation = rot_z @ rot_y @ rot_x
        translation = create_translation(-self.position.x, -self.position.y, -self.position.z)
        self.view_mat = rotation @ translation

    def set_orientation(self, tilt_x, tilt_y, tilt_z):
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.tilt_z = tilt_z
        self.update_view_matrix()

    def project(self, vertex):
        transformed = self.view_mat.apply(vertex)
        if transformed.z == 0:
            return None
        scale = self.screen_distance / transformed.z
        x_proj = transformed.x * scale
        y_proj = transformed.y * scale
        return (x_proj, y_proj)

# Класс модели
class Model:
    def __init__(self, vert_file, face_file):
        self.vertices = []
        self.faces = []
        self.load_vertices(vert_file)
        self.load_faces(face_file)
        self.normalize()

    def load_vertices(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x, y, z = map(float, parts[:3])
                self.vertices.append(Vec3(x, y, z))

    def load_faces(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                indices = tuple(map(int, line.strip().split()))
                if len(indices) < 2:
                    continue
                self.faces.append(indices)

    def normalize(self):
        max_val = max(max(abs(v.x), abs(v.y), abs(v.z)) for v in self.vertices)
        if max_val == 0:
            return
        for v in self.vertices:
            v.x /= max_val
            v.y /= max_val
            v.z /= max_val

    def transform(self, mat):
        self.vertices = [mat.apply(v) for v in self.vertices]

# Класс сцены для визуализации
class SceneViewer:
    def __init__(self, model, camera):
        self.model = model
        self.camera = camera
        self.root = tk.Tk()
        self.root.title("3D Model Viewer")
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        self.center = (self.canvas_width // 2, self.canvas_height // 2)
        self.create_controls()
        self.draw_model()

    def create_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Наклон по X
        tk.Label(control_frame, text="Поворот по X:").grid(row=0, column=0, padx=5)
        self.angle_x = tk.DoubleVar(value=self.camera.tilt_x)
        tk.Entry(control_frame, textvariable=self.angle_x, width=10).grid(row=0, column=1, padx=5)

        # Наклон по Y
        tk.Label(control_frame, text="Поворот по Y:").grid(row=1, column=0, padx=5)
        self.angle_y = tk.DoubleVar(value=self.camera.tilt_y)
        tk.Entry(control_frame, textvariable=self.angle_y, width=10).grid(row=1, column=1, padx=5)

        # Наклон по Z
        tk.Label(control_frame, text="Поворот по Z:").grid(row=2, column=0, padx=5)
        self.angle_z = tk.DoubleVar(value=self.camera.tilt_z)
        tk.Entry(control_frame, textvariable=self.angle_z, width=10).grid(row=2, column=1, padx=5)

        # Кнопка применения наклона
        apply_btn = tk.Button(control_frame, text="Применить", command=self.update_view)
        apply_btn.grid(row=3, column=0, columnspan=2, pady=5)

    def project_point(self, point):
        proj = self.camera.project(point)
        if proj is None:
            return None
        x, y = proj
        screen_x = self.center[0] + x * 100
        screen_y = self.center[1] - y * 100  # Инверсия Y для экранных координат
        return (screen_x, screen_y)

    def draw_model(self):
        for face in self.model.faces:
            projected = []
            for idx in face:
                if 0 <= idx < len(self.model.vertices):
                    p = self.project_point(self.model.vertices[idx])
                    if p:
                        projected.append(p)
            if len(projected) >= 2:
                for i in range(len(projected)):
                    start = projected[i]
                    end = projected[(i + 1) % len(projected)]
                    self.canvas.create_line(start[0], start[1], end[0], end[1], fill="black")

    def update_view(self):
        # Обновление ориентации камеры
        self.camera.set_orientation(self.angle_x.get(), self.angle_y.get(), self.angle_z.get())
        # Очистка канваса и перерисовка модели
        self.canvas.delete("all")
        self.draw_model()

    def run(self):
        self.root.mainloop()

# Основной блок выполнения
if __name__ == "__main__":
    # Загрузка модели из файлов
    модель = Model("vertices_cube.txt", "faces_cube.txt")

    # Настройка камеры
    позиция_камеры = Vec3(1, 0, -5)
    камера = Camera(pos=позиция_камеры, tilt_x=0, tilt_y=0, tilt_z=0, screen_dist=1)

    # Создание и запуск визуализатора сцены
    визуализатор = SceneViewer(модель, камера)
    визуализатор.run()
