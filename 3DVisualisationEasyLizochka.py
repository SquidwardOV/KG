import tkinter as tk
from math import cos, sin, radians

# Класс для представления трехмерного вектора
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Преобразование вектора в список для матричных операций
    def to_list(self):
        return [self.x, self.y, self.z, 1]

# Класс для 4x4 матриц
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
    def __init__(self, pos, tilt_x=0, tilt_y=0, tilt_z=0, screen_dist=1, projection='orthographic'):
        self.position = pos
        self.tilt_x = tilt_x  # Угол наклона по оси X
        self.tilt_y = tilt_y  # Угол наклона по оси Y
        self.tilt_z = tilt_z  # Угол наклона по оси Z
        self.screen_distance = screen_dist
        self.projection = projection  # Только 'orthographic'
        self.view_mat = None
        self.update_view_matrix()

    def update_view_matrix(self):
        rot_x = create_rotation_x(self.tilt_x)
        rot_y = create_rotation_y(self.tilt_y)
        rot_z = create_rotation_z(self.tilt_z)
        rotation = rot_z @ rot_y @ rot_x
        translation = create_translation(-self.position.x, -self.position.y, -self.position.z)
        self.view_mat = rotation @ translation

    def increment_tilt(self, delta_x, delta_y, delta_z):
        self.tilt_x += delta_x
        self.tilt_y += delta_y
        self.tilt_z += delta_z
        self.update_view_matrix()

    # Ортографическая проекция
    def project(self, vertex):
        transformed = self.view_mat.apply(vertex)
        # Ортографическая проекция с учетом screen_distance как масштаб
        x = transformed.x / self.screen_distance
        y = transformed.y / self.screen_distance
        return (x, y)

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
        self.collect_unique_edges()
        self.draw_model()

    def collect_unique_edges(self):
        self.unique_edges = set()
        for face in self.model.faces:
            num_vertices = len(face)
            for i in range(num_vertices):
                v1 = face[i]
                v2 = face[(i + 1) % num_vertices]
                edge = tuple(sorted((v1, v2)))
                self.unique_edges.add(edge)

    def create_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Наклон по X
        tk.Label(control_frame, text="Поворот по X:").grid(row=0, column=0, padx=5, pady=2)
        self.angle_x = tk.DoubleVar(value=0)
        tilt_x_entry = tk.Entry(control_frame, textvariable=self.angle_x, width=10)
        tilt_x_entry.grid(row=0, column=1, padx=5, pady=2)

        # Наклон по Y
        tk.Label(control_frame, text="Поворот по Y:").grid(row=1, column=0, padx=5, pady=2)
        self.angle_y = tk.DoubleVar(value=0)
        tilt_y_entry = tk.Entry(control_frame, textvariable=self.angle_y, width=10)
        tilt_y_entry.grid(row=1, column=1, padx=5, pady=2)

        # Наклон по Z
        tk.Label(control_frame, text="Поворот по Z:").grid(row=2, column=0, padx=5, pady=2)
        self.angle_z = tk.DoubleVar(value=0)
        tilt_z_entry = tk.Entry(control_frame, textvariable=self.angle_z, width=10)
        tilt_z_entry.grid(row=2, column=1, padx=5, pady=2)

        # Кнопка применения наклона
        apply_btn = tk.Button(control_frame, text="Применить", command=self.update_view)
        apply_btn.grid(row=3, column=0, columnspan=2, pady=5)

    def project_point(self, point):
        proj = self.camera.project(point)
        if proj is None:
            return None
        x, y = proj
        scale = 300  # Масштабирование
        screen_x = self.center[0] + x * scale
        screen_y = self.center[1] - y * scale  # Инверсия Y для экранных координат
        return (screen_x, screen_y)

    def draw_model(self):
        for edge in self.unique_edges:
            v1, v2 = edge
            if 0 <= v1 < len(self.model.vertices) and 0 <= v2 < len(self.model.vertices):
                vertex1 = self.model.vertices[v1]
                vertex2 = self.model.vertices[v2]
                screen1 = self.project_point(vertex1)
                screen2 = self.project_point(vertex2)
                if screen1 and screen2:
                    self.canvas.create_line(screen1[0], screen1[1], screen2[0], screen2[1], fill="black")

    def update_view(self):
        # Обновление ориентации камеры с накопительным эффектом
        delta_x = self.angle_x.get()
        delta_y = self.angle_y.get()
        delta_z = self.angle_z.get()
        self.camera.increment_tilt(delta_x, delta_y, delta_z)
        # Очистка канваса и перерисовка модели
        self.canvas.delete("all")
        self.draw_model()


    def run(self):
        self.root.mainloop()


# Основной блок выполнения
if __name__ == "__main__":
    # Загрузка модели куба из готовых файлов
    модель = Model("vertices_cube.txt", "faces_cube.txt")

    # Настройка камеры
    позиция_камеры = Vec3(0, 0, 3)
    камера = Camera(pos=позиция_камеры, tilt_x=0, tilt_y=0, tilt_z=0, screen_dist=5)

    # Создание и запуск визуализатора сцены
    визуализатор = SceneViewer(модель, камера)
    визуализатор.run()
