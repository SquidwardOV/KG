import tkinter as tk
from math import cos, sin, radians

# Класс для трехмерных векторов
class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Преобразование вектора в список
    def to_list(self):
        return [self.x, self.y, self.z, 1]

# Класс для 4x4 матриц
class Matrix4x4:
    def __init__(self, data):
        self.data = data  # data - двумерный список 4x4

    # Умножение матрицы на матрицу
    def __matmul__(self, other):
        result = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix4x4(result)

    # Умножение матрицы на вектор
    def transform_vector(self, vec):
        res = [0] * 4
        vec_list = vec.to_list()
        for i in range(4):
            for j in range(4):
                res[i] += self.data[i][j] * vec_list[j]
        # Нормализация координаты w
        if res[3] != 0:
            return Vector3D(res[0] / res[3], res[1] / res[3], res[2] / res[3])
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
    def __init__(self, position, tilt_x=0, tilt_y=0, tilt_z=0, screen_distance=1, projection='orthographic'):
        self.position = position
        self.tilt_x = tilt_x  # Угол наклона по оси X
        self.tilt_y = tilt_y  # Угол наклона по оси Y
        self.tilt_z = tilt_z  # Угол наклона по оси Z
        self.screen_distance = screen_distance
        self.projection = projection  # 'orthographic'
        self.update_view_matrix()

    def update_view_matrix(self):
        rx = rotation_matrix_x(self.tilt_x)
        ry = rotation_matrix_y(self.tilt_y)
        rz = rotation_matrix_z(self.tilt_z)
        rotation = rz @ ry @ rx
        translation = translation_matrix(-self.position.x, -self.position.y, -self.position.z)
        self.view_matrix = rotation @ translation

    def increment_tilt(self, delta_x, delta_y, delta_z):
        self.tilt_x += delta_x
        self.tilt_y += delta_y
        self.tilt_z += delta_z
        self.update_view_matrix()

    def increment_position(self, dx, dy, dz):
        self.position.x += dx
        self.position.y += dy
        self.position.z += dz
        self.update_view_matrix()

    # Проекция точки на экран
    def project_vertex(self, vertex):
        transformed = self.view_matrix.transform_vector(vertex)
        if self.projection == 'orthographic':
            # Ортографическая проекция с учетом screen_distance как масштаб
            x = transformed.x / (1 - transformed.z/self.screen_distance)
            y = transformed.y / (1 - transformed.z/self.screen_distance)
            return (x, y)
        else:
            raise ValueError("Unsupported projection type.")

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
                x, y, z = map(float, parts[:3])
                self.vertices.append(Vector3D(x, y, z))

    def load_faces(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                indices = tuple(map(int, line.strip().split()))
                self.faces.append(indices)

    def normalize_vertices(self):
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
        window = tk.Tk()
        window.title("3D Model Visualization")
        canvas_width = 800
        canvas_height = 600
        canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
        canvas.pack()

        center_x = canvas_width // 2
        center_y = canvas_height // 2

        unique_edges = set()
        for face in self.model.faces:
            num_vertices = len(face)
            for i in range(num_vertices):
                v1 = face[i]
                v2 = face[(i + 1) % num_vertices]
                edge = tuple(sorted((v1, v2)))
                unique_edges.add(edge)

        def draw_edges():
            for edge in unique_edges:
                v1, v2 = edge
                vertex1 = self.model.vertices[v1]
                vertex2 = self.model.vertices[v2]
                screen1 = self.camera.project_vertex(vertex1)
                screen2 = self.camera.project_vertex(vertex2)
                if screen1 and screen2:
                    scale = 300  # Масштабирование
                    x1 = center_x + screen1[0] * scale
                    y1 = center_y - screen1[1] * scale  # Инвертируем Y для экранных координат
                    x2 = center_x + screen2[0] * scale
                    y2 = center_y - screen2[1] * scale
                    canvas.create_line(x1, y1, x2, y2, fill="black")

        draw_edges()

        def update_scene():
            canvas.delete("all")
            draw_edges()

        # Интерфейс для изменения наклона камеры и положения
        control_frame = tk.Frame(window)
        control_frame.pack(pady=10)

        # Группируем в два столбца
        left_frame = tk.Frame(control_frame)
        left_frame.grid(row=0, column=0, padx=10)

        right_frame = tk.Frame(control_frame)
        right_frame.grid(row=0, column=1, padx=10)

        # Наклоны
        tk.Label(left_frame, text="Наклон по X (Δ):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        tilt_x_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_x_var, width=5).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(left_frame, text="Наклон по Y (Δ):").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        tilt_y_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_y_var, width=5).grid(row=1, column=1, padx=5, pady=2)

        tk.Label(left_frame, text="Наклон по Z (Δ):").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        tilt_z_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_z_var, width=5).grid(row=2, column=1, padx=5, pady=2)

        # Позиция камеры
        tk.Label(right_frame, text="Перемещение камеры по X (Δ):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        move_x_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_x_var, width=5).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(right_frame, text="Перемещение камеры по Y (Δ):").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        move_y_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_y_var, width=5).grid(row=1, column=1, padx=5, pady=2)

        tk.Label(right_frame, text="Перемещение камеры по Z (Δ):").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        move_z_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_z_var, width=5).grid(row=2, column=1, padx=5, pady=2)

        # Изменение расстояния до экрана
        tk.Label(right_frame, text="Изменение расстояния до экрана (Δ):").grid(row=3, column=0, padx=5, pady=2,
                                                                               sticky='w')
        screen_dist_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=screen_dist_var, width=5).grid(row=3, column=1, padx=5, pady=2)

        def apply_changes():
            # Применение изменений наклона
            delta_x = tilt_x_var.get()
            delta_y = tilt_y_var.get()
            delta_z = tilt_z_var.get()
            self.camera.increment_tilt(delta_x, delta_y, delta_z)

            # Применение изменений позиции
            dx = move_x_var.get()
            dy = move_y_var.get()
            dz = move_z_var.get()
            self.camera.increment_position(dx, dy, dz)

            # Применение изменения расстояния до экрана
            dsd = screen_dist_var.get()
            self.camera.screen_distance += dsd

            update_scene()

        apply_button = tk.Button(control_frame, text="Применить изменения", command=apply_changes)
        apply_button.grid(row=1, column=0, columnspan=2, pady=10)

        window.mainloop()


if __name__ == "__main__":
    # Создание файлов модели (пирамида)
    with open("vertices_pyramide.txt", "w") as f:
        f.write("""0 1 0
-1 -1 -1
1 -1 -1
1 -1 1
-1 -1 1
""")

    with open("faces_pyramide.txt", "w") as f:
        f.write("""0 1 2
0 2 3
0 3 4
0 4 1
1 2 3 4
""")

    # Загрузка модели
    model = Model("vertices_pyramide.txt", "faces_pyramide.txt")

    # Параметры камеры
    camera_position = Vector3D(0, 0, 0)
    camera = Camera(
        position=camera_position,
        tilt_x=0,
        tilt_y=0,
        tilt_z=0,
        screen_distance=100,
        projection='orthographic'
    )

    # Создание и отображение сцены
    scene = Scene(model, camera)
    scene.render()
