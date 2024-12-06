import tkinter as tk
from math import cos, sin, radians, sqrt, atan2, degrees

# Класс для трехмерных векторов
class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_list(self):
        return [self.x, self.y, self.z, 1]

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        # Скалярное произведение
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        # Векторное произведение
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)

# Класс для 4x4 матриц
class Matrix4x4:
    def __init__(self, data):
        self.data = data  # data - двумерный список 4x4

    def __matmul__(self, other):
        result = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.data[i][k]*other.data[k][j]
        return Matrix4x4(result)

    def transform_vector(self, vec):
        res = [0]*4
        vec_list = vec.to_list()
        for i in range(4):
            for j in range(4):
                res[i] += self.data[i][j]*vec_list[j]
        if res[3] != 0:
            return Vector3D(res[0]/res[3], res[1]/res[3], res[2]/res[3])
        else:
            return Vector3D(res[0], res[1], res[2])

def translation_matrix(dx, dy, dz):
    return Matrix4x4([
        [1,0,0,dx],
        [0,1,0,dy],
        [0,0,1,dz],
        [0,0,0,1]
    ])

def rotation_matrix_x(angle_deg):
    angle = radians(angle_deg)
    return Matrix4x4([
        [1,          0,           0,          0],
        [0, cos(angle), -sin(angle), 0],
        [0, sin(angle),  cos(angle), 0],
        [0,          0,           0,          1]
    ])

def rotation_matrix_y(angle_deg):
    angle = radians(angle_deg)
    return Matrix4x4([
        [ cos(angle), 0, sin(angle), 0],
        [0,           1, 0,          0],
        [-sin(angle),0, cos(angle),  0],
        [0,           0, 0,          1]
    ])

def rotation_matrix_z(angle_deg):
    angle = radians(angle_deg)
    return Matrix4x4([
        [cos(angle), -sin(angle), 0, 0],
        [sin(angle),  cos(angle), 0, 0],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])

# Класс камеры
class Camera:
    def __init__(self, position, tilt_x=0, tilt_y=0, tilt_z=0, screen_distance=1, projection='orthographic'):
        self.position = position
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.tilt_z = tilt_z
        self.screen_distance = screen_distance
        self.projection = projection
        self.update_view_matrix()

    def update_view_matrix(self):
        rx = rotation_matrix_x(self.tilt_x)
        ry = rotation_matrix_y(self.tilt_y)
        rz = rotation_matrix_z(self.tilt_z)
        rotation = rz @ ry @ rx
        translation = translation_matrix(-self.position.x, -self.position.y, -self.position.z)
        self.view_matrix = rotation @ translation

    def increment_tilt(self, dx, dy, dz):
        self.tilt_x += dx
        self.tilt_y += dy
        self.tilt_z += dz
        self.update_view_matrix()

    def increment_position(self, dx, dy, dz):
        self.position.x += dx
        self.position.y += dy
        self.position.z += dz
        self.update_view_matrix()

    def project_vertex(self, vertex):
        transformed = self.view_matrix.transform_vector(vertex)
        if self.projection == 'orthographic':
            x = transformed.x/(1 - transformed.z/self.screen_distance)
            y = transformed.y/(1 - transformed.z/self.screen_distance)
            return (x, y)
        else:
            raise ValueError("Unsupported projection type.")

# Класс модели
class Model:
    def __init__(self, vertices_file, faces_file):
        self.vertices = []
        self.faces = []
        self.load_vertices(vertices_file)
        self.load_faces(faces_file)
        self.normalize_vertices()

    def load_vertices(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)<3:
                    continue
                x,y,z = map(float,parts[:3])
                self.vertices.append(Vector3D(x,y,z))

    def load_faces(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                indices = tuple(map(int,line.strip().split()))
                self.faces.append(indices)

    def normalize_vertices(self):
        max_coord = max(max(abs(v.x), abs(v.y), abs(v.z)) for v in self.vertices)
        if max_coord!=0:
            for v in self.vertices:
                v.x/=max_coord
                v.y/=max_coord
                v.z/=max_coord

    def apply_transformation(self, M):
        new_vertices = []
        for v in self.vertices:
            nv = M.transform_vector(v)
            new_vertices.append(nv)
        self.vertices = new_vertices

    def get_bottom_plane_equation(self):
        bottom_face = self.faces[-1]
        v1 = self.vertices[bottom_face[0]]
        v2 = self.vertices[bottom_face[1]]
        v3 = self.vertices[bottom_face[2]]
        n = (v2 - v1).cross(v3 - v1)
        A,B,C = n.x,n.y,n.z
        D = -(A*v1.x + B*v1.y + C*v1.z)
        return A,B,C,D

def reflect_model_stepwise(model):
    # 1. Получаем уравнение плоскости нижней грани пирамиды
    # Уравнение: Ax + By + Cz + D = 0
    A, B, C, D = model.get_bottom_plane_equation()

    # Нормализуем нормаль
    norm_length = sqrt(A**2 + B**2 + C**2)
    if norm_length < 1e-14:
        return  # Дегенеративный случай
    A_n, B_n, C_n = A / norm_length, B / norm_length, C / norm_length

    # --------------------------
    # Вычисляем точку M0 на плоскости
    # Если C ≠ 0: x = 0, y = 0, z = -D / C
    if abs(C) > 1e-14:
        M0 = Vector3D(0, 0, -D / C)
    elif abs(B) > 1e-14:
        M0 = Vector3D(0, -D / B, 0)
    else:
        M0 = Vector3D(-D / A, 0, 0)

    # --------------------------
    # Матрица переноса на -M0
    T_neg = translation_matrix(-M0.x, -M0.y, -M0.z)

    # Применяем перенос к модели
    model.apply_transformation(T_neg)

    # --------------------------
    # ШАГ 1: Перевод нормали в плоскость OXY
    # Выполняем поворот вокруг оси OX на угол -ψ, чтобы нормаль оказалась в плоскости OXY.
    bc_length = sqrt(B_n**2 + C_n**2)  # Длина проекции нормали на плоскость YZ
    if bc_length < 1e-14:
        cos_psi, sin_psi = 1, 0  # Если нормаль уже в плоскости OXY
    else:
        cos_psi = B_n / bc_length
        sin_psi = C_n / bc_length

    # Матрица поворота вокруг OX на -ψ
    Rx_minus_psi = Matrix4x4([
        [1, 0,       0,        0],
        [0, cos_psi, sin_psi,  0],
        [0, -sin_psi, cos_psi, 0],
        [0, 0,       0,        1]
    ])

    # --------------------------
    # ШАГ 2: Перевод нормали в ось OX
    # Выполняем поворот вокруг оси OZ на угол -φ, чтобы нормаль совпала с осью OX.
    ab_length = sqrt(A_n**2 + B_n**2 + C_n**2)  # Длина проекции нормали на плоскость XY
    if ab_length < 1e-14:
        cos_phi, sin_phi = 1, 0  # Если нормаль уже вдоль оси OX
    else:
        cos_phi = A_n / ab_length
        sin_phi = sqrt(B_n**2+C_n**2) / ab_length

    # Матрица поворота вокруг OZ на -φ
    Rz_minus_phi = Matrix4x4([
        [cos_phi, sin_phi, 0, 0],
        [-sin_phi, cos_phi, 0, 0],
        [0,       0,       1, 0],
        [0,       0,       0, 1]
    ])

    # --------------------------
    # ШАГ 3: Отражение относительно плоскости YOZ
    reflect_yz = Matrix4x4([
        [-1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ])

    # --------------------------
    # ШАГ 4: Возврат из оси OX в исходное положение
    Rz_phi = Matrix4x4([
        [cos_phi, -sin_phi, 0, 0],
        [sin_phi,  cos_phi, 0, 0],
        [0,       0,        1, 0],
        [0,       0,        0, 1]
    ])

    # --------------------------
    # ШАГ 5: Возврат из плоскости OXY в исходное положение
    Rx_psi = Matrix4x4([
        [1, 0,        0,        0],
        [0, cos_psi, -sin_psi,  0],
        [0, sin_psi,  cos_psi,  0],
        [0, 0,        0,        1]
    ])

    # --------------------------
    # Итоговая матрица
    M_result = Rx_psi @ Rz_phi @ reflect_yz @ Rz_minus_phi @ Rx_minus_psi

    # Применяем итоговую матрицу к модели
    model.apply_transformation(M_result)

    # --------------------------
    # Возврат модели в исходное положение
    # Матрица переноса на +M0
    T_pos = translation_matrix(M0.x, M0.y, M0.z)

    # Применяем обратный перенос к модели
    model.apply_transformation(T_pos)



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
                v2 = face[(i+1)%num_vertices]
                edge = tuple(sorted((v1,v2)))
                unique_edges.add(edge)

        def draw_edges():
            for edge in unique_edges:
                v1,v2 = edge
                vertex1 = self.model.vertices[v1]
                vertex2 = self.model.vertices[v2]
                s1 = self.camera.project_vertex(vertex1)
                s2 = self.camera.project_vertex(vertex2)
                if s1 and s2:
                    scale=300
                    x1 = center_x+s1[0]*scale
                    y1 = center_y - s1[1]*scale
                    x2 = center_x+s2[0]*scale
                    y2 = center_y - s2[1]*scale
                    canvas.create_line(x1,y1,x2,y2,fill="black")

        draw_edges()

        def update_scene():
            canvas.delete("all")
            draw_edges()

        # Интерфейс
        control_frame = tk.Frame(window)
        control_frame.pack(pady=10)

        left_frame = tk.Frame(control_frame)
        left_frame.grid(row=0,column=0,padx=10)

        right_frame = tk.Frame(control_frame)
        right_frame.grid(row=0,column=1,padx=10)

        tk.Label(left_frame, text="Наклон по X (Δ):").grid(row=0,column=0, padx=5,pady=2,sticky='w')
        tilt_x_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_x_var, width=5).grid(row=0,column=1, padx=5,pady=2)

        tk.Label(left_frame, text="Наклон по Y (Δ):").grid(row=1,column=0, padx=5,pady=2,sticky='w')
        tilt_y_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_y_var, width=5).grid(row=1,column=1, padx=5,pady=2)

        tk.Label(left_frame, text="Наклон по Z (Δ):").grid(row=2,column=0, padx=5,pady=2,sticky='w')
        tilt_z_var = tk.DoubleVar(value=0)
        tk.Entry(left_frame, textvariable=tilt_z_var, width=5).grid(row=2,column=1, padx=5,pady=2)

        tk.Label(right_frame, text="Перемещение камеры по X (Δ):").grid(row=0,column=0, padx=5,pady=2,sticky='w')
        move_x_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_x_var, width=5).grid(row=0,column=1, padx=5,pady=2)

        tk.Label(right_frame, text="Перемещение камеры по Y (Δ):").grid(row=1,column=0, padx=5,pady=2,sticky='w')
        move_y_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_y_var, width=5).grid(row=1,column=1, padx=5,pady=2)

        tk.Label(right_frame, text="Перемещение камеры по Z (Δ):").grid(row=2,column=0, padx=5,pady=2,sticky='w')
        move_z_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=move_z_var, width=5).grid(row=2,column=1, padx=5,pady=2)

        tk.Label(right_frame, text="Изменение расстояния до экрана (Δ):").grid(row=3,column=0, padx=5,pady=2,sticky='w')
        screen_dist_var = tk.DoubleVar(value=0)
        tk.Entry(right_frame, textvariable=screen_dist_var, width=5).grid(row=3,column=1, padx=5,pady=2)

        def apply_changes():
            delta_x = tilt_x_var.get()
            delta_y = tilt_y_var.get()
            delta_z = tilt_z_var.get()
            self.camera.increment_tilt(delta_x,delta_y,delta_z)

            dx = move_x_var.get()
            dy = move_y_var.get()
            dz = move_z_var.get()
            self.camera.increment_position(dx,dy,dz)

            dsd = screen_dist_var.get()
            self.camera.screen_distance+=dsd
            update_scene()

        def reflect_model_button():
            reflect_model_stepwise(self.model)
            update_scene()

        apply_button = tk.Button(control_frame,text="Применить изменения",command=apply_changes)
        apply_button.grid(row=1,column=0,pady=10)

        reflect_button = tk.Button(control_frame,text="Отразить относительно нижней плоскости",command=reflect_model_button)
        reflect_button.grid(row=1,column=1,pady=10)

        window.mainloop()

if __name__=="__main__":
    with open("vertices_pyramide.txt","w") as f:
        f.write("""0 1 0
-1 -1 -1
1 -1 -1
1 -1 1
-1 -1 1
""")

    with open("faces_pyramide.txt","w") as f:
        f.write("""0 1 2
0 2 3
0 3 4
0 4 1
1 2 3 4
""")

    model = Model("vertices_pyramide.txt","faces_pyramide.txt")
    camera_position = Vector3D(0,0,0)
    camera = Camera(position=camera_position, tilt_x=0, tilt_y=0, tilt_z=0, screen_distance=100, projection='orthographic')
    scene = Scene(model,camera)
    scene.render()
