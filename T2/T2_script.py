#""" To add a new cell, type ''
# To add a new markdown cell, type ''

"""
Plano refatoração:
 - Alterar nome das variaveis
 - Refatorar Classe 'Objetoss'
"""

# Computação Gráfica - Trabalho 2

# Aluno:                          NUSP:
# Michelle Wingter da Silva       10783243
# Juliano Fantozzi                9791218
# Luís Filipe Vasconcelos Peres   10310641

# Primeiro, vamos importar as bibliotecas necessárias.
# Verifique no código anterior um script para instalar as dependências necessárias (OpenGL e GLFW) antes de prosseguir.

import glm
import glfw
import math
import random
import threading
import numpy as np
from PIL import Image
from time import sleep
from OpenGL.GL import *
import OpenGL.GL.shaders


# Inicializando janela e váriaveis


# Configurações globais
W = 1200      # largura da janela
H = 1600      # altura da janela

titulo = "T2"


# Inicializando janela


glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(W, H, titulo, None, None)
glfw.make_context_current(window)


# GLSL para Vertex Shader
# 
# No Pipeline programável, podemos interagir com Vertex Shaders.
# 
# No código abaixo, estamos fazendo o seguinte:
# 
#  Definindo uma variável chamada position do tipo vec3.
#  Definindo matrizes Model, View e Projection que acumulam transformações geométricas 3D e permitem navegação no cenário.
#  void main() é o ponto de entrada do nosso programa (função principal)
#  gl_Position é uma variável especial do GLSL. Variáveis que começam com 'gl_' são desse tipo. Nesse caso, determina a posição de um vértice. Observe que todo vértice tem 4 coordenadas, por isso nós combinamos nossa variável vec2 com uma variável vec4. Além disso, nós modificamos nosso vetor com base nas transformações Model, View e Projection.


vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        varying vec2 out_texture;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
        }
        """


# GLSL para Fragment Shader
# 
# No Pipeline programável, podemos interagir com Fragment Shaders.
# 
# No código abaixo, estamos fazendo o seguinte:
# 
#  void main() é o ponto de entrada do nosso programa (função principal)
#  gl_FragColor é uma variável especial do GLSL. Variáveis que começam com 'gl_' são desse tipo. Nesse caso, determina a cor de um fragmento. Nesse caso é um ponto, mas poderia ser outro objeto (ponto, linha, triangulos, etc).

# Possibilitando modificar a cor.
# 
# Nos exemplos anteriores, a variável gl_FragColor estava definida de forma fixa (com cor R=0, G=0, B=0).
# 
# Agora, nós vamos criar uma variável do tipo "uniform", de quatro posições (vec4), para receber o dado de cor do nosso programa rodando em CPU.


fragment_code = """
        uniform vec4 color;
        varying vec2 out_texture;
        uniform sampler2D samplerTexture;
        
        void main(){
            vec4 texture = texture2D(samplerTexture, out_texture);
            //gl_FragColor = vec4(1.0/4, 1.0/2, 1.0, 1.0);
            gl_FragColor = texture;
        }
        """


# Requisitando slot para a GPU para nossos programas Vertex e Fragment Shaders


# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)


# Associando nosso código-fonte aos slots solicitados


# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)


# Compilando o Vertex Shader
# 
# Se há algum erro em nosso programa Vertex Shader, nosso app para por aqui.


# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")


# Compilando o Fragment Shader
# 
# Se há algum erro em nosso programa Fragment Shader, nosso app para por aqui.


glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")


# Associando os programas compilado ao programa principal


# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)


# Linkagem do programa


# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)


# Preparando dados para enviar a GPU
# 
# Nesse momento, nós compilamos nossos Vertex e Program Shaders para que a GPU possa processá-los.
# 
# Por outro lado, as informações de vértices geralmente estão na CPU e devem ser transmitidas para a GPU.
# 


glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_LINE_SMOOTH)
glEnable(GL_TEXTURE_2D)

qtd_texturas = 20
textures = glGenTextures(qtd_texturas)


# Classe dos objetos desenhados
# 
# Para facilitar o desenho dos objetos, faremos uma classe que representa qualquer objeto que se deseja desenhar na tela. Dessa forma, não é necessário criar uma função exclusiva para cada um desses objetos - economizando linhas de código. Além disso, com essa classe fica mais fácil manipular os objetos.
OBJECT_PATH = "objects/%s/vertex.obj"
TEXTURE_PATH = "objects/%s/texture_%s.png"

class Objetoss:

    _currentTextureIndex = 0    # conta os ids das texturas
    _points = None              # armazena todos os pontos de todos os vertices
    _textures = None            # armazena todos os pontos de todas as texturas

    def __init__(self, objName = None, 
            x = 0.0, y = 0.0, z = 0.0, 
            rot_x = 0.0, rot_y = 0.0, rot_z = 0.0, 
            sx = 1.0, sy = 1.0, sz = 1.0, 
            visible = True, ntextures = 1):
        #  Construtor de um objeto 3D que será desenhado na tela. 

        self._thread = None 
        self._threadStop = False

        # Definir posição inicial do objeto no mundo.
        self.x = x
        self.y = y
        self.z = z

        # Definir rotação inicial do objeto (em radianos).
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

        # Definir escala inicial do objeto.
        self.sx = sx
        self.sy = sy
        self.sz = sz

        # Definir objeto como visível.
        self.visible = visible

        # Definição do índice de textura desse objeto, início dos vértices e término.
        self._textureIndex = None
        self._vertexIndex = None
        self._vertexLength = None

        if objName is not None:
            print("Carregando pontos do objeto %s...\n" % (objName.upper()), end = "")
            self.setVertex(OBJECT_PATH % (objName))

            print("Carregando textura... ", end = "")
            for cont in range (0, ntextures):
                self.load_texture_from_file(TEXTURE_PATH % (objName, str(cont)))            
            
           

    def setVertex(self, file_path):
      
        if self._vertexIndex is not None or self._vertexLength is not None:
            raise Exception("Objeto já existente")

        vertices = [ ]
        
        texture_coords = [ ]
        faces = [ ]
        material = None

       
        with open(file_path, "r") as f:
            for line in f: 
                if not line[0] == '#':
                
                    points = line.split() 
                    if not points:
                        pass
                    elif points[0] == 'v': 
                        vertices.append(points[1:4])
                    elif points[0] == 'vt': 
                        texture_coords.append(points[1:3])
                    elif points[0] in ('usemtl', 'usemat'): 
                        material = points[1]
                    elif points[0] == 'f' and material is not None: 
                        face = []
                        face_texture = []
                        for v in points[1:]:
                            w = v.split('/')
                            face.append(int(w[0]))
                            if len(w) >= 2 and len(w[1]) > 0:
                                face_texture.append(int(w[1]))
                            else:
                                face_texture.append(0)

                        faces.append((face, face_texture, material))

        model = {
            'faces': faces,
            'vertices': vertices,
            'texture': texture_coords
        }

        if Objetoss._points is None or Objetoss._textures is None:
            Objetoss._points = [ ]
            Objetoss._textures = [ ]

        self._vertexIndex = len(Objetoss._points)
        faces_visited = []
        for face in model['faces']:
            if face[2] not in faces_visited:
                print(face[2], "vertice inicial = ", len(Objetoss._points))
                faces_visited.append(face[2])
            for vertice_id in face[0]:                
                Objetoss._points.append(model['vertices'][vertice_id - 1])
            for texture_id in face[1]:
                Objetoss._textures.append(model['texture'][texture_id - 1])
            
        self._vertexLength = len(Objetoss._points) - self._vertexIndex
       
        
        
        

    def load_texture_from_file(self, filePath):
        #  Essa função vai definir a textura deste objeto no mundo usando um arquivo em disco. 
        if self._textureIndex is None:
            self._textureIndex = Objetoss._currentTextureIndex
            Objetoss._currentTextureIndex += 1
        img = Image.open(filePath)
        img_width = img.size[0]
        img_height = img.size[1]
        image_data = img.convert("RGBA").tobytes("raw", "RGBA", 0, -1)
        glBindTexture(GL_TEXTURE_2D, self._textureIndex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        
        

        
    
    @staticmethod
    def syncGPU():
        #  Essa função sincroniza os dados dos vértices e coordenadas de textura com a GPU. Basicamente, ela envia os dados pra GPU. 

        if Objetoss._points is None or Objetoss._textures is None:
            raise Exception("Objeto não foi definido")

        buffer = glGenBuffers(2) # Requisitar slots de buffer pra GPU.

        # Definir vértices.
        vertices = np.zeros(len(Objetoss._points), [("position", np.float32, 3)])
        vertices['position'] = Objetoss._points

        # Carregar vértices.
        glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        stride = vertices.strides[0]
        offset = ctypes.c_void_p(0)
        loc_vertices = glGetAttribLocation(program, "position")
        glEnableVertexAttribArray(loc_vertices)
        glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

        # Definir coordenadas de texturas.
        textures = np.zeros(len(Objetoss._textures), [("position", np.float32, 2)]) # Duas coordenadas.
        textures['position'] = Objetoss._textures

        # Carregar coordenadas de texturas.
        glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
        glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
        stride = textures.strides[0]
        offset = ctypes.c_void_p(0)
        loc_texture_coord = glGetAttribLocation(program, "texture_coord")
        glEnableVertexAttribArray(loc_texture_coord)
        glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)

        
        
    # aplica a matriz model
    def draw(self):        

        if self._textureIndex is None or self._vertexIndex is None or self._vertexLength is None:
            raise Exception("objeto tentou ser carregado, mas o obj não foi processado")

        if not self.visible: # Se objeto não estiver visível, não precisa desenhar.
            return
    
        matrix_transform = glm.mat4(1.0) # Instanciando uma matriz identidade.
        
        # Aplicando translação em X, Y e Z.
        matrix_transform = glm.translate(matrix_transform, glm.vec3(self.x, self.y, self.z))

        # Aplicando rotação no eixo X.
        matrix_transform = glm.rotate(matrix_transform, self.rot_x, glm.vec3(1.0, 0.0, 0.0))

        # Aplicando rotação no eixo Y.
        matrix_transform = glm.rotate(matrix_transform, self.rot_y, glm.vec3(0.0, 1.0, 0.0))

        # Aplicando rotação no eixo Z.
        matrix_transform = glm.rotate(matrix_transform, self.rot_z, glm.vec3(0.0, 0.0, 1.0))
        
        # Aplicando escala em X, Y e Z.
        matrix_transform = glm.scale(matrix_transform, glm.vec3(self.sx, self.sy, self.sz))
        
        matrix_transform = np.array(matrix_transform).T

        loc_model = glGetUniformLocation(program, "model")
        glUniformMatrix4fv(loc_model, 1, GL_TRUE, matrix_transform)
        
        # Define o id da textura do modelo.
        glBindTexture(GL_TEXTURE_2D, self._textureIndex)
        
        # Desenha o modelo.
        glDrawArrays(GL_TRIANGLES, self._vertexIndex, self._vertexLength) 

    def _alive():
         pass 
    def spawn(self):
        
        if self._thread is not None:
            raise Exception("O programa tentou carregar duas vezes o mesmo objeto")
        
        self._thread = threading.Thread(target = self.__threadLoop)
        self._threadStop = False
        self._thread.start()
        return self
    
    def kill(self):
        
        if self._thread is None:
            raise Exception("O programa tentou matar um objeto que não estava vivo")

        self._threadStop = True
        self._thread.join()
        self._thread = None
    
    def isAlive(self):
        if self._thread is None:
            return False
        else:
            return True

    def __threadLoop(self):    
        while(not self._threadStop):
            self._alive()


# Criando os objetos imóveis, carregar os vértices e texturas.


Objs = [ ]
Objs.append( Objetoss(objName = "mountains", x = 18.0, rot_y = math.pi/2) )
Objs.append( Objetoss(objName = "mountains", x = -18.0, rot_y = math.pi/2) )
Objs.append( Objetoss(objName = "mountains", z = 18.0) )
Objs.append( Objetoss(objName = "mountains", z = -18.0) )
Objs.append( Objetoss(objName = "ground", y = -0.9, sx = 20.0, sz = 20.0) )
Objs.append( Objetoss(objName = "chair", y = 0.1, x = 7.8, z=0.4, sx = 2, sz = 2, sy = 2,rot_y = math.pi) ) 
Objs.append( Objetoss(objName = "notebook", y = 0.427, x = 8, sx = 1, sz = 1, sy = 1) )
Objs.append( Objetoss(objName = "barn", y = 0.1, x = 8, sx = 1, sz = 1, sy = 1) )
Objs.append( Objetoss(objName = "house", y = 0.3, x = -5, sx = 3, sz = 3, sy = 3, rot_y = math.pi/2) )
Objs.append( Objetoss(objName = "office", y = 0.1, x = 8, sx = 2, sz = 2, sy = 2) )
Objs.append( Objetoss(objName = "mill", y = -0.2, x = 12, z = 6, sx = 0.5, sz = 0.5, sy = 0.5) ) 
Objs.append( Objetoss(objName = "street", y = -0.89, x = 0, z = 0, sx = 3/2, sz = 18) )
Objs.append( Objetoss(objName = "horse", y = 0.1, x = 10, z = 5, sx = 5, sy = 5, sz = 5) )
Objs.append( Objetoss(objName = "cow", y = 0.1, x = 7, z = 6.3, sx = 0.35, sy = 0.35, sz = 0.35, rot_y = -math.pi/2) )
Objs.append( Objetoss(objName = "rainbow", y = 0.4, x = 8, sx = 10, sz = 10, sy = 10) )
Objs.append( Objetoss(objName = "fire", y = 0.1, x = 6.7, sx = 1, sz = 1, sy = 1, rot_y = math.pi/2) ) 


# Objetos com animação
# 


ObjsAnimados = [ ]    

class Marte(Objetoss): # A lua vai nascendo e se pondo no horizonte aos poucos.
    def _alive(self):
        self.y += 0.05
        self.rot_z = math.pi/6
        if self.y > 100.0:
            self.y = -20
        sleep(0.1)
        
class Ufo(Objetoss): # A lua vai nascendo e se pondo no horizonte aos poucos.
    def _alive(self):
        self.x += 0.05
        if self.x > 100.0:
            self.x = -20
        sleep(0.1)

class nuvens(Objetoss): # Para mover as nuvens no céu 
    def _alive(self):
        self.rot_y = (self.rot_y + 0.0005) % (2 * math.pi)
        sleep(0.1)

class Car(Objetoss): 
    def _alive(self):
        self.z += 0.05
        if self.z > 15.0:
            self.z = -15
        sleep(0.1)
        
ObjsAnimados.append( Car(objName = "car", x = 0.5, y = 0.35, z = 0, sx = 60, sz = 60, sy = 60) )
ObjsAnimados.append( nuvens(objName = "nuvens", sx = 100.0, sy = 100.0, sz = 100.0) )
ObjsAnimados.append( Marte(objName = "mars", x = 45, y = 50, z = 40, sx = 18.0, sy = 18.0, sz = 18.0) )
ObjsAnimados.append( Ufo(objName = "ufo", x = 1.5, y = 10, z = 10) )


# Enviar dados para a GPU.
# 
# Nossa classe já possue um método estático que faz isso (requisitar slots de buffer -> enviar vértices -> enviar coordenadas de texturas). Basta chamar esse método.


Objetoss.syncGPU()


# Eventos para modificar a posição da câmera.
# 
#  Usei as teclas A, S, D e W para movimentação no espaço tridimensional
#  Usei a posição do mouse para "direcionar" a câmera
# 
#  O mouse é capturado pela janela, e podemos soltar preccionando a tecla enter ou ESQ
#  Podemos liberar a camera pressionando a tecla c


CAMERA_SPEED = 0.2
CAMERA_Y = 0.7
CAMERA_X_MIN = -15
CAMERA_X_MAX = 15
CAMERA_Z_MIN = -15
CAMERA_Z_MAX = 15

polygonal_mode = False
free_camera = False
paused = False

W_pressed = False
S_pressed = False
A_pressed = False
D_pressed = False

cameraPos   = glm.vec3(0.0,  CAMERA_Y,  1.0)
cameraFront = glm.vec3(0.0,  0.0, -1.0)
cameraUp    = glm.vec3(0.0,  1.0,  0.0)

class Camera(Objetoss):
    def _alive(self):
        global cameraPos, cameraFront, cameraUp, paused, CAMERA_SPEED
        
        if paused:
            sleep(1.0)
            return

        if free_camera: # No modo de câmera livre, a velocidade é maior
            camera_speed = 4 * CAMERA_SPEED
        else:
            camera_speed = CAMERA_SPEED

        if W_pressed:
            cameraPos += cameraFront * camera_speed

        if S_pressed:
            cameraPos -= cameraFront * camera_speed

        if A_pressed:
            cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * camera_speed

        if D_pressed:
            cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * camera_speed

        if not free_camera:
            cameraPos[1] = CAMERA_Y
            cameraPos[0] = max(CAMERA_X_MIN, min(CAMERA_X_MAX, cameraPos[0]))
            cameraPos[2] = max(CAMERA_Z_MIN, min(CAMERA_Z_MAX, cameraPos[2]))

        sleep(0.05)


def key_event(window,key,scancode,action,mods):
    global polygonal_mode, paused, free_camera, W_pressed, S_pressed, A_pressed, D_pressed
    
    cameraSpeed = 0.2
    if key == 87: # Tecla W.
        W_pressed = False if (action == 0) else True
    
    elif key == 83: # Tecla S.
        S_pressed = False if (action == 0) else True
    
    elif key == 65: # Tecla A.
        A_pressed = False if (action == 0) else True
        
    elif key == 68: # Tecla D.
        D_pressed = False if (action == 0) else True

    elif key == 80 and action == 0: # Botão P (modo polígono).
        polygonal_mode = not polygonal_mode

    elif key == 67 and action == 0: # Botão C (modo de câmera: bloqueado ou livre).
        free_camera = not free_camera
    
    elif (key == 257 or key == 256) and action == 0: # Botão ENTER ou ESC (câmera lock).
        paused = not paused
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL if paused else glfw.CURSOR_HIDDEN)

        
yaw = -90.0 
pitch = 0.0
lastX =  W/2
lastY =  H/2

def mouse_event(window, xpos, ypos):
    global paused, firstMouse, cameraFront, yaw, pitch, lastX, lastY

    if paused:
        return

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    # lastX = xpos
    # lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset

    
    if pitch >= 80.0: pitch = 80.0
    if pitch <= -80.0: pitch = -80.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)

    glfw.set_cursor_pos(window, lastX, lastY)

camera = Camera()
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)


# Matrizes Model, View e Projection


def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
       
    # aplicando rotacao
    if angle!=0:
        matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
  
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp)
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global W, W
    mat_projection = glm.perspective(glm.radians(90.0), W/W, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection


# Nesse momento, nós exibimos a janela!
# 


glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)


# Loop principal da janela.
# Enquanto a janela não for fechada, esse laço será executado. É neste espaço que trabalhamos com algumas interações com a OpenGL.


glEnable(GL_DEPTH_TEST)  #importante para 3D

for obj in ObjsAnimados:
    obj.spawn()
camera.spawn()

while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    #glClearColor(1.0/4,1.0/4,1.0,1.0)
    glClearColor(255/255,230/255,204/255,1)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    elif polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)

    for obj in Objs:
        obj.draw()

    for obj in ObjsAnimados:
        obj.draw()
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)
    
    glfw.swap_buffers(window)

for obj in ObjsAnimados: # Tirar vida dos objetos dinâmicos.
    obj.kill()
camera.kill()

glfw.terminate()










