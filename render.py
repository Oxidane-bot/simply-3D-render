import cv2
import numpy as np
from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import copy
from math import tan, cos, sin, sqrt, pi
import time

width = 600
height = 800
num_channels = 4
image = Image(width, height, Color(255, 255, 255, 255))


# Camera settings
camera_position = Vector(0, 0, 3)
fov = 90
aspect_ratio = float(width) / float(height)
near_plane = 2
far_plane = 10

# target_position = Vector(-1, -1, 2)

# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model


model = Model('data/headset.obj')
model.normalizeGeometry()


def correctDistortion(point, c1, c2):
    x = point.x - width / 2
    y = point.y - height / 2
    r = sqrt(x**2 + y**2) / (width / 2)  
    ru = r + (c1*r**2 + c2*r**4 + c2**2*r**4 + c2**2*r**8 + 2*c1*c2*r**6) / (1 + 4*c1*r**2 + 6*c2*r**4)
    factor = ru / r if r != 0 else 1
    x *= factor
    y *= factor
    return Point(round(x + width / 2), round(y + height / 2), point.z, point.color)

def calculate_sphere(model):
    center = Vector(0, 0, 0)
    for vertex in model.vertices:
        center += vertex
    center /= len(model.vertices)

    radius = max((vertex - center).length() for vertex in model.vertices)

    return center, radius

def is_collision(center1, radius1, center2, radius2):
    distance = (center2 - center1).length()
    return distance < (radius1 + radius2)


def isWithinBounds(point, width, height):
    return 0 <= point.x < width and 0 <= point.y < height

def lookAt(camera_position, target_position, up_direction):
    # Compute forward, right and up vectors
    forward = (target_position - camera_position).normalize()
    right = up_direction.cross(forward).normalize()
    up = forward.cross(right)

    # Construct lookAt matrix
    lookAtMatrix = [
        [right.x, right.y, right.z, -camera_position.dot(right)],
        [up.x, up.y, up.z, -camera_position.dot(up)],
        [-forward.x, -forward.y, -forward.z, camera_position.dot(forward)],
        [0, 0, 0, 1]
    ]

    return lookAtMatrix


def getPerspectiveProjection(x, y, z):
    # Convert vertex from world space to screen space using perspective projection
    x, y, z = x - camera_position.x, y - camera_position.y, z - camera_position.z
    d = 1 / (near_plane * tan(fov * 0.5))
    proj_matrix = [
        [d / aspect_ratio, 0, 0, 0],
        [0, d, 0, 0],
        [0, 0, (far_plane + near_plane) / (near_plane - far_plane),
                2 * far_plane * near_plane / (near_plane - far_plane)],
        [0, 0, -1, 0]
    ]

    vertex = Vector(x, y, z, 1)
    vertex_clip = Vector(
        vertex.dot(proj_matrix[0]),
        vertex.dot(proj_matrix[1]),
        vertex.dot(proj_matrix[2]),
        vertex.dot(proj_matrix[3])
    )
    # Perspective divide
    if vertex_clip.w != 0:
        vertex_ndc = vertex_clip / vertex_clip.w
    else:
        vertex_ndc = vertex_clip

    screenX = int((vertex_ndc.x + 1.0) * width / 2.0)
    screenY = int((vertex_ndc.y + 1.0) * height / 2.0)

    return screenX, screenY



##


def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_x(angle):
    c = cos(angle)
    s = sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_y(angle):
    c = cos(angle)
    s = sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_z(angle):
    c = cos(angle)
    s = sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def transform_vector(v, matrix):
    v_np = np.array([v.x, v.y, v.z, 1])
    v_transformed = np.dot(matrix, v_np)
    return Vector(v_transformed[0], v_transformed[1], v_transformed[2])

###


def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])


def init_image_and_zbuffer():
    global image, zBuffer
    image = Image(width, height, Color(255, 255, 255, 255))
    zBuffer = [-float('inf')] * width * height


def rendering():
    c1 =  0.5
    c2 = 1
    global angle_x
    global angle_y
    global angle_z
    global translation
    global scaling
	# Define the light direction
    lightDir = Vector(0,0,1)    
    # Calculate rotation matrix
    rotation_x = rotation_matrix_x(angle_x)
    rotation_y = rotation_matrix_y(angle_y)
    rotation_z = rotation_matrix_z(angle_z)
    translation_m = translation_matrix(*translation)
    scaling_m = scaling_matrix(*scaling)

    model = copy.deepcopy(original_model)
    
	# Calculate face normals
    faceNormals = {}
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        # Calculate two possible face normals
        faceNormal1 = (p2-p0).cross(p1-p0).normalize()
        faceNormal2 = (p1-p0).cross(p2-p0).normalize()

        # Choose the one that is closer to the light direction
        if faceNormal1 * lightDir > faceNormal2 * lightDir:
            faceNormal = faceNormal1
        else:
            faceNormal = faceNormal2

        for i in face:
            if not i in faceNormals:
                faceNormals[i] = []

            faceNormals[i].append(faceNormal)
    
	# Calculate vertex normals
    vertexNormals = []
    for vertIndex in range(len(model.vertices)):
        vertNorm = getVertexNormal(vertIndex, faceNormals)
        vertexNormals.append(vertNorm)

    for i in range(len(model.vertices)):
        # Apply the transformations to each vertex
        
        model.vertices[i] = transform_vector(model.vertices[i], rotation_x)
        model.vertices[i] = transform_vector(model.vertices[i], rotation_y)
        model.vertices[i] = transform_vector(model.vertices[i], rotation_z)
        model.vertices[i] = transform_vector(model.vertices[i], scaling_m)
        model.vertices[i] = transform_vector(model.vertices[i], translation_m)


    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        n0, n1, n2 = [vertexNormals[i] for i in face]


        cull = False

        transformedPoints = []
        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
            intensity = n * lightDir

            if intensity < 0:
                cull = True
                break

            screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z)
            correctedPoint = correctDistortion(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)), c1, c2)

            if isWithinBounds(correctedPoint, width, height):
                transformedPoints.append(correctedPoint)

        if not cull:
            if len(transformedPoints) == 3:  # Add this line
                Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)

    




# change per render
angle_x = 0
angle_y = 0
angle_z = 0
translation = [0,0,0]
scaling = (1,1,1)


g= 9.8  
drag_coefficient = 0.5  
air_density = 1.3  
area = 0.2  
time_step = 0.1
prev_time = time.time()

position = [0,0,0]


while True:
    angle_y += 0.5
    angle_x += 0.5

    gravity_force = [0, 0, -g]
    drag_force = [-0.5 * drag_coefficient * air_density * area * abs(velocity) * velocity for velocity in position]
    net_force = [gravity + drag for gravity, drag in zip(gravity_force, drag_force)]

    position = [pos + velocity * time_step + 0.5 * force * time_step**2 for pos, velocity, force in zip(position, position, net_force)]
    translation =copy.deepcopy(position)
    # Update the position of the object
    

    
    #init the image and render
    init_image_and_zbuffer()
    original_model = copy.deepcopy(model)
    rendering()
    # image.saveAsPNG('1.png')
    # reformat image buffer
    byte_array = bytearray(image.buffer)
    flat_array = [byte_array[i] for i in range(len(byte_array)) if i % ((num_channels * width) + 1) != 0]
    image_data = np.array(flat_array, dtype=np.uint8).reshape((height, width, num_channels))
    bgr_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
    
    # Display the image and wait for a key press
    key = cv2.waitKey(35)

    # Check if the user has pressed the 'q' key to quit
    if key & 0xFF == ord('q'):
        break
    #clear page
    cv2.imshow("Image", bgr_image)

    curr_time = time.time()
    delta_time = curr_time - prev_time
    prev_time = curr_time
    # Check if it's time to update the image
    if delta_time >= 1000:
        # Reset the timer
        prev_time = curr_time

# Close all windows and exit
cv2.destroyAllWindows()