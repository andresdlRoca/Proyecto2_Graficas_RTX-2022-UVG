import numpy as np
import math
import mathlib as ml

WHITE = (1,1,1)
BLACK = (0,0,0)

OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2


class Intersect(object):
    def __init__(self, distance, point, normal, texcoords, sceneObj):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.texcoords = texcoords
        self.sceneObj = sceneObj

class Material(object):
    def __init__(self, diffuse = WHITE, spec = 1.0, ior = 1.0, texture = None, matType = OPAQUE):
        self.diffuse = diffuse
        self.spec = spec
        self.ior = ior
        self.texture = texture
        self.matType = matType


class Sphere(object):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):
        L = ml.subtract(self.center, orig)
        tca = ml.dot(L, dir)
        d = (ml.norm(L) ** 2 - tca ** 2) ** 0.5

        # print(f'd: {d}')
        # print(f'self.radius: {self.radius}')
        if isinstance(d, complex):
            d= d.real

        try:
            if d > self.radius:
                return None
        except:
            print(ml.norm(L))
            print(f'L: {L}')
            print(f'TCA: {tca}')
            print(f'd: {d}')
            print(f'self.radius: {self.radius}')

        thc = (self.radius ** 2 - d ** 2) ** 0.5

        if isinstance(thc, complex):
            thc = thc.real

        t0 = tca - thc
        t1 = tca + thc

        if isinstance(t0, complex):
            t0 = t0.real

        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return None
        
        # P = O + t0 * D
        t0pordir = [t0 * x for x in dir]
        # P = ml.add(orig, t0 * np.array(dir))
        P = ml.add(orig, t0pordir)
        normal = ml.subtract(P, self.center)
        normal = [x / ml.norm(normal) for x in normal]
        # normal = normal / np.linalg.norm(normal)

        u = 1 - ((np.arctan2(normal[2], normal[0]) / (2 * np.pi)) + 0.5)
        v = np.arccos(-normal[1]) / np.pi

        uvs = (u,v)
        # print(normal)
        return Intersect(distance = t0,
                         point = P,
                         normal = normal,
                         texcoords = uvs,
                         sceneObj = self)


class Plane(object):
    def __init__(self, position, normal,  material):
        self.position = position
        self.normal = [x / ml.norm(normal) for x in normal]
        self.material = material

    def ray_intersect(self, orig, dir):
        # Distancia = (( planePos - origRayo) o normal) / (direccionRayo o normal)
        denom = ml.dot( dir, self.normal)

        if abs(denom) > 0.0001:
            num = ml.dot( ml.subtract(self.position, orig), self.normal)
            t = num / denom

            if t > 0:
                # P = O + t*D
                P = ml.add(orig, [t * x for x in dir])
                return Intersect(distance = t,
                                 point = P,
                                 normal = self.normal,
                                 texcoords = None,
                                 sceneObj = self)

        return None

class Disk(object):
    def __init__(self, position, radius, normal,  material):
        self.plane = Plane(position, normal, material)
        self.material = material
        self.radius = radius

    def ray_intersect(self, orig, dir):

        intersect = self.plane.ray_intersect(orig, dir)

        if intersect is None:
            return None

        contact = ml.subtract(intersect.point, self.plane.position)
        contact = ml.norm(contact) 

        if contact > self.radius:
            return None

        return Intersect(distance = intersect.distance,
                         point = intersect.point,
                         normal = self.plane.normal,
                         texcoords = None,
                         sceneObj = self)

#Referencia: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
class Triangle(object):
    def __init__(self, v0, v1, v2, t, material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.t = t
        self.material = material
    
    def ray_intersect(self, orig, dir):
        v0v1 = ml.subtract(self.v1, self.v0)
        v0v2 = ml.subtract(self.v2, self.v0)

        N = ml.crossProduct(v0v1, v0v2)
        area2 = len(N)

        eps = 1.0
        while eps + 1 > 1:
            eps /= 2
        eps *= 2

        NdotDireccionRayo = ml.dot(N, dir)
        if math.fabs(NdotDireccionRayo) < eps:
            return None
        
        d = -(ml.dot(N, self.v0))

        self.t = -(ml.dot(N, orig) + d) / NdotDireccionRayo

        if self.t < 0:
            return None
        
        # P = orig+self.t*dir
        P = [self.t * x for x in dir]
        try:
            P =[orig.x + P[0], orig.y + P[1], orig.z + P[2]]
        except:
            P = [orig[0] + P[0], orig[1] + P[1], orig[2] + P[2]]
        #Edge 0
        edge0 = ml.subtract(self.v1, self.v0)
        vp0 = ml.subtract(P, self.v0)
        
        C = ml.crossProduct(edge0, vp0)

        if ml.dot(N, C) < 0:
            return None
        
        #Edge 1
        
        edge1 = ml.subtract(self.v2, self.v1)
        vp1 = ml.subtract(P, self.v1)
        C = ml.crossProduct(edge1, vp1)

        if ml.dot(N, C) < 0:
            return None

        #Edge 2
        edge2 = ml.subtract(self.v0, self.v2)
        vp2 = ml.subtract(P, self.v2)
        C = ml.crossProduct(edge2, vp2)

        if ml.dot(N,C) < 0:
            return None

        return Intersect(
            distance=self.t,
            point=P,
            normal=N,
            texcoords=None,
            sceneObj=self
        )



class AABB(object):
    # Axis Aligned Bounding Box

    def __init__(self, position, size, material):
        self.position = position
        self.size = size
        self.material = material

        self.planes = []

        halfSizes = [0,0,0]

        halfSizes[0] = size[0] / 3
        halfSizes[1] = size[1] / 3
        halfSizes[2] = size[2] / 3

        # Sides
        self.planes.append( Plane( ml.add(position, (halfSizes[0],0,0)), (1,0,0), material ))
        self.planes.append( Plane( ml.add(position, (-halfSizes[0],0,0)), (-1,0,0), material ))

        # Up and Down
        self.planes.append( Plane( ml.add(position, (0,halfSizes[1],0)), (0,1,0), material ))
        self.planes.append( Plane( ml.add(position, (0,-halfSizes[1],0)), (0,-1,0), material ))

        # Front and back
        self.planes.append( Plane( ml.add(position, (0,0,halfSizes[2])), (0,0,1), material ))
        self.planes.append( Plane( ml.add(position, (0,0,-halfSizes[2])), (0,0,-1), material ))

        #Bounds
        self.boundsMin = [0,0,0]
        self.boundsMax = [0,0,0]

        epsilon = 0.001

        for i in range(3):
            self.boundsMin[i] = self.position[i] - (epsilon + halfSizes[i]) 
            self.boundsMax[i] = self.position[i] + (epsilon + halfSizes[i]) 


    def ray_intersect(self, orig, dir):
        intersect = None
        t = float('inf')

        for plane in self.planes:
            planeInter = plane.ray_intersect(orig, dir)
            if planeInter is not None:

                planePoint = planeInter.point

                if self.boundsMin[0] <= planePoint[0] <= self.boundsMax[0]:
                    if self.boundsMin[1] <= planePoint[1] <= self.boundsMax[1]:
                        if self.boundsMin[2] <= planePoint[2] <= self.boundsMax[2]:

                            if planeInter.distance < t:
                                t = planeInter.distance
                                intersect = planeInter

                                # Tex Coords

                                u, v = 0, 0

                                # Las uvs de las caras de los lados
                                if abs(plane.normal[0]) > 0:
                                    # Mapear uvs para el eje x, usando las coordenadas de Y y Z
                                    u = (planeInter.point[1] - self.boundsMin[1]) / self.size[1]
                                    v = (planeInter.point[2] - self.boundsMin[2]) / self.size[2]

                                elif abs(plane.normal[1] > 0):
                                    # Mapear uvs para el eje y, usando las coordenadas de X y Z
                                    u = (planeInter.point[0] - self.boundsMin[0]) / self.size[0]
                                    v = (planeInter.point[2] - self.boundsMin[2]) / self.size[2]

                                elif abs(plane.normal[2] > 0):
                                    # Mapear uvs para el eje z, usando las coordenadas de X y Y
                                    u = (planeInter.point[0] - self.boundsMin[0]) / self.size[0]
                                    v = (planeInter.point[1] - self.boundsMin[1]) / self.size[1]


        if intersect is None:
            return None

        return Intersect(distance = t,
                         point = intersect.point,
                         normal = intersect.normal,
                         texcoords = (u,v),
                         sceneObj = self)

