from platform import machine
from typing import Text
from gl import Raytracer, V3
from texture import *
from figures import *
from lights import *


width = 512
height = 512

# Materiales

brick = Material(diffuse = (0.8, 0.3, 0.3), spec = 16)
stone = Material(diffuse = (0.4, 0.4, 0.4), spec = 8)
grass = Material(diffuse = (0.3, 1.0, 0.3), spec = 64)
marble = Material(spec = 64, texture = Texture("marble.bmp"), matType= REFLECTIVE)
mirror = Material(diffuse = (0.9, 0.9, 0.9), spec = 64, matType = REFLECTIVE)
glass = Material(diffuse = (0.9, 0.9, 0.9), spec = 64, ior = 1.5, matType = TRANSPARENT)
kirby = Material(texture=Texture("kirby.bmp"), spec=64, ior=1.5, matType= TRANSPARENT)
machine = Material(texture=Texture("machine.bmp"), spec=16, matType = REFLECTIVE)
mat1 = Material(diffuse=(0.5, 0.8, 0.2), spec=64, matType=REFLECTIVE)
headmat = Material(diffuse = (0.9,0.9,0.9), texture=Texture("head.bmp"), spec=64, matType=REFLECTIVE)

rtx = Raytracer(width, height)

rtx.envMap = Texture("envmap.bmp")

rtx.lights.append( AmbientLight(intensity = 0.1 ))
rtx.lights.append( DirectionalLight(direction = (0,0,-1), intensity = 0.5 ))
rtx.lights.append( DirectionalLight(direction = (0,0,-5), intensity = 0.8 ))
# rtx.lights.append( PointLight(point = (-1,-1,0) ))
# rtx.lights.append( PointLight(point = (0,0,-5)))
# rtx.lights.append( PointLight(point = (-5,3,-7)))

rtx.scene.append( Sphere(center=(0,-2,-10), radius=2, material=kirby))
rtx.scene.append( Sphere(center=(-5,-2,-10), radius=3, material=headmat))
rtx.scene.append( Sphere(center=(-5,-2,-5), radius=1, material=headmat))
rtx.scene.append( Sphere(center=(-5,-2,-4), radius=0.75, material=headmat))
rtx.scene.append( Sphere(center=(5, -2, -10), radius=3, material=marble))
rtx.scene.append( Sphere(center=(5, -2, -5), radius=1, material=marble))
rtx.scene.append( Sphere(center=(5, -2, -4), radius=0.75, material=marble))
rtx.scene.append( AABB(position=(-3.5, 3, -5), size=(2, 5, 5), material=machine))
rtx.scene.append( Triangle(v0=(3,3,-15), v1=(-5, 2, -7) , v2=(-2, 5, -10), t= None, material = mirror) )
rtx.scene.append( Disk(position=(3, -5, -10), radius=3, normal=(0,1,0), material=grass))
rtx.scene.append( Disk(position=(-3, -5, -10), radius=3, normal=(0,1,0), material=brick))


rtx.glRender()

rtx.glFinish("output.bmp")