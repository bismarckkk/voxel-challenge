from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=20)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))
loop = [[0, 0], [1, 0], [0, 1], [1, 1]]

@ti.func
def lerp(e1:ti.f32, e2:ti.f32, w:ti.f32):return e1 + (e2 - e1) * (w ** 2 * (3 - 2 * w))

@ti.func
def perlinNoise(size: ti.i8, amp: ti.f32):
    for x, y in ti.ndrange(size+1, size+1):
        scene.set_voxel(vec3(x, 1, y), 1, vec3(*[ti.random() for _ in range(3)]))
    for x, y in ti.ndrange(127, 127):
        u, v = ti.floor(x * size / 127), ti.floor(y * size / 127)
        x1, y1 = x / 127. * size - u, y / 127. * size - v
        info = [scene.get_voxel(vec3(u+it[0], 1, v+it[1]))[1] for it in loop]
        zs = [info[i][0]+info[i][1]*(x1 - loop[i][0])+info[i][2]*(y1 - loop[i][1]) for i in range(4)]
        ls = [lerp(zs[i * 2], zs[i * 2 + 1], x1) for i in range(2)]
        scene.set_voxel(vec3(x-64, 0, y-64), 1, vec3(lerp(ls[0], ls[1], y1)*amp+0.05+scene.get_voxel(vec3(x-64, 0, y-64))[1][0]))


@ti.kernel
def initialize_voxels():
    perlinNoise(2, 0.3)
    perlinNoise(8, 0.2)
    perlinNoise(32, 0.1)
    for x, y in ti.ndrange(127, 127):
        zl = ti.cast(scene.get_voxel(vec3(x-64, 0, y-64))[1][0] * 64, ti.i16)-16
        for i in range(zl):
            scene.set_voxel(vec3(x-64, i+16, y-64), 1, vec3(0.01))



initialize_voxels()
scene.finish()
