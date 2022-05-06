from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1.5, voxel_edges=0)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 2, -1), 0.2, (1, 0.8, 0.6))
manual_seed, riverCount, treeCount = 3, 10, 10
loop = [[0, 0], [1, 0], [0, 1], [1, 1]]

@ti.func
def lerp(e1: ti.f32, e2: ti.f32, w: ti.f32): return e1 + (e2 - e1) * (w ** 2 * (3 - 2 * w))
@ti.func
def perlinNoise(size: ti.i8, amp: ti.f32):
    for x, y in ti.ndrange(size + 1, size + 1):
        scene.set_voxel(vec3(x, 1, y), 1, vec3(*[ti.random() for _ in range(3)]))
    for x, y in ti.ndrange(127, 127):
        u, v = ti.floor(x * size / 127), ti.floor(y * size / 127)
        x1, y1 = x / 127. * size - u, y / 127. * size - v
        info = [scene.get_voxel(vec3(u + it[0], 1, v + it[1]))[1] for it in loop]
        zs = [info[i][0] + info[i][1] * (x1 - loop[i][0]) + info[i][2] * (y1 - loop[i][1]) for i in range(4)]
        ls = [lerp(zs[i * 2], zs[i * 2 + 1], x1) for i in range(2)]
        scene.set_voxel(vec3(x - 64, 0, y - 64), 1,
                        vec3(lerp(ls[0], ls[1], y1) * amp + 0.04 + scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0]))
@ti.func
def createRiver(x: ti.i8, y: ti.i8):
    count = 0
    while 0 < x < 127 and 0 < y < 127 and count < 128:
        zl = ti.cast(scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0] * 80, ti.i16)
        scene.set_voxel(vec3(x - 64, zl + 1, y - 64), 2, vec3(0.12, 0.56, 1))
        for u, v in ti.ndrange((x - ti.random() - 1, x + ti.random() + 2), (y - ti.random() - 1, y + ti.random() + 2)):
            zll = ti.cast(scene.get_voxel(vec3(u - 64, 0, v - 64))[1][0] * 80, ti.i16)
            for z in range(zll - 4, zll + 2): scene.set_voxel(vec3(u - 64, z, v - 64), 1 if z < zl else 0,
                                                              vec3(0.12, 0.56, 1) + (ti.random() - 0.5) * 0.15)
        dx = ti.Vector([1, 1, 0, -1, -1, -1, 0, 1, 2, 2, 0, -2, -2, -2, 0, 2])
        dy = ti.Vector([0, 1, 1, 1, 0, -1, -1, -1, 0, 2, 2, 2, 0, -2, -2, -2])
        zm = 1
        for i in ti.static(range(16)):
            if ti.cast(scene.get_voxel(vec3(x - 64 + dx[i], 0, y - 64 + dy[i]))[1][0] * 80, ti.i16) - zl < zm:
                zm = ti.cast(scene.get_voxel(vec3(x - 64 + dx[i], 0, y - 64 + dy[i]))[1][0] * 80, ti.i16) - zl
                x, y = x + dx[i], y + dy[i]; break
        if zm > 0: break
        count += 1
@ti.func
def createTree(x: ti.i16, y: ti.i16):
    zl = ti.cast(scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0] * 80, ti.i16)
    high = ti.cast(ti.random() * 5, ti.i16)
    for z in range(zl, zl + high + 10):
        scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
        if high > 3:
            scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
            scene.set_voxel(vec3(x - 63, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
            scene.set_voxel(vec3(x - 63, z, y - 63), 1, vec3(0.85, 0.65, 0.41))
    for z in range(zl + high + 8, zl + high + 16):
        r = zl + high + 18 - z
        for u, v in ti.ndrange((-r, r), (-r, r)):
            prob = ti.random() * 0.5 + (1 - ti.sqrt(u ** 2 + v ** 2) / r) * 0.7
            if prob > 0.3:
                scene.set_voxel(vec3(u + x - 64, z, v + y - 64), 1, vec3(0.22, 0.37, 0.06) + (ti.random() - 0.5) * 0.05)

@ti.kernel
def createTerrain():
    for x, y, z in ti.ndrange(127, 127, 127): scene.set_voxel(vec3(x - 64, z - 64, y - 64), 0, vec3(0))
    perlinNoise(2, 0.3)
    perlinNoise(8, 0.2)
    perlinNoise(32, 0.1)
    for x, y in ti.ndrange(127, 127):
        zl = ti.cast(scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0] * 80, ti.i16)
        for i in range(zl):
            color = vec3(0.5) if i < zl - (ti.random() - 0.5) * 2 - 6 else vec3(0.45, 0.29,0.07) \
                if i < zl - 1 else vec3(0.13,0.55,0.13)
            scene.set_voxel(vec3(x - 64, i + 2, y - 64), 1, color + (ti.random() - 0.5) * 0.15)
    [createRiver(ti.random() * 127, ti.random() * 127) for _ in range(20)]


@ti.kernel
def createRivers():
    [createRiver(ti.random() * 110 + 5, ti.random() * 110 + 5) for _ in range(riverCount)]


@ti.kernel
def createTrees():
    [createTree(ti.random() * 100 + 13, ti.random() * 100 + 13) for _ in range(treeCount)]


@ti.kernel
def fillTemp():
    for x, y, z in ti.ndrange(127, 127, 2):
        scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.5) + (ti.random() - 0.5) * 0.15)


[createTerrain() for _ in range(manual_seed)]
createRivers()
createTrees()
fillTemp()
scene.finish()
