from scene import Scene
import taichi as ti
from taichi.math import *
manual_seed, biome, snow = 2, 'island', False
loop= [[0, 0], [1, 0], [0, 1], [1, 1]]
terrain = [[2, 0.12], [8, 0.06], [32, 0.03]] \
    if biome != 'forest' and biome != 'island' else [[2, 0.3], [8, 0.2], [32, 0.1]]
scene = Scene(exposure=1.5 if biome != 'desert' else 1.2, voxel_edges=0)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.94, 1, 1))
scene.set_directional_light((1, 2, -1), 0.2, (1, 0.8, 0.6))
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
    if biome == 'desert':
        for z in range(zl+2, int(zl+4+ti.random()*2)): scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(.22, .37, .06))
    elif zl + 1 >= 45 or biome != 'island':
        high = ti.cast(ti.random() * 5, ti.i16)
        for z in range(zl, zl + high + 10):
            scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
            if high > 3:
                scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
                scene.set_voxel(vec3(x - 63, z, y - 64), 1, vec3(0.85, 0.65, 0.41))
                scene.set_voxel(vec3(x - 63, z, y - 63), 1, vec3(0.85, 0.65, 0.41))
        for z in range(zl + high + (5 if snow else 8), zl + high + 16):
            r = (zl + high + 18 - z) * (0.5 if snow else 1)
            for u, v in ti.ndrange((-r, r), (-r, r)):
                prob = ti.random() * 0.5 + (1 - ti.sqrt(u ** 2 + v ** 2) / r) * 0.7
                if prob > 0.3:
                    scene.set_voxel(vec3(u + x - 64, z, v + y - 64), 1, vec3(.22, .37, .06) + (ti.random() - .5) * .05)
@ti.kernel
def createTerrain():
    for x, y, z in ti.ndrange(127, 127, 127): scene.set_voxel(vec3(x - 64, z - 64, y - 64), 0, vec3(0))
    if True: [perlinNoise(it[0], it[1]) for it in terrain]
    for x, y in ti.ndrange(127, 127):
        zl = ti.cast(scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0] * 80, ti.i16)
        for i in range(zl):
            color = vec3(.97, .97, 1) if snow else vec3(0.13,0.55,0.13)
            if i < zl - (ti.random() - 0.5) * 2 - 6: color = vec3(0.5)
            elif biome == 'desert': color = vec3(.96, .87, .7)
            elif zl < 41 and biome == 'island': color = vec3(.96, .87, .7)
            elif i < zl - 1: color = vec3(0.45, 0.29,0.07)
            scene.set_voxel(vec3(x - 64, i + 2, y - 64), 1, color + (ti.random() - 0.5) * 0.15)
@ti.kernel
def createRivers():
    [createRiver(ti.random() * 110 + 5, ti.random() * 110 + 5) for _ in range(20)]
@ti.kernel
def createTrees():
    [createTree(ti.random() * 100 + 13, ti.random() * 100 + 13) for _ in range(5)]
@ti.kernel
def fillTemp():
    if biome == 'island':
        for x, y in ti.ndrange(127, 127):
            zl = ti.cast(scene.get_voxel(vec3(x - 64, 0, y - 64))[1][0] * 80, ti.i16)+2
            if zl >= 42: continue
            for z in range(zl, 42):scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(.07, .49, .95) + ti.random() * 0.1)
    for x, y, z in ti.ndrange(127, 127, 2):
        scene.set_voxel(vec3(x - 64, z, y - 64), 1, vec3(0.5) + (ti.random() - 0.5) * 0.15)
[createTerrain() for _ in range(manual_seed)]
if biome == 'grass' or biome == 'desert': createTrees()
elif biome == 'forest': createRivers(); createTrees(); createTrees()
elif biome == 'island': createTrees(); createTrees(); createTrees()
fillTemp()
scene.finish()
