"""Render one Chrono-exported demo episode to a PNG sequence, with the PLANNED route drawn on the ground.

blender -b --factory-startup --python this.py -- --asset <exported.assets.py> --frames N --out <dir>
        --route <xyz.npy>  [--color 0.1,0.8,0.2] [--goal x,y,r] [--res 1280x720] [--every 1]

The route ribbon is built from a pre-sampled XYZ polyline (terrain height + clearance), so the
viewer sees what the planner asked for and what the vehicle actually did with it.
"""
import bpy, math, sys, os, time
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d

ASSET = opt('--asset'); OUT = opt('--out'); NF = int(opt('--frames', '100'))
RES = opt('--res', '1280x720'); RW, RH = (int(x) for x in RES.split('x'))
EVERY = int(opt('--every', '1'))
ROUTE = opt('--route'); GOAL = opt('--goal'); COLOR = opt('--color', '0.05,0.75,0.25')
os.makedirs(OUT, exist_ok=True)


def freeze():
    rm = []
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, '__name__', '') == 'callback_post':
            bpy.app.handlers.frame_change_post.remove(h); rm.append(h)
    return rm


def restore(hs):
    for h in hs:
        if h not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(h)


def mat(name, color, rough=0.6, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    if b:
        b.inputs['Base Color'].default_value = color
        b.inputs['Roughness'].default_value = rough
        b.inputs['Metallic'].default_value = metal
        if emit:
            if 'Emission Color' in b.inputs: b.inputs['Emission Color'].default_value = color
            if 'Emission Strength' in b.inputs: b.inputs['Emission Strength'].default_value = emit
    m.diffuse_color = color
    return m


def names(o):
    s = [o.name]; p = o.parent
    while p is not None:
        s.append(p.name); p = p.parent
    return ' '.join(s).lower()


bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.preferences.addon_enable(module='chrono_import')
t0 = time.time()
bpy.ops.import_chrono.data(filepath=ASSET, setting_materials=True, setting_merge=False)
print('IMPORT_S', round(time.time() - t0, 1), flush=True)

sc = bpy.context.scene
sc.frame_start = 0; sc.frame_end = NF - 1
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x = RW; sc.render.resolution_y = RH
sc.render.image_settings.file_format = 'PNG'
# Blender 4+/5 defaults to the AgX view transform, which washes every material out to pale grey.
try:
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0.0
except Exception as e:
    print('view transform not set:', e)

cd = bpy.data.cameras.new('chase'); cam = bpy.data.objects.new('chase', cd)
sc.collection.objects.link(cam); cd.clip_end = 6000; cd.lens = 32

sd = bpy.data.lights.new('sun', 'SUN'); sun = bpy.data.objects.new('sun', sd)
sc.collection.objects.link(sun)
sun.rotation_euler = (math.radians(48), 0, math.radians(140)); sd.energy = 1.25
sd.angle = math.radians(3)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.68, 0.80, 0.98, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 0.35

M_BODY = mat('body', (0.24, 0.27, 0.15, 1), rough=0.55)
M_TIRE = mat('tire', (0.045, 0.045, 0.05, 1), rough=0.9)
M_TERR = mat('terrain', (0.40, 0.34, 0.22, 1), rough=0.95)
M_METAL = mat('metal', (0.25, 0.26, 0.27, 1), rough=0.4, metal=0.7)

EXTRA = set()

if ROUTE:
    p = np.load(ROUTE).astype(float)
    c = tuple(float(x) for x in COLOR.split(',')) + (1.0,)
    M_ROUTE = mat('route', c, rough=0.35, emit=1.4)
    half = 0.45
    d = np.gradient(p[:, :2], axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    verts, faces = [], []
    for i in range(len(p)):
        verts.append((p[i, 0] - nrm[i, 0] * half, p[i, 1] - nrm[i, 1] * half, p[i, 2]))
        verts.append((p[i, 0] + nrm[i, 0] * half, p[i, 1] + nrm[i, 1] * half, p[i, 2]))
    for i in range(len(p) - 1):
        faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    me = bpy.data.meshes.new('routemesh'); me.from_pydata(verts, [], faces); me.update()
    ob = bpy.data.objects.new('plannedroute', me); sc.collection.objects.link(ob)
    ob.data.materials.append(M_ROUTE); EXTRA.add(ob.name)

if GOAL:
    gx, gy, gr = (float(x) for x in GOAL.split(','))
    gz = float(np.load(ROUTE)[-1, 2]) if ROUTE else 0.0
    bpy.ops.mesh.primitive_cylinder_add(radius=gr, depth=0.18, location=(gx, gy, gz + 0.09))
    g = bpy.context.active_object; g.name = 'goalmark'
    g.data.materials.append(mat('goal', (1.0, 0.85, 0.1, 1), rough=0.3, emit=2.0)); EXTRA.add(g.name)

styled = set()


def style():
    for o in bpy.data.objects:
        if o.name in EXTRA:
            continue
        if o.type in {'GREASEPENCIL', 'GPENCIL', 'CURVE'}:
            o.hide_render = True; o.hide_viewport = True
            continue
        if o.type != 'MESH' or o.name in styled:
            continue
        styled.add(o.name)
        n = names(o)
        if 'patch' in n or 'terrain' in n:
            m = M_TERR
        elif 'tire' in n or 'wheel' in n:
            m = M_TIRE
        elif 'spindle' in n or 'susp' in n or 'link' in n or 'arm' in n:
            m = M_METAL
        else:
            m = M_BODY
        try:
            o.data.materials.clear(); o.data.materials.append(m)
        except Exception:
            pass
        # The Chrono importer links some materials at OBJECT level, which overrides the mesh data
        # slot; write the slot too or those bodies keep the importer's pale default.
        try:
            for s in o.material_slots:
                s.link = 'OBJECT'; s.material = m
        except Exception:
            pass


rendered = 0
t0 = time.time()
for f in range(0, NF, EVERY):
    sc.frame_set(f); bpy.context.view_layer.update()
    style()
    ch = bpy.data.objects.get('Chassis body')
    if ch is None:
        continue
    p = ch.matrix_world.translation.copy()
    yaw = ch.matrix_world.to_euler().z
    back = Vector((-math.cos(yaw), -math.sin(yaw), 0))
    side = Vector((-math.sin(yaw), math.cos(yaw), 0))
    sc.camera = cam
    cam.location = p + back * 14.5 + side * 4.5 + Vector((0, 0, 8.5))
    cam.rotation_euler = ((p + Vector((0, 0, 1.0))) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(OUT, f'frame{rendered:04d}.png')
    hs = freeze(); bpy.ops.render.render(write_still=True); restore(hs)
    rendered += 1
    if rendered % 50 == 0:
        print('RENDERED', rendered, round(time.time() - t0, 1), 's', flush=True)
print('DONE frames', rendered, 'wall', round(time.time() - t0, 1), flush=True)
