"""Print the Vulkan physical devices the loader sees (name reveals llvmpipe's LLVM version and SIMD width)."""
import ctypes, json, struct, sys

lib = ctypes.CDLL('libvulkan.so.1')


class AppInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('pApplicationName', ctypes.c_char_p),
                ('applicationVersion', ctypes.c_uint32), ('pEngineName', ctypes.c_char_p),
                ('engineVersion', ctypes.c_uint32), ('apiVersion', ctypes.c_uint32)]


class InstInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('flags', ctypes.c_uint32),
                ('pApplicationInfo', ctypes.POINTER(AppInfo)), ('enabledLayerCount', ctypes.c_uint32),
                ('ppEnabledLayerNames', ctypes.c_void_p), ('enabledExtensionCount', ctypes.c_uint32),
                ('ppEnabledExtensionNames', ctypes.c_void_p)]


app = AppInfo(0, None, b'vk_device', 1, b'probe', 1, (1 << 22) | (3 << 12))
ci = InstInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
inst = ctypes.c_void_p()
r = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
if r != 0:
    print(json.dumps({'vkCreateInstance': r})); sys.exit(0)
lib.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
lib.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
n = ctypes.c_uint32(0)
lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
devs = (ctypes.c_void_p * n.value)()
lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
out = []
for d in devs:
    buf = ctypes.create_string_buffer(16384)
    lib.vkGetPhysicalDeviceProperties(d, buf)
    api, drv, vendor, devid, dtype = struct.unpack_from('<5I', buf.raw, 0)
    name = buf.raw[20:276].split(b'\0')[0].decode()
    out.append({'name': name, 'api': f'{api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}',
                'driver_version_raw': drv, 'driver_version_mesa': f'{drv >> 22}.{(drv >> 12) & 0x3ff}.{drv & 0xfff}',
                'vendor': hex(vendor), 'device_type': dtype})
print(json.dumps(out))
