from gltflib import GLTF

gltf = GLTF.load_glb("resources/estensi.glb", load_file_resources=True)

resource = gltf.resources[0]
print(gltf.model.buffers[0].uri)

print(1)
