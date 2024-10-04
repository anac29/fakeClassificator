from PIL import Image

# Abre la imagen
image_path = 'pytorchdata/test/fake/e47010eca54238fc160c4a9045bf0eef_jpeg.rf.1259bb2175796eceb2ded37f7f0f4c78.jpg'
image = Image.open(image_path)

# Obtiene las dimensiones
width, height = image.size
print(f'Ancho: {width}, Alto: {height}')
print(image.mode)