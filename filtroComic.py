import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("careta3.jpg")

# Verificar si la imagen cargó
if imagen is None:
    print("Error. No se pudo cargar la imagen")
    exit()

# 1. Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Aplicar desenfoque mediano para reducir el ruido
gris_suave = cv2.medianBlur(gris, 7)

# 3. Detectar bordes con umbral adaptativo 
# Usar THRESH_BINARY_INV para obtener líneas negras directamente
bordes = cv2.adaptiveThreshold(gris_suave, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,  # Invertido para líneas negras
                               blockSize=9, C=2)

# 4. Crear fondo completamente blanco
alto, ancho = bordes.shape
fondo_blanco = np.ones((alto, ancho), dtype=np.uint8) * 255

# 5. Donde hay líneas negras (valor 255 en bordes), poner negro (0)
# Donde no hay líneas (valor 0 en bordes), mantener blanco (255)
comic_gris = np.where(bordes == 255, 0, 255).astype(np.uint8)

# 6. Convertir a imagen a color (3 canales)
comic_final = cv2.cvtColor(comic_gris, cv2.COLOR_GRAY2BGR)

# 6. Opcional: Aplicar un filtro bilateral para suavizar un poco las líneas
# manteniendo los bordes definidos
comic_final = cv2.bilateralFilter(comic_final, 9, 200, 200)

# Mostrar imágenes lado a lado
# Redimensionar si es necesario para mejor visualización
altura, ancho = imagen.shape[:2]
if ancho > 800:  # Si la imagen es muy grande, redimensionar
    factor = 800 / ancho
    nuevo_ancho = int(ancho * factor)
    nueva_altura = int(altura * factor)
    imagen_mostrar = cv2.resize(imagen, (nuevo_ancho, nueva_altura))
    comic_mostrar = cv2.resize(comic_final, (nuevo_ancho, nueva_altura))
else:
    imagen_mostrar = imagen
    comic_mostrar = comic_final

# Mostrar imágenes
cv2.imshow("Imagen Original", imagen_mostrar)
cv2.imshow("Filtro Comic - Lineas Negras sobre Fondo Blanco", comic_mostrar)

# Guardar la imagen procesada
cv2.imwrite("careta_comic_corregido.jpeg", comic_final)
print("Imagen estilo cómic guardada correctamente con líneas negras sobre fondo blanco")

# Esperar hasta que se presione una tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()