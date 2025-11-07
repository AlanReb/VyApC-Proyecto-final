import argparse
import cv2
import struct
import math
import numpy as np
import os
import concurrent.futures
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial import cKDTree
from PIL import Image  
from collections import defaultdict


def abrir_imagen_arg():
        # Configurar argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--image", required=True, help="Path to the input image")
        args = vars(parser.parse_args())

        # Captura de imagen
        input_image = cv2.imread(args["image"])

        if input_image is None:
            print("Error: No se pudo cargar la imagen.")
            exit()
        return input_image

def abrir_imagen(path):
        orig_img = cv2.imread(path)
        
        return orig_img

def paso_gris(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        return gray
    
def to_int_xy(arr,w,h):
        # redondeo y recorte al tamaño de imagen
        arr_r = np.rint(arr).astype(int)
        arr_r[:, 0] = np.clip(arr_r[:, 0], 0, w - 1)  # x
        arr_r[:, 1] = np.clip(arr_r[:, 1], 0, h - 1)  # y
        return arr_r
    
def binarizar(img_gray, thr=200):
        """Umbral fijo -> matriz 0/1"""
        return (img_gray >= thr).astype(np.uint8)
    
def binarizar_rang(img_gray, uthr=200, lthr=140):
        """Umbral fijo -> matriz 0/1"""
        return ((img_gray >= lthr) & (img_gray <= uthr)).astype(np.uint8)

def invertir_colores(img):
    img_uint8 = (img > 0).astype(np.uint8) * 255  # asegurar 0/255
    inverted = cv2.bitwise_not(img_uint8)
    return (inverted > 0).astype(np.uint8)

def recortar(img,x1,x2,y1,y2):
    return img[y1:y2, x1:x2]

def rgb_to_cmy_pixel(r, g, b):
    """Entradas r,g,b en [0,1]. Devuelve C,M,Y en [0,1]."""
    C = 1.0 - r
    M = 1.0 - g
    Y = 1.0 - b
    # K vacío (0) → no se oscurece nada, solo para visualización
    K = 0.0
    return C, M, Y, K
    
def cmy_to_rgb_pixel(C, M, Y):
    """Entradas C,M,Y en [0,1]. Devuelve r,g,b en [0,1]."""
    r = 1.0 - C
    g = 1.0 - M
    b = 1.0 - Y
    return r, g, b

def rgb_to_cmyk_pixel(r, g, b):
    """Entradas r,g,b en [0,1]. Devuelve C,M,Y,K en [0,1]."""
    # Calcular CMY inicial
    C = 1.0 - r
    M = 1.0 - g
    Y = 1.0 - b
    
    #Calcular K
    K = min(C, M, Y)

    # Ajustar C,M,Y con K
    if K >= 1.0:  # Pixel negro puro
        C_adj = 0.0
        M_adj = 0.0
        Y_adj = 0.0
    else:
        C_adj = (C - K) / (1.0 - K)
        M_adj = (M - K) / (1.0 - K)
        Y_adj = (Y - K) / (1.0 - K)
    
    return C_adj, M_adj, Y_adj, K

def cmyk_to_rgb_pixel(C, M, Y, K):
    """
    Entradas C,M,Y,K en [0,1].
    Devuelve r,g,b en [0,1].
    """
    
    r = 1.0 - (C * (1.0 - K)) - K
    g = 1.0 - (M * (1.0 - K)) - K
    b = 1.0 - (Y * (1.0 - K)) - K
    
    return r, g, b

def rgb_to_hsi_pixel(r, g, b):
    """
    RGB->HSI (pixel)
    r,g,b in [0,1]
    H returned in degrees [0,360)
    S in [0,1], I in [0,1]
    Fórmula clásica HSI (no HSV).
    """
    # intensidad
    I = (r + g + b) / 3.0

    # saturación (definición HSI clásica)
    denom = (r + g + b)
    if denom == 0:
        S = 0.0
    else:
        m = min(r, g, b)
        S = 1.0 - (3.0 * m / denom)

    # hue
    # evitar división por cero numérica
    num = 0.5 * ((r - g) + (r - b))
    den = math.sqrt((r - g) * (r - g) + (r - b) * (g - b))
    if den == 0:
        theta = 0.0
    else:
        val = num / den
        # acotar por [-1,1] por seguridad numérica
        val = max(-1.0, min(1.0, val))
        theta = math.degrees(math.acos(val))  # en grados

    if b <= g:
        H = theta
    else:
        H = 360.0 - theta

    # Asegurar rango
    H = H % 360.0
    return H, S, I

def hsi_to_rgb_pixel(H_deg, S, I):

    #H in degrees [0,360), S in [0,1], I in [0,1]
    #Devuelve r,g,b en [0,1]
    
    H = H_deg % 360.0
    H_rad = math.radians(H)
    # si S == 0, color es gris
    if S == 0:
        return I, I, I

    # sector 0: 0 <= H < 120
    if 0 <= H < 120:
        # formulas con H en radianes
        # R = I*(1 + S*cos(H)/cos(60°-H))
        # B = I*(1 - S)
        cosH = math.cos(H_rad)
        cos60_H = math.cos(math.radians(60.0) - H_rad)
        if cos60_H == 0:
            # numérico - usar aproximación
            R = I * (1 + S)
        else:
            R = I * (1 + (S * cosH) / cos60_H)
        B = I * (1 - S)
        G = 3 * I - (R + B)
    elif 120 <= H < 240:
        H2 = H - 120.0
        H2_rad = math.radians(H2)
        cosH2 = math.cos(H2_rad)
        cos60_H2 = math.cos(math.radians(60.0) - H2_rad)
        if cos60_H2 == 0:
            G = I * (1 + S)
        else:
            G = I * (1 + (S * cosH2) / cos60_H2)
        R = I * (1 - S)
        B = 3 * I - (R + G)
    else:  # 240 <= H < 360
        H3 = H - 240.0
        H3_rad = math.radians(H3)
        cosH3 = math.cos(H3_rad)
        cos60_H3 = math.cos(math.radians(60.0) - H3_rad)
        if cos60_H3 == 0:
            B = I * (1 + S)
        else:
            B = I * (1 + (S * cosH3) / cos60_H3)
        G = I * (1 - S)
        R = 3 * I - (G + B)

    # recortar a [0,1]
    def clamp01(x): return max(0.0, min(1.0, x))
    return clamp01(R), clamp01(G), clamp01(B)

def apply_pixelwise(img_uint8, func_pixel):

    #Devuelve array con dtype float64 del resultado por pixel

    h, w, ch = img_uint8.shape
    assert ch == 3
    # probamos 3 canales de salida asumiendo 3 (para HSI: 3), o 4 (CMYK) manejamos arriba
    # ejecutamos dos pasadas: averiguamos longitud del retorno con la primera llamada
    first = True
    out_arr = None
    for y in range(h):
        row_vals = []
        for x in range(w):
            r = img_uint8[y, x, 0] / 255.0
            g = img_uint8[y, x, 1] / 255.0
            b = img_uint8[y, x, 2] / 255.0
            out = func_pixel(r, g, b)
            if first:
                nchan = len(out)
                out_arr = np.zeros((h, w, nchan), dtype=np.float64)
                first = False
            row_vals.append(out)
            out_arr[y, x, :] = out
    return out_arr

def rgb_image_to_cmy_and_save(img_rgb_uint8, path_tiff):
    """Convierte RGB uint8 a CMYK float64 y guarda TIFF"""
    def f(r, g, b):
        return rgb_to_cmy_pixel(r, g, b)
    
    # Convertimos pixel a pixel
    cmy = apply_pixelwise(img_rgb_uint8, f)  # float64 [0,1]
    
    # Escalar a uint8
    cmy_uint8 = (cmy * 255).astype(np.uint8)

    # Guardar como TIFF CMYK
    img_cmyk_pil = Image.fromarray(cmy_uint8, mode="CMYK")
    img_cmyk_pil.save(path_tiff, format="TIFF")

    return cmy

def rgb_image_to_cmyk_and_save(img_rgb_uint8, path_tiff):
    """Convierte RGB uint8 a CMYK float64 y guarda TIFF"""
    def f(r, g, b):
        return rgb_to_cmyk_pixel(r, g, b)
    
    # Convertimos pixel a pixel
    cmyk = apply_pixelwise(img_rgb_uint8, f)  # float64 [0,1]
    
    # Escalar a uint8
    cmyk_uint8 = (cmyk * 255).astype(np.uint8)

    # Guardar como TIFF CMYK
    img_cmyk_pil = Image.fromarray(cmyk_uint8, mode="CMYK")
    img_cmyk_pil.save(path_tiff, format="TIFF")

    return cmyk

def cmy_image_to_rgb(cmy_img):
    """cmy_img float [0,1] HxWx4 -> devuelve rgb uint8 (0-255)"""
    h, w, c = cmy_img.shape
    if c < 3:
        raise ValueError("La imagen CMY(K) debe tener al menos 3 canales")
    
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            # Usamos solo los tres primeros canales (C, M, Y) y descartamos K
            C, M, Y = cmy_img[y, x, :3]
            r, g, b = cmy_to_rgb_pixel(C, M, Y)
            rgb[y, x, :] = (r, g, b)
    
    return (rgb * 255.0).astype(np.uint8)

def cmyk_image_to_rgb(cmyk_img):
    """cmyk_img float [0,1] HxWx4 -> devuelve rgb uint8 (0-255)"""
    h, w, c = cmyk_img.shape
    if c < 3:
        raise ValueError("La imagen CMYK debe tener al menos 3 canales")
    
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            # Usamos solo los canales (C, M, Y) y K
            C, M, Y, K = cmyk_img[y, x, :4]
            r, g, b = cmyk_to_rgb_pixel(C, M, Y, K)
            rgb[y, x, :] = (r, g, b)
    
    return (rgb * 255.0).astype(np.uint8)

def rgb_image_to_hsi(img_rgb_uint8):
    """Devuelve H (deg),S(0-1),I(0-1) float64 HxWx3"""
    def f(r,g,b): return rgb_to_hsi_pixel(r,g,b)
    hsi = apply_pixelwise(img_rgb_uint8, f)
    return hsi

def hsi_image_to_rgb(hsi_img):
    """hsi_img: float HxWx3 with H in degrees, S,I in 0-1 -> devuelve RGB uint8"""
    h,w,_ = hsi_img.shape
    rgb = np.zeros((h,w,3), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            H,S,I = hsi_img[y,x,:]
            r,g,b = hsi_to_rgb_pixel(H,S,I)
            rgb[y,x,:] = (r,g,b)
    return (rgb * 255.0).astype(np.uint8)

def save_hsi_file(path, hsi_img):
    """
    Formato:
     - 4 bytes: magic b'HSI1'
     - 4 bytes: width (uint32)
     - 4 bytes: height (uint32)
     - Then width*height pixels, each pixel 3 float32: H (deg), S, I
     - Total bytes = 12 + 12*w*h
    """
    h, w, ch = hsi_img.shape
    assert ch == 3
    with open(path, "wb") as f:
        f.write(b'HSI1')
        f.write(struct.pack("<I", w))
        f.write(struct.pack("<I", h))
        # escribir datos en orden row-major H,S,I float32
        # convertir a float32 array shape (-1,3)
        arr = hsi_img.astype(np.float32).reshape(-1, 3)
        f.write(arr.tobytes())

def load_hsi_file(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b'HSI1':
            raise ValueError("No es un archivo HSI válido")
        w = struct.unpack("<I", f.read(4))[0]
        h = struct.unpack("<I", f.read(4))[0]
        data = f.read()
        arr = np.frombuffer(data, dtype=np.float32)
        arr = arr.reshape((h*w, 3))
        arr = arr.reshape((h, w, 3))
    return arr.astype(np.float64)  # H,S,I

def compare_images_or_col(img_rgb,cmy_img, cmyk_img, hsi_img):
    # Mostrar RGB original
    cv2.imshow("Original RGB", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Convertir y mostrar CMY como RGB para preview
    rgb_from_cmy = cmy_image_to_rgb(cmy_img)
    cv2.imshow("CMY a RGB preview", cv2.cvtColor(rgb_from_cmy, cv2.COLOR_RGB2BGR))
    
    # Convertir y mostrar CMYK como RGB para preview
    rgb_from_cmyk = cmyk_image_to_rgb(cmyk_img)
    cv2.imshow("CMYK a RGB preview", cv2.cvtColor(rgb_from_cmyk, cv2.COLOR_RGB2BGR))

    # Convertir y mostrar HSI como RGB para preview
    rgb_from_hsi = hsi_image_to_rgb(hsi_img)
    cv2.imshow("HSI a RGB preview", cv2.cvtColor(rgb_from_hsi, cv2.COLOR_RGB2BGR))

    print("Presiona cualquier tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_images_col(img_rgb, cmyk_img):
    
    #Comparacion entre imagen CMY con CMY de pillow
    # --- Convertir a objeto PIL ---
    pil_img_rgb = Image.fromarray(img_rgb)
    #Comparacion entre imagen CMYK con CMYK de pillow
    # --- RGB a CMYK ---
    pil_img_cmyk = pil_img_rgb.convert("CMYK")
    # --- Guardar CMYK ---
    pil_img_cmyk.save("imagen_cmyk_pil.tiff")
    # --- CMYK a RGB ---
    pil_img_rgb_2 = pil_img_cmyk.convert("RGB")

    # --- Convertir de vuelta a OpenCV (BGR) ---
    img_bgr_2 = cv2.cvtColor(np.array(pil_img_rgb_2), cv2.COLOR_RGB2BGR)


    # Convertir y mostrar CMYK como RGB manualmente
    rgb_from_cmyk = cmyk_image_to_rgb(cmyk_img)
    
    
    # --- Mostrar ---
    cv2.imshow("Original RGB (OpenCV)", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("CMYK a RGB manualmente", cv2.cvtColor(rgb_from_cmyk, cv2.COLOR_RGB2BGR))
    cv2.imshow("CMYK a RGB usando pillow", img_bgr_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def vecinos(i, j, h, w):
    if i > 0:       yield i-1, j
    if i+1 < h:     yield i+1, j
    if j > 0:       yield i, j-1
    if j+1 < w:     yield i, j+1

def bfs_4(img, start, visited):
    """BFS 4-conexa desde 'start' (solo si start es 1). Devuelve lista de coordenadas."""
    (si, sj) = start
    if img[si, sj] != 1 or visited[si, sj]:
        return []

    h, w = img.shape
    q = deque([(si, sj)])
    visited[si, sj] = True
    coord = [(si, sj)]

    while q:
        ci, cj = q.popleft()
        for ni, nj in vecinos(ci, cj, h, w):
            if img[ni, nj] == 1 and not visited[ni, nj]:
                visited[ni, nj] = True
                q.append((ni, nj))
                coord.append((ni, nj))
    return coord

def match_pattern(block, E, I):
    """Devuelve ('E' o 'I', índice_rot) si block=2x2 coincide; si no, None."""
    for k in range(4):
        if np.array_equal(block, E[k]): return ('E', k)
    for k in range(4):
        if np.array_equal(block, I[k]): return ('I', k)
    return None

def find_holes_via_corners(img):
    """
    1) Recorre parches 2x2, detecta E/I.
    2) Por cada coincidencia, toma como pivote los píxeles ==1 del parche (en coords absolutas).
    3) BFS 4-conexa desde esos pivotes (evitando repetidos con 'visited').
    4) Devuelve lista de regiones (coords), contadores E/I y máscara visitada.
    """
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)

    holes = []        # lista de listas de coords
    E_count = 0
    I_count = 0

    for i in range(h-1):
        for j in range(w-1):
            block = np.array([[img[i, j],     img[i, j+1]],
                               [img[i+1, j],   img[i+1, j+1]]], dtype=np.uint8)

            m = match_pattern(block)
            if m is None:
                continue

            if m[0] == 'E': E_count += 1
            else:           I_count += 1

            # Seeds = todos los '1' del bloque, en coordenadas absolutas
            ones_local = np.argwhere(block == 1)
            for (di, dj) in ones_local:
                si, sj = i + di, j + dj
                if not visited[si, sj] and img[si, sj] == 1:
                    coords = bfs_4(img, (si, sj), visited)
                    if coords:   # puede estar vacío si ya visitado
                        holes.append(coords)

    # Fallback: por si quedaron islas 1 sin haber tenido esquina 2x2 (caso degenerado)
    for i in range(h):
        for j in range(w):
            if img[i, j] == 1 and not visited[i, j]:
                coords = bfs_4(img, (i, j), visited)
                if coords:
                    holes.append(coords)

    return holes, E_count, I_count

def save_holes_as_images(holes, shape, out_prefix="hueco"):
    saved = []
    for idx, coords in enumerate(holes, start=1):
        mask = np.zeros(shape, dtype=np.uint8)
        for (i, j) in coords:
            mask[i, j] = 255
        fname = f"{out_prefix}_{idx}.png"
        cv2.imwrite(fname, mask)
        saved.append(fname)
    return saved
  
def apply_cornerHarris(og, bls, ks, k):
    gr2 = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gr2 = np.float32(gr2)

    dst = cv2.cornerHarris(gr2, bls, ks, k)
    dst = cv2.dilate(dst, None)

    # Umbral: pixels que son esquinas
    mask = dst > 0.075 * dst.max()

    # Obtener coordenadas (y,x) -> (fila,columna)
    coords = np.argwhere(mask)

    vis = og.copy()
    vis[mask] = [0, 0, 255]  # pintar esquinas en rojo

    return vis, coords  # imagen con puntos + coordenadas

def apply_shiTomasi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10, useHarrisDetector=False)

    if corners is None:
        return img.copy(), np.empty((0, 2), int)

    corners = np.int0(corners)
    vis = img.copy()

    coords = []
    for i in corners:
        x, y = i.ravel()
        coords.append([x, y])
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # rojo

    return vis, np.array(coords)

def apply_Harris_Subpixel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    _, dst_bin = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_bin = np.uint8(dst_bin)

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_bin)

    # Refinamiento subpíxel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    vis = img.copy()
    coords = []
    for cx, cy in corners:
        coords.append([int(cx), int(cy)])
        cv2.circle(vis, (int(cx), int(cy)), 3, (255, 0, 0), -1)  # verde

    return vis, np.array(coords)

def apply_ShiTomasi_Subpixel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = cv2.goodFeaturesToTrack(
        gray, maxCorners=100, qualityLevel=0.01, minDistance=10, useHarrisDetector=False
    )
    if dst is None:
        return img.copy(), np.empty((0, 2), int)

    centroids = dst.reshape(-1, 2)  # (N,2)

    # Refinamiento subpíxel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    vis = img.copy()
    coords = []
    for cx, cy in corners:
        coords.append([int(cx), int(cy)])
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)  # verde

    return vis, np.array(coords)

def comparar_puntos(coordsA, coordsB, nombreA="A", nombreB="B"):
    """
    Compara dos conjuntos de puntos (x,y) y devuelve el error promedio.
    Usa un árbol KD para encontrar correspondencias más cercanas.
    """
    if len(coordsA) == 0 or len(coordsB) == 0:
        print(f"No se pudo comparar {nombreA} con {nombreB}: conjuntos vacíos")
        return None

    tree = cKDTree(coordsB)  # estructura rápida de búsqueda
    dists, idxs = tree.query(coordsA, k=1)  # distancia al más cercano en B

    error_prom = np.mean(dists)
    error_max = np.max(dists)
    error_min = np.min(dists)

    print(f"Comparación {nombreA} vs {nombreB}:")
    print(f"   Error promedio: {error_prom:.2f} px")
    print(f"   Error mínimo:   {error_min:.2f} px")
    print(f"   Error máximo:   {error_max:.2f} px")
    print(f"   Total comparados: {len(dists)}\n")

    return dists

def hough_transform(img, edge_img, theta_res=1, rho_res=1, threshold=190):
    """
   Entradas:
        img        : Imagen original en BGR
        edge_img   : Imagen binaria de bordes (ej. salida de Canny)
        theta_res  : resolución angular en grados (default=1°)
        rho_res    : resolución en píxeles (default=1 px)
        threshold  : umbral de número mínimo de votos para aceptar una línea

    Retorna:
        img_lines  : Imagen con las líneas dibujadas
        accumulator: matriz de votos (p, θ)
        thetas     : array de ángulos usados
        rhos       : array de valores de p usados
    """

    # 1. Definir espacio de parámetros
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    width, height = edge_img.shape
    diag_len = int(np.ceil(np.sqrt(width**2 + height**2)))  # diagonal
    rhos = np.arange(-diag_len, diag_len, rho_res)

    # 2. Crear acumulador (ρ, θ)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # 3. Encontrar índices de píxeles de borde
    y_idxs, x_idxs = np.nonzero(edge_img)  # coordenadas de bordes

    # 4. Votar en el espacio ρ-θ
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, t_idx] += 1

    # 5. Dibujar líneas con votos suficientes
    img_lines = img.copy()
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] > threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]

                # Convertir de (ρ, θ) a puntos en imagen
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return img_lines, accumulator, thetas, rhos

def max_pooling_lineas_negras(image, pool_size=(2,2), stride=2):
    h_out = (image.shape[0] - pool_size[0]) // stride + 1
    w_out = (image.shape[1] - pool_size[1]) // stride + 1
    pooled = np.zeros((h_out, w_out), dtype=np.uint8)

    for i in range(h_out):
        for j in range(w_out):
            window = image[i*stride:i*stride+pool_size[0], j*stride:j*stride+pool_size[1]]
            # si hay algún negro (0) en la ventana, mantenerlo negro
            pooled[i, j] = 0 if np.any(window == 0) else 255

    return pooled

def apply_SIFT(img, n_features=0):
    """
    Aplica SIFT con OpenCV a una imagen.
    
    Parámetros:
        img : imagen en BGR (cv2.imread)
        n_features : número máximo de features (0 = sin límite)

    Retorna:
        img_sift : imagen con los keypoints dibujados
        coords   : array Nx2 con coordenadas (x, y) de los puntos
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crear detector SIFT
    sift = cv2.SIFT_create(n_features)

    # Detectar puntos clave y descriptores
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Dibujar keypoints en la imagen
    img_sift = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Extraer coordenadas (x, y)
    coords = np.array([kp.pt for kp in keypoints], dtype=int)

    return img_sift, coords

def fast_detect_manual(img, threshold=20, contiguous=12, nms=True, resize_for_speed=None):
    """
    Detección FAST manual (versión didáctica).
    - img: BGR o grayscale (numpy array)
    - threshold: umbral t para comparar vecinos con Ip +/- t
    - contiguous: número mínimo de vecinos consecutivos (ej. 12)
    - nms: aplicar supresión de no-máximos local (vecindad 3x3)
    - resize_for_speed: (w,h) opcional tuple para reducir imagen antes de procesar
    
    Devuelve: vis_img (BGR), coords (Nx2 array int), scores (N array float)
    """
    # permitir redimensionar externamente (opcional)
    if resize_for_speed is not None:
        img = cv2.resize(img, resize_for_speed, interpolation=cv2.INTER_AREA)

    # asegurar imagen en escala de grises uint8
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.uint8)

    h, w = gray.shape

    # Offsets del círculo de 16 píxeles (dx,dy) ordenado circularmente (clockwise).
    offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0),  (3, 1),  (2, 2),  (1, 3),
        (0, 3),  (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1),(-2, -2),(-1, -3)
    ]

    # índices usados en la comprobación acelerada: posiciones 1,5,9,13 -> indices 0,4,8,12
    accel_idx = [0, 4, 8, 12]

    scores_img = np.zeros((h, w), dtype=np.float32)
    candidates = []

    # recorrer píxeles (evitar bordes de 3 píxeles)
    margin = 3
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            Ip = int(gray[y, x])

            # calcular etiquetas de los 16 vecinos: 1 (más brillante), -1 (más oscuro), 0 (sin evidencia)
            labels = []
            for dx, dy in offsets:
                Ik = int(gray[y + dy, x + dx])
                if Ik > Ip + threshold:
                    labels.append(1)
                elif Ik < Ip - threshold:
                    labels.append(-1)
                else:
                    labels.append(0)

            # Chequeo acelerado: si entre las 4 posiciones no hay evidencia suficiente -> descartar
            pos_count = sum(1 for i in accel_idx if labels[i] == 1)
            neg_count = sum(1 for i in accel_idx if labels[i] == -1)
            if not (pos_count >= 3 or neg_count >= 3):
                continue

            # Test de segmento: buscar segmento contiguo (circular) de longitud >= contiguous
            lbl = labels
            lbl2 = lbl + lbl  # duplicar para manejar envoltura
            found = False
            for sign in (1, -1):
                run = 0
                for v in lbl2:
                    if v == sign:
                        run += 1
                        if run >= contiguous:
                            found = True
                            break
                    else:
                        run = 0
                if found:
                    break
            if not found:
                continue

            # calcular "score" (medida simple de fuerza de la esquina)
            diffs = [abs(int(gray[y + dy, x + dx]) - Ip) for (dx, dy), lab in zip(offsets, labels) if lab != 0]
            score = float(sum(diffs)) if diffs else 0.0

            candidates.append((x, y, score))
            scores_img[y, x] = score

    # Supresión de no-máximos simple: conservar solo puntos que son máximo local 3x3
    final_points = []
    if nms and candidates:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dil = cv2.dilate(scores_img, kernel)
        ys, xs = np.nonzero((scores_img > 0) & (scores_img == dil))
        for yy, xx in zip(ys, xs):
            final_points.append((int(xx), int(yy), float(scores_img[yy, xx])))
    else:
        final_points = candidates

    # Dibujar en imagen de salida (si era grayscale lo convertimos a BGR)
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    coords = []
    scores = []
    for x, y, s in final_points:
        coords.append([x, y])
        scores.append(s)
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), 1)  # marca roja para FAST

    if len(coords) == 0:
        coords = np.empty((0, 2), dtype=int)
        scores = np.empty((0,), dtype=float)
    else:
        coords = np.array(coords, dtype=int)
        scores = np.array(scores, dtype=float)

    return vis, coords, scores

def sift_describe_at_points(img, coords, size=11, draw=True):
    """
    Dado img (BGR) y coords Nx2 int (x,y), crea KeyPoints y obtiene descriptores SIFT.
    Retorna: img_kp (imagen con keypoints dibujados si draw=True), keypoints_coords (Nx2 int), descriptors (Nx128 float)
    """
    if coords is None or len(coords) == 0:
        # nada que describir
        if draw:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), coords, None
            else:
                return img.copy(), coords, None
        else:
            return None, coords, None

    # asegurar gris para computar descriptores
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = gray.astype(np.uint8)

    # construir keypoints desde coords
    keypoints = [cv2.KeyPoint(float(x), float(y), float(size)) for (x, y) in coords]

    # crear SIFT y computar descriptores en esos keypoints
    sift = cv2.SIFT_create()
    keypoints_out, descriptors = sift.compute(gray, keypoints)  # keypoints_out pueden haber sido ajustados

    # extraer coordenadas finales (float->int)
    kp_coords = np.array([[int(k.pt[0]), int(k.pt[1])] for k in keypoints_out], dtype=int)

    # imagen con keypoints dibujados (opcional)
    if draw:
        img_kp = cv2.drawKeypoints(img, keypoints_out, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        img_kp = None

    return img_kp, kp_coords, descriptors

def negate(B):
    """
    Convierte imagen binaria en -1 (foreground) y 0 (background).
    """
    LB = np.where(B == 1, -1, 0)
    return LB

def negate_color(B, background=[0,0,0]):
    """
    Convierte una imagen a color en una matriz de etiquetas inicial.
    - Los píxeles de fondo (background) se quedan en 0.
    - Los píxeles distintos se convierten en -1 (no procesados).
    """
    mask = np.all(B == background, axis=-1)  # True donde es fondo
    LB = np.where(mask, 0, -1)
    return LB

def neighbors(LB, r, c):
    """
    Devuelve los vecinos 4-conexos válidos de (r,c).
    """
    vecinos = []
    filas, cols = LB.shape

    # Arriba
    if r-1 >= 0:
        vecinos.append((r-1, c))
    # Abajo
    if r+1 < filas:
        vecinos.append((r+1, c))
    # Izquierda
    if c-1 >= 0:
        vecinos.append((r, c-1))
    # Derecha
    if c+1 < cols:
        vecinos.append((r, c+1))

    return vecinos

def search(LB, r, c, etiqueta):
    """
    Marca recursivamente el componente conexo al que pertenece (r,c).
    """
    LB[r, c] = etiqueta
    for nr, nc in neighbors(LB, r, c):
        if LB[nr, nc] == -1:
            search(LB, nr, nc, etiqueta)

#Busqueda de conexiones conexas con recursividad
def label_components_rec(B):
    """
    Etiquetado de componentes conexos en una imagen binaria con vecindad-4.
    """
    LB = negate(B)
    etiqueta = 0

    filas, cols = LB.shape
    for r in range(filas):
        for c in range(cols):
            if LB[r, c] == -1:  # píxel no procesado
                etiqueta += 1
                search(LB, r, c, etiqueta)

    return LB

def label_components_color_rec(B, background=[0,0,0]):
    """
    Etiquetado de componentes conexos en una imagen a color con vecindad-4.
    """
    LB = negate_color(B, background)
    etiqueta = 0

    filas, cols = LB.shape
    for r in range(filas):
        for c in range(cols):
            if LB[r, c] == -1:  # píxel no procesado
                etiqueta += 1
                search(LB, r, c, etiqueta)

    return LB

#Busqueda de conexiones conexas con bfs iterativa usando colas
def label_components(img_bin):
    """
    Etiquetado de componentes conectadas usando BFS (4-conexa).
    Entrada: img_bin = imagen binaria (0/1)
    Salida:  LB = matriz con etiquetas enteras (0 = fondo, 1..N = objetos)
    """
    h, w = img_bin.shape
    LB = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros_like(img_bin, dtype=bool)

    etiqueta = 0
    for i in range(h):
        for j in range(w):
            if img_bin[i, j] == 1 and not visited[i, j]:
                etiqueta += 1
                # BFS desde este píxel
                coords = bfs_4(img_bin, (i, j), visited)
                for (r, c) in coords:
                    LB[r, c] = etiqueta
    return LB, etiqueta

def contar_pixeles(LB):
    """Cuenta píxeles de componentes (etiquetas > 0)."""
    return np.sum(LB > 0)

def variacion_porcentual(pix_ini, pix_fin):
    """Calcula variación porcentual."""
    if pix_ini == 0:   # evita división por cero
        return 0
    return (pix_fin - pix_ini) / pix_ini * 100

def calcular_cambio(bin1, bin2):

    # Etiquetamos regiones en 2013
    LB2013, n_labels = label_components(bin1)

    cambios_por_region = {}
    total_pix_og = 0
    total_pix_cambiados = 0

    for etiqueta in range(1, n_labels+1):
        # máscara de la región en 2013
        mask_region = (LB2013 == etiqueta)

        pix_total = np.sum(mask_region)
        if pix_total == 0:
            continue

        # pixeles en 2016 que ya no cumplen condición
        pix_cambiados = np.sum(mask_region & (bin2 == 0))

        cambios_por_region[etiqueta] = (pix_cambiados / pix_total) * 100

        total_pix_og += pix_total
        total_pix_cambiados += pix_cambiados

    cambio_global = (total_pix_cambiados / total_pix_og) * 100

    return cambios_por_region, cambio_global

def analizar_cambio_reservas(img_a, img_b):
    """
    Compara dos imágenes binarias (0 y 1).
    Devuelve una tupla (tendencia, diferencia_porcentual):
      - tendencia: 'Aumento', 'Reducción' o 'Sin cambio'
      - diferencia_porcentual: porcentaje de cambio relativo
    """
    total_a = np.sum(img_a)
    total_b = np.sum(img_b)

    # Asegurar tipo flotante para evitar overflow
    dif = ((np.float64(total_b) - np.float64(total_a)) / np.float64(total_a)) * 100 if total_a != 0 else 0.0

    if total_a == 0:
        return "Sin referencia", 0.0

    if dif > 0:
        tendencia = "Aumento de reservas"
    elif dif < 0:
        tendencia = "Reducción de reservas"
    else:
        tendencia = "Sin cambio"

    return tendencia, dif

def mostrar_resultado_rios(nombre, img_a, img_b, cambio):
    tendencia, dif_porcentual = analizar_cambio_reservas(img_a, img_b)
    print(f"\nCambio global de las reservas por region en {nombre}: {cambio:.2f}%")
    print(f"Hubo: {tendencia} \n")
    
class UnionFind:
    def __init__(self, max_labels):
        self.parent = np.zeros(max_labels + 1, dtype=int)  # índice 0 reservado
        self.rank = np.zeros(max_labels + 1, dtype=int)

    def make_set(self, x):
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x):
        # Búsqueda con compresión de caminos
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Unión por rango para mantener árboles planos
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def vecinos_anteriores(L, r, c, conectividad=8):
    """ Devuelve las etiquetas de los vecinos ya procesados (izquierda, arriba, etc.) """
    vecinos = []
    filas, cols = L.shape

    if conectividad == 4:
        coords = [(-1, 0), (0, -1)]  # norte y oeste
    else:  # conectividad-8
        coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for dr, dc in coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < filas and 0 <= nc < cols and L[nr, nc] != 0:
            vecinos.append(L[nr, nc])

    return vecinos

def label_components_union_find(binaria, conectividad=8):
    """
    Etiqueta componentes conexas en una imagen binaria usando el método
    de Rosenfeld y Pfaltz (1966) con Union-Find (dos pasadas).
    """
    filas, cols = binaria.shape
    L = np.zeros_like(binaria, dtype=int)
    etiqueta = 1

    # Estimación máxima de etiquetas posibles (cada píxel podría ser único)
    uf = UnionFind(filas * cols)

    # Primera pasada
    for r in range(filas):
        for c in range(cols):
            if binaria[r, c] == 0:
                continue

            vecinos = vecinos_anteriores(L, r, c, conectividad)

            if not vecinos:
                # Nueva etiqueta
                L[r, c] = etiqueta
                uf.make_set(etiqueta)
                etiqueta += 1
            else:
                # Asignar la menor etiqueta vecina
                etiqueta_min = min(vecinos)
                L[r, c] = etiqueta_min

                # Registrar equivalencias
                for v in vecinos:
                    uf.union(etiqueta_min, v)

    # Segunda pasada
    for r in range(filas):
        for c in range(cols):
            if L[r, c] != 0:
                L[r, c] = uf.find(L[r, c])

    # Normalizar etiquetas consecutivas
    etiquetas_unicas = np.unique(L[L > 0])
    nueva_L = np.zeros_like(L)
    for i, etiq in enumerate(etiquetas_unicas, start=1):
        nueva_L[L == etiq] = i

    return nueva_L, len(etiquetas_unicas)

class Nodo:
    def __init__(self, coord):
        self.coord = coord
        self.hijos = []

    def add_child(self, nodo):
        self.hijos.append(nodo)

    def __repr__(self):
        return f"Nodo({self.coord}, hijos={len(self.hijos)})"

def color_similar_rgb(c1, c2, umbral=40):
    """Compara dos colores RGB (0-255)."""
    return np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float)) < umbral

def color_similar_hsi(c1, c2, thr_H=20, thr_S=0.2, thr_I=0.2):
    """Compara dos colores HSI (H[0-360], S,I[0-1])."""
    dH = min(abs(c1[0] - c2[0]), 360 - abs(c1[0] - c2[0]))  # diferencia circular de tono
    dS = abs(c1[1] - c2[1])
    dI = abs(c1[2] - c2[2])
    return (dH < thr_H) and (dS < thr_S) and (dI < thr_I)

def crecer_componente(img, seed, umbral=40, modo='rgb'):
    """
    Crece una componente conexa desde una semilla (x,y).
    Retorna:
        - lista de píxeles visitados [(y,x)]
        - árbol de nodos (Nodo raíz)
        - imagen con la componente marcada
    """
    h, w, _ = img.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    x0, y0 = seed
    root = Nodo((y0, x0))

    if modo == 'rgb':
        color_ref = img[y0, x0, :].astype(np.float32)
    else:
        hsi_img = rgb_image_to_hsi(img)
        color_ref = hsi_img[y0, x0, :].astype(np.float32)

    visited[y0, x0] = True
    q.append(root)

    coords = [(y0, x0)]

    while q:
        nodo_actual = q.popleft()
        cy, cx = nodo_actual.coord

        for ny, nx in vecinos(cy, cx, h, w):
            if not visited[ny, nx]:
                if modo == 'rgb':
                    color = img[ny, nx, :].astype(np.float32)
                    similar = color_similar_rgb(color, color_ref, umbral)
                else:
                    color = hsi_img[ny, nx, :].astype(np.float32)
                    similar = color_similar_hsi(color, color_ref)

                if similar:
                    visited[ny, nx] = True
                    nuevo = Nodo((ny, nx))
                    nodo_actual.add_child(nuevo)
                    q.append(nuevo)
                    coords.append((ny, nx))

    # Dibujar resultado
    img_res = img.copy()
    for (yy, xx) in coords:
        img_res[yy, xx] = [0, 255, 0]  # verde: componente crecida

    return coords, img_res, root

def crecer_componentes_paralelo(img, seeds, modo='rgb', umbral=40):
    """
    Ejecuta el crecimiento desde varias semillas en paralelo.
    Retorna lista de resultados [(coords, img_parcial, arbol), ...]
    """
    resultados = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(crecer_componente, img.copy(), seed, umbral, modo)
            for seed in seeds
        ]
        for fut in concurrent.futures.as_completed(futures):
            resultados.append(fut.result())
    return resultados

def combinar_resultados(img, resultados):
    img_final = img.copy()
    for coords, _, _ in resultados:
        for (y, x) in coords:
            img_final[y, x] = [0, 255, 0]
    return img_final

def start_cascade_detect():
    
    haar_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(
        os.path.join(haar_path, 'haarcascade_frontalface_alt.xml')
    )

    eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')
    
    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')
    if eye_cascade.empty():
         raise IOError('Unable to load the eye cascade classifier xml file')
  
 
    return face_cascade, eye_cascade