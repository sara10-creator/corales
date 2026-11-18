# Coral verdadera vs falsa coral: PDI clasico (mismo codigo del notebook de rembg)
# Se ejecuta en el navegador con Pyodide.

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

NEGRO, ROJO, AMARILLO, BLANCO = 1, 2, 3, 4
NOMBRES = {NEGRO: "Negro", ROJO: "Rojo", AMARILLO: "Amarillo", BLANCO: "Blanco"}

# False: el caso "una franja negra y el rojo toca al negro" se clasifica como falsa coral.
# True: ese caso queda "indeterminada (se trata como verdadera)", que evita errores peligrosos
#       a costa de mas falsas alarmas.
MODO_SEGURO = False


def elipse(k):
    k = int(max(3, k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def redimensionar(imagen, lado_max=1024):
    alto, ancho = imagen.shape[:2]
    escala = lado_max / max(alto, ancho)
    if escala < 1:
        imagen = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)),
                            interpolation=cv2.INTER_AREA)
    return imagen


def reducir_ruido(imagen):
    return cv2.bilateralFilter(imagen, 9, 75, 75)


def preparar_imagen(imagen):
    # Preprocesamiento: solo el tamano (maximo 1024 px). No toca los colores.
    return redimensionar(imagen)


def balance_blancos(imagen, cuerpo):
    # Correccion de color DESPUES de segmentar: mundo gris calculado solo con el fondo
    # (pixeles fuera de la serpiente que son casi neutros: suelo, rocas, hojarasca seca).
    # Asi el color de la serpiente no influye en la correccion.
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    fondo = cv2.erode(255 - cuerpo, elipse(15)) > 0
    neutros = fondo & (hsv[..., 1] < 60) & (hsv[..., 2] > 40) & (hsv[..., 2] < 240)

    if neutros.sum() < 0.02 * neutros.size:
        return imagen

    medias = imagen[neutros].astype(np.float32).mean(axis=0)
    factores = np.clip(medias.mean() / np.maximum(medias, 1.0), 0.8, 1.25)

    return np.clip(imagen.astype(np.float32) * factores, 0, 255).astype(np.uint8)


def recuperar_anillos_claros(imagen, cuerpo):
    # Si la segmentacion deja fuera algun anillo claro (fondo claro, arena), se recupera:
    # Se recuperan los huecos que un cierre del tamano del cuerpo rellena y que son
    # blancos o amarillentos (antes solo se aceptaban blancos).
    if np.count_nonzero(cuerpo) == 0:
        return cuerpo
    lado = max(cuerpo.shape[:2])
    grosor, _ = grosor_del_cuerpo(cuerpo)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    H, S, V = [c.astype(np.int32) for c in cv2.split(hsv)]
    claro = ((S < 75) & (V >= 150)) | ((H >= 14) & (H <= 40) & (S >= 60) & (V >= 120))
    k = float(np.clip(1.2 * grosor, 0.02 * lado, 0.06 * lado))
    cerrado = cv2.morphologyEx(cuerpo, cv2.MORPH_CLOSE, elipse(k))
    salida = cuerpo.copy()
    salida[(cerrado > 0) & (cuerpo == 0) & claro] = 255
    return salida



# =====================================================
# SEGMENTACION CON REMBG (modelo U2-Net ligero, u2netp)
# =====================================================
# En la pagina, la red corre en el navegador con onnxruntime-web.
# Aqui se hace lo mismo que rembg antes y despues de la red:
#   entrada_u2net: foto -> 320x320, normalizada como en rembg
#   mascara_desde_u2net: salida de la red -> mascara del tamano de la foto, umbral 127

MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIACION = np.array([0.229, 0.224, 0.225], np.float32)


def entrada_u2net(imagen):
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    pequena = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    pequena = pequena / max(float(pequena.max()), 1e-6)
    pequena = (pequena - MEDIA) / DESVIACION
    return np.ascontiguousarray(pequena.transpose(2, 0, 1)[None], dtype=np.float32)


def mascara_desde_u2net(prediccion, imagen):
    pred = np.asarray(prediccion, np.float32).reshape(320, 320)
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    alfa = cv2.resize((pred * 255).astype(np.uint8), (imagen.shape[1], imagen.shape[0]),
                      interpolation=cv2.INTER_LANCZOS4)
    mascara = ((alfa > 127) * 255).astype(np.uint8)
    return recuperar_anillos_claros(imagen, mascara)


# =====================================================
# LIMPIEZA DE LA MASCARA
# =====================================================

def grosor_del_cuerpo(cuerpo):
    # Borde de ceros: evita distancias infinitas si la mascara llega al borde de la imagen
    con_borde = cv2.copyMakeBorder(cuerpo, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distancia = cv2.distanceTransform(con_borde, cv2.DIST_L2, 5)[1:-1, 1:-1]
    esqueleto = skeletonize(cuerpo > 0)
    if esqueleto.sum() == 0:
        return 1.0, esqueleto
    grosor = float(2 * np.median(distancia[esqueleto]))
    if not np.isfinite(grosor) or grosor < 1:
        grosor = 1.0
    return grosor, esqueleto


def cerrar_y_limpiar(mascara):
    if np.count_nonzero(mascara) == 0:
        return mascara

    lado = max(mascara.shape[:2])
    grosor, _ = grosor_del_cuerpo(mascara)

    # Cierre: une los tramos que la segmentacion corta en los anillos claros.
    # Elipse de 0,3 veces el grosor (con tope), para no unir espiras vecinas.
    k = min(0.3 * grosor, 0.03 * lado)
    cuerpo = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, elipse(k))

    # Apertura pequena: quita puntas y pixeles sueltos del borde
    cuerpo = cv2.morphologyEx(cuerpo, cv2.MORPH_OPEN, elipse(k * 0.5))

    # Conservar solo las regiones grandes
    n, comp, est, _ = cv2.connectedComponentsWithStats(cuerpo, connectivity=8)
    if n <= 1:
        return cuerpo
    areas = est[1:, cv2.CC_STAT_AREA]
    limpio = np.zeros_like(cuerpo)
    for i, area in enumerate(areas, start=1):
        if area >= 0.15 * areas.max():
            limpio[comp == i] = 255

    # Rellenar solo los huecos pequenos (los grandes son espacio real entre espiras)
    hueco_maximo = min((0.6 * grosor) ** 2, 0.002 * limpio.size)
    n, comp, est, _ = cv2.connectedComponentsWithStats(255 - limpio, connectivity=4)
    for i in range(1, n):
        if est[i, cv2.CC_STAT_AREA] < hueco_maximo:
            limpio[comp == i] = 255

    return limpio


# =====================================================
# COLOR DE CADA PIXEL
# =====================================================

def umbral_otsu(histograma):
    total = histograma.sum()
    if total == 0:
        return None
    niveles = np.arange(len(histograma))
    peso1 = np.cumsum(histograma)
    suma1 = np.cumsum(histograma * niveles)
    peso2 = total - peso1
    media1 = suma1 / np.maximum(peso1, 1)
    media2 = (suma1[-1] - suma1) / np.maximum(peso2, 1)
    varianza = peso1 * peso2 * (media1 - media2) ** 2
    return int(np.argmax(varianza))


def reglas_hsv(imagen, cuerpo):
    # Reglas excluyentes: cada pixel del cuerpo recibe un unico color
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    H, S, V = [c.astype(np.int32) for c in cv2.split(hsv)]

    # Tono desplazado: los rojos (H 160-180 y 0-16) quedan juntos en 0-36,
    # los naranjas y amarillos siguen hacia 37-65
    h = (H + 20) % 180
    calido = h <= 65
    rojo_oscuro = calido & (h <= 36) & (S >= 120) & (V >= 50)

    # Negro: muy oscuro, o gris oscuro. Un rojo saturado en sombra NO es negro.
    negro = (V < 50) | ((S < 70) & (V < 150)) | ((V < 85) & ~rojo_oscuro)
    blanco = ~negro & (S < 75)
    candidatos = ~negro & ~blanco & calido

    # Corte rojo/amarillo adaptado a cada foto (Otsu sobre el tono de los pixeles calidos).
    # Si no hay dos grupos de tono claros, se usa el corte fijo H = 16.
    corte = 36
    valores = h[candidatos & (cuerpo > 0)]
    if valores.size >= 200:
        histograma = np.bincount(np.clip(valores, 0, 65), minlength=66).astype(np.float64)
        t = umbral_otsu(histograma)
        if t is not None:
            bajos, altos = valores[valores <= t], valores[valores > t]
            if (bajos.size and altos.size and altos.mean() - bajos.mean() >= 12
                    and 28 <= t <= 44):
                corte = t

    rojo = candidatos & (h <= corte)
    amarillo = candidatos & (h > corte)

    etiquetas = np.zeros(cuerpo.shape, np.uint8)
    dentro = cuerpo > 0
    etiquetas[dentro & negro] = NEGRO
    etiquetas[dentro & blanco] = BLANCO
    etiquetas[dentro & rojo] = ROJO
    etiquetas[dentro & amarillo] = AMARILLO

    return etiquetas


def anchos_de_franjas(mascara, area_minima):
    # Ancho de cada franja = diametro del circulo inscrito (transformada de distancia)
    mascara = mascara.astype(np.uint8)
    n, comp, est, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    distancia = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    return [2 * float(distancia[comp == i].max())
            for i in range(1, n) if est[i, cv2.CC_STAT_AREA] >= area_minima]


def estimar_escala(etiquetas, cuerpo, grosor):
    # Si la serpiente esta enrollada, la mascara es una mancha y su grosor no es el del cuerpo.
    # Por eso la escala se toma tambien del ancho tipico de las franjas negras.
    negro = cv2.morphologyEx((etiquetas == NEGRO).astype(np.uint8), cv2.MORPH_OPEN, elipse(3))
    anchos = anchos_de_franjas(negro, max(20, 0.001 * np.count_nonzero(cuerpo)))

    if not anchos:
        return grosor

    return float(max(4.0, min(grosor, 1.6 * np.median(anchos))))


def etiquetar_colores(imagen, cuerpo, grosor):
    crudas = reglas_hsv(imagen, cuerpo)
    escala = estimar_escala(crudas, cuerpo, grosor)

    etiquetas = filtro_de_mayoria(crudas, cuerpo, escala * 0.12)
    etiquetas = absorber_regiones_pequenas(etiquetas, cuerpo, (escala * 0.25) ** 2)

    return etiquetas, escala


def filtro_de_mayoria(etiquetas, cuerpo, k):
    k = int(max(3, k)) | 1
    votos = []
    for color in NOMBRES:
        votos.append(cv2.blur((etiquetas == color).astype(np.float32), (k, k)))
    ganador = np.argmax(np.stack(votos), axis=0) + 1
    salida = ganador.astype(np.uint8)
    salida[cuerpo == 0] = 0
    return salida


def absorber_regiones_pequenas(etiquetas, cuerpo, area_minima):
    # Brillos, sombras y pixeles sin color definido: se reemplazan por el color vecino mas cercano
    validos = np.zeros(etiquetas.shape, np.uint8)
    for color in NOMBRES:
        mascara = (etiquetas == color).astype(np.uint8)
        n, comp, est, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        for i in range(1, n):
            if est[i, cv2.CC_STAT_AREA] >= area_minima:
                validos[comp == i] = 1

    if validos.sum() == 0:
        return etiquetas

    _, indice = cv2.distanceTransformWithLabels(
        1 - validos, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(validos)
    tabla = np.zeros(indice.max() + 1, np.uint8)
    tabla[indice[ys, xs]] = etiquetas[ys, xs]

    salida = tabla[indice]
    salida[cuerpo == 0] = 0
    return salida


def medir_colores(etiquetas, cuerpo, escala):
    # Fraccion del cuerpo de cada color, numero de anillos y ancho tipico de las franjas
    total = max(1, np.count_nonzero(cuerpo))
    area_minima = (escala * 0.2) ** 2

    claro = np.isin(etiquetas, (AMARILLO, BLANCO))
    rojo = etiquetas == ROJO
    negro = etiquetas == NEGRO

    anchos_claro = anchos_de_franjas(claro, area_minima)
    anchos_rojo = anchos_de_franjas(rojo, area_minima)
    anchos_negro = anchos_de_franjas(negro, area_minima)

    return {
        "negro": np.count_nonzero(negro) / total,
        "rojo": np.count_nonzero(rojo) / total,
        "claro": np.count_nonzero(claro) / total,
        "anillos_claros": len(anchos_claro),
        "anillos_rojos": len(anchos_rojo),
        "ancho_claro": float(np.median(anchos_claro)) if anchos_claro else 0.0,
        "ancho_rojo": float(np.median(anchos_rojo)) if anchos_rojo else 0.0,
        "ancho_negro": float(np.median(anchos_negro)) if anchos_negro else 0.0,
    }


# =====================================================
# CONTACTO ENTRE COLORES (dilatacion eliptica + AND)
# =====================================================

def medir_contactos(etiquetas, cuerpo, grosor):
    # Nucleo pequeno: solo cuentan los vecinos inmediatos, para no saltar anillos claros delgados
    nucleo = elipse(grosor * 0.06)
    rojo = (etiquetas == ROJO).astype(np.uint8)
    # Corona alrededor del rojo, solo dentro del cuerpo
    # Se ignora una franja junto al contorno: ahi suele quedar sombra etiquetada como negro
    interior = (cv2.erode(cuerpo, elipse(grosor * 0.25)) > 0).astype(np.uint8)
    corona = cv2.dilate(rojo, nucleo) & (1 - rojo) & interior

    claro = np.count_nonzero(corona & np.isin(etiquetas, (AMARILLO, BLANCO)))
    negro = np.count_nonzero(corona & (etiquetas == NEGRO))
    total = claro + negro

    # Corona alrededor del negro: que proporcion de su borde toca al rojo
    negro_m = (etiquetas == NEGRO).astype(np.uint8)
    corona_negro = cv2.dilate(negro_m, nucleo) & (1 - negro_m) & interior
    negro_rojo = np.count_nonzero(corona_negro & (etiquetas == ROJO))
    negro_claro = np.count_nonzero(corona_negro & np.isin(etiquetas, (AMARILLO, BLANCO)))
    total_negro = negro_rojo + negro_claro

    return {
        "indice_negro_rojo": negro_rojo / total_negro if total_negro else np.nan,
        "contacto_rojo_claro": claro,
        "contacto_rojo_negro": negro,
        "indice_rojo_claro": claro / total if total else np.nan,
        "corona": corona * 255,
    }


# =====================================================
# FRANJAS NEGRAS ENTRE DOS ROJOS (votacion)
# =====================================================

def contar_anillos_por_bloque(etiquetas, cuerpo, grosor):
    area_min_anillo = (grosor * 0.25) ** 2
    area_min_bloque = (grosor * 0.5) ** 2

    no_rojo = ((etiquetas > 0) & (etiquetas != ROJO)).astype(np.uint8)
    n, bloques, est, cent = cv2.connectedComponentsWithStats(no_rojo, connectivity=8)

    rojo_dil = cv2.dilate((etiquetas == ROJO).astype(np.uint8), elipse(grosor * 0.15))

    resultado = []
    for i in range(1, n):
        if est[i, cv2.CC_STAT_AREA] < area_min_bloque:
            continue
        bloque = bloques == i

        # Solo cuentan los bloques encerrados entre rojos (no la cabeza ni la cola)
        toca_rojo = rojo_dil & bloque.astype(np.uint8)
        n_toques, _ = cv2.connectedComponents(cv2.dilate(toca_rojo, elipse(grosor * 0.3)))
        entre_rojos = (n_toques - 1) >= 2

        # Forma A: franjas negras del bloque (componentes conectados)
        negro = (bloque & (etiquetas == NEGRO)).astype(np.uint8)
        _, _, est_n, _ = cv2.connectedComponentsWithStats(negro, connectivity=8)
        negros = int(np.sum(est_n[1:, cv2.CC_STAT_AREA] >= area_min_anillo))

        # Forma B: anillos claros internos, los que NO tocan al rojo (dilatacion + AND).
        # Monadal R-A-N-A-R: 0 internos. Diadas R-N-A-N-R: 1. Triadas: 2.
        claro = (bloque & np.isin(etiquetas, (AMARILLO, BLANCO))).astype(np.uint8)
        m, comp_c, est_c, _ = cv2.connectedComponentsWithStats(claro, connectivity=8)
        internos = 0
        for j in range(1, m):
            if est_c[j, cv2.CC_STAT_AREA] < area_min_anillo:
                continue
            if not np.any(rojo_dil[comp_c == j]):
                internos += 1

        resultado.append({
            "bloque": i,
            "negros": negros,
            "claros_internos": internos,
            "entre_rojos": bool(entre_rojos),
            "centro": (int(cent[i][0]), int(cent[i][1])),
        })
    return resultado


def camino_del_esqueleto(esqueleto):
    # Camino mas largo dentro del esqueleto (eje medial): de una punta a la otra del cuerpo
    ys, xs = np.nonzero(esqueleto)
    if len(xs) < 10:
        return None

    indice = -np.ones(esqueleto.shape, np.int64)
    indice[ys, xs] = np.arange(len(xs))
    alto, ancho = esqueleto.shape

    filas, columnas, pesos = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ny, nx = ys + dy, xs + dx
        valido = (ny < alto) & (nx >= 0) & (nx < ancho)
        vecino = np.full(len(xs), -1)
        vecino[valido] = indice[ny[valido], nx[valido]]
        a = np.nonzero(vecino >= 0)[0]
        filas.append(a)
        columnas.append(vecino[a])
        pesos.append(np.full(len(a), np.hypot(dy, dx)))

    filas, columnas, pesos = np.concatenate(filas), np.concatenate(columnas), np.concatenate(pesos)
    grafo = coo_matrix((pesos, (filas, columnas)), shape=(len(xs), len(xs))).tocsr()

    # Componente mas grande del esqueleto
    _, componente = connected_components(grafo, directed=False)
    principal = np.argmax(np.bincount(componente))
    inicio = int(np.nonzero(componente == principal)[0][0])

    # Dos barridos: el punto mas lejano de cualquier punto es una punta; desde ahi, la otra punta
    distancias = dijkstra(grafo, directed=False, indices=inicio)
    a = int(np.nanargmax(np.where(np.isfinite(distancias), distancias, -1)))
    distancias, previo = dijkstra(grafo, directed=False, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(distancias), distancias, -1)))

    camino = [b]
    while camino[-1] != a and previo[camino[-1]] >= 0:
        camino.append(int(previo[camino[-1]]))

    camino = np.array(camino[::-1])
    longitud = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ys[camino]), np.diff(xs[camino])))])
    return ys[camino], xs[camino], longitud


def leer_perfil(etiquetas, esqueleto, escala):
    # Recorre el eje medial y escribe la secuencia de colores, por ejemplo R C N C R C N C R
    camino = camino_del_esqueleto(esqueleto)
    if camino is None:
        return {"secuencia": "", "negros": [], "valor": None}

    ys, xs, longitud = camino
    colores = etiquetas[ys, xs]
    letra = {NEGRO: "N", ROJO: "R", AMARILLO: "C", BLANCO: "C"}

    # Tramos seguidos del mismo color, con su largo
    tramos = []
    for k in range(len(colores)):
        if colores[k] == 0:
            continue
        paso = longitud[k] - longitud[k - 1] if k > 0 else 0
        c = letra[int(colores[k])]
        if tramos and tramos[-1][0] == c:
            tramos[-1][1] += paso
        else:
            tramos.append([c, paso])

    # Los tramos muy cortos son ruido (brillos, cruces): se eliminan y se vuelven a unir
    minimo = 0.15 * escala
    limpios = []
    for c, largo in tramos:
        if largo < minimo:
            continue
        if limpios and limpios[-1] == c:
            continue
        limpios.append(c)

    secuencia = "".join(limpios)

    # Franjas negras entre cada par de rojos consecutivos
    rojos = [i for i, c in enumerate(limpios) if c == "R"]
    negros = [limpios[i + 1:j].count("N") for i, j in zip(rojos, rojos[1:])]
    negros = [n for n in negros if n > 0]

    valor = None
    if len(negros) >= 2:
        valores, cuentas = np.unique(negros, return_counts=True)
        if cuentas.max() >= 0.6 * len(negros):
            valor = int(min(valores[np.argmax(cuentas)], 3))

    return {"secuencia": secuencia, "negros": negros, "valor": valor}


def moda_confiable(valores, minimo=3):
    # Valor mas repetido, solo si hay suficientes datos y la mayoria coincide
    if len(valores) < minimo:
        return None
    valores_unicos, cuentas = np.unique(valores, return_counts=True)
    if cuentas.max() < 0.6 * len(valores):
        return None
    return int(min(valores_unicos[np.argmax(cuentas)], 3))


def votar_negros(contactos, bloques, perfil):
    # Cuatro formas independientes de saber cuantas franjas negras hay entre dos rojos
    entre = [b for b in bloques if b["entre_rojos"]]
    rc = contactos["indice_rojo_claro"]
    nr = contactos["indice_negro_rojo"]

    votos = {
        "bloques": moda_confiable([b["negros"] for b in entre if b["negros"] > 0]),
        "claros internos": moda_confiable([b["claros_internos"] + 1 for b in entre if b["negros"] > 0]),
        "perfil": perfil["valor"],
    }

    if not np.isnan(rc) and rc >= 0.5:
        votos["contacto"] = 1
    elif np.isnan(nr):
        votos["contacto"] = None
    elif nr >= 0.80:
        votos["contacto"] = 1
    elif nr >= 0.42:
        votos["contacto"] = 2
    elif nr < 0.38:
        votos["contacto"] = 3
    else:
        votos["contacto"] = None

    validos = [v for v in votos.values() if v is not None]
    detalle = ", ".join(f"{k}={v}" for k, v in votos.items() if v is not None)

    if not validos:
        return None, "sin votos"

    valores, cuentas = np.unique(validos, return_counts=True)
    empatados = valores[cuentas == cuentas.max()]

    # Empate: se elige el numero mayor (3 antes que 2), que es la opcion mas segura
    return int(empatados.max()), detalle


# =====================================================
# REGLA DE DECISION
# =====================================================

def es_patron_rabo_de_aji(colores):
    # Cuerpo mayormente negro con anillos delgados: claros (occidente) o rojos (poblaciones
    # andinas). Los anillos deben ser bastante mas angostos que las franjas negras.
    if colores["negro"] < 0.55 or colores["ancho_negro"] == 0:
        return False
    if colores["claro"] + colores["rojo"] < 0.05:
        return False

    anchos = []
    anillos = 0
    if colores["anillos_claros"] >= 2:
        anchos.append(colores["ancho_claro"])
        anillos += colores["anillos_claros"]
    if colores["anillos_rojos"] >= 3:
        anchos.append(colores["ancho_rojo"])
        anillos += colores["anillos_rojos"]

    if anillos < 4 or not anchos:
        return False

    return max(anchos) <= 0.6 * colores["ancho_negro"]


def decidir(contactos, bloques, colores, perfil, modo_seguro=None, umbral_contacto=0.5):

    if modo_seguro is None:
        modo_seguro = MODO_SEGURO

    rc = contactos["indice_rojo_claro"]

    # Paso 0: patron rabo de aji
    if es_patron_rabo_de_aji(colores):
        return ("Coral verdadera (probable)",
                "Cuerpo mayormente negro con anillos delgados (patron rabo de aji)",
                None, "paso 0")

    if colores["rojo"] < 0.03 or np.isnan(rc):
        return "Sin resultado", "No hay franjas rojas suficientes para aplicar la regla", None, None

    # Paso 1: cuantas franjas negras hay entre dos rojos (votacion)
    negros, metodo = votar_negros(contactos, bloques, perfil)

    if negros is None:
        return "Sin resultado", "No se pudo medir cuantos negros hay entre rojos", None, metodo
    if negros == 3:
        return "Coral verdadera (probable)", "Tres franjas negras entre dos rojos", negros, metodo
    if negros == 2:
        return "Falsa coral", "Dos franjas negras entre dos rojos", negros, metodo

    # Paso 2 (una sola franja negra): que es lo que mas toca al rojo
    if rc >= umbral_contacto:
        return ("Coral verdadera (probable)",
                "Una franja negra entre rojos y el rojo toca amarillo o blanco", negros, metodo)

    if modo_seguro:
        return ("Indeterminada (se trata como verdadera)",
                "Una franja negra entre rojos y el rojo parece tocar al negro", negros, metodo)

    return ("Falsa coral",
            "Una franja negra entre rojos y el rojo toca al negro (bicolor)", negros, metodo)


def tratar_como(clase):
    if clase.startswith("Coral verdadera") or clase.startswith("Indeterminada"):
        return "Coral verdadera"
    if clase == "Falsa coral":
        return "Falsa coral"
    return "Sin resultado"


# =====================================================
# ANALISIS COMPLETO
# =====================================================

def quitar_motas_negras(etiquetas, escala):
    # Muchas corales (M. tener, M. fulvius, M. mipartitus) tienen las escamas rojas con la
    # punta negra. Esas motas negras dentro del rojo no son anillos, pero hacen que el rojo
    # "toque negro". Toda mancha negra pequena rodeada casi solo de rojo pasa a ser rojo.
    salida = etiquetas.copy()
    negro = (etiquetas == NEGRO).astype(np.uint8)
    n, comp, est, _ = cv2.connectedComponentsWithStats(negro, connectivity=8)
    area_max = (0.7 * escala) ** 2
    nucleo = elipse(3)
    for i in range(1, n):
        if est[i, cv2.CC_STAT_AREA] > area_max:
            continue
        x, y, w, h = est[i, :4]
        y0, y1, x0, x1 = max(0, y - 4), y + h + 4, max(0, x - 4), x + w + 4
        mota = (comp[y0:y1, x0:x1] == i).astype(np.uint8)
        borde = (cv2.dilate(mota, nucleo) > 0) & (mota == 0)
        vecinos = etiquetas[y0:y1, x0:x1][borde]
        vecinos = vecinos[vecinos > 0]
        if vecinos.size and np.mean(vecinos == ROJO) >= 0.7:
            salida[y0:y1, x0:x1][mota > 0] = ROJO
    return salida


def analizar_con_mascara(imagen, mascara, modo_seguro=None, umbral_contacto=0.5):
    cuerpo = cerrar_y_limpiar(mascara)
    corregida = balance_blancos(imagen, cuerpo)
    suave = reducir_ruido(corregida)
    fraccion = np.count_nonzero(cuerpo) / cuerpo.size

    if fraccion == 0 or fraccion > 0.85:
        return {"imagen": imagen, "corregida": corregida, "cuerpo": cuerpo, "clase": "Sin resultado",
                "patron": "No se pudo separar la serpiente del fondo",
                "negros_entre_rojos": None, "metodo": None, "secuencia": "",
                "indice_rojo_claro": np.nan, "indice_negro_rojo": np.nan}

    grosor, esqueleto = grosor_del_cuerpo(cuerpo)
    etiquetas, escala = etiquetar_colores(suave, cuerpo, grosor)

    # NUEVO: las motas negras dentro de las franjas rojas pasan a ser rojo
    etiquetas = quitar_motas_negras(etiquetas, escala)


    colores = medir_colores(etiquetas, cuerpo, escala)
    contactos = medir_contactos(etiquetas, cuerpo, escala)
    bloques = contar_anillos_por_bloque(etiquetas, cuerpo, escala)
    perfil = leer_perfil(etiquetas, esqueleto, escala)

    clase, patron, negros, metodo = decidir(contactos, bloques, colores, perfil,
                                            modo_seguro, umbral_contacto)
    return {
        "imagen": imagen, "corregida": corregida, "cuerpo": cuerpo, "etiquetas": etiquetas,
        "esqueleto": esqueleto, "escala": escala, "colores": colores, "bloques": bloques,
        "perfil": perfil, "corona": contactos["corona"], "secuencia": perfil["secuencia"],
        "indice_rojo_claro": contactos["indice_rojo_claro"],
        "indice_negro_rojo": contactos["indice_negro_rojo"],
        "negros_entre_rojos": negros, "metodo": metodo,
        "clase": clase, "patron": patron, "se_trata_como": tratar_como(clase),
    }


COLORES_DIBUJO = {0: (0, 0, 0), NEGRO: (70, 70, 70), ROJO: (40, 40, 220),
                  AMARILLO: (0, 215, 255), BLANCO: (255, 255, 255)}


def pintar_etiquetas(etiquetas):
    salida = np.zeros(etiquetas.shape + (3,), np.uint8)
    for color, bgr in COLORES_DIBUJO.items():
        salida[etiquetas == color] = bgr
    return salida


# =====================================================
# REGLA DE DECISION (paso 0, claro >= X, paso 1, paso 2)
# =====================================================


# Umbrales de la regla. Pega aqui los que imprimio la celda de calibrar del notebook.
PARAMS = {
    "claro_min": 0.08,     # X: fraccion de amarillo + blanco que manda al paso 2
    "rojo_poco": 0.06,     # menos rojo que esto es "poco rojo"
    "negro_dom": 0.45,     # para "poco rojo = verdadera" el cuerpo debe ser al menos esto de negro
    "rc": 0.50,            # indice rojo-claro: desde aqui el rojo "toca claro"
    "nr_triada": 0.38,     # indice negro-rojo por debajo de esto parece triada (3 negras)
    "sin_rojo": "Coral verdadera",   # que hacer si en el paso 2 no aparece el rojo
}

ULTIMAS_MEDIDAS = {}


def medidas_para_regla(contactos, bloques, colores, perfil):
    # Todo lo que la regla necesita, en un diccionario de numeros
    entre = [b for b in bloques if b["entre_rojos"]]
    return {
        "rojo": colores["rojo"],
        "claro": colores["claro"],
        "negro": colores["negro"],
        "rabo_de_aji": es_patron_rabo_de_aji(colores),
        "rc": contactos["indice_rojo_claro"],
        "nr": contactos["indice_negro_rojo"],
        "v_bloques": moda_confiable([b["negros"] for b in entre if b["negros"] > 0]),
        "v_claros": moda_confiable([b["claros_internos"] + 1 for b in entre if b["negros"] > 0]),
        "v_perfil": perfil["valor"],
    }


def contar_negras(m, p):
    # Votacion de cuantas franjas negras hay entre dos rojos
    votos = {"bloques": m["v_bloques"], "claros internos": m["v_claros"], "perfil": m["v_perfil"]}

    # Voto por contacto (dilatacion + AND)
    if not np.isnan(m["rc"]) and m["rc"] >= p["rc"]:
        votos["contacto"] = 1
    elif not np.isnan(m["nr"]):
        votos["contacto"] = 3 if m["nr"] < p["nr_triada"] else 2

    validos = [int(v) for v in votos.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    detalle = ", ".join(f"{k}={v}" for k, v in votos.items() if v is not None and not (isinstance(v, float) and np.isnan(v)))
    if not validos:
        return None, "sin votos"

    valores, cuentas = np.unique(validos, return_counts=True)
    empatados = valores[cuentas == cuentas.max()]
    return int(empatados.max()), detalle   # empate: el numero mayor (mas seguro)


def regla_nueva(m, p=None):
    # Devuelve (clase, motivo, negras, paso)
    p = PARAMS if p is None else p
    rc, nr = m["rc"], m["nr"]

    # Paso 0
    if m["rabo_de_aji"]:
        return "Coral verdadera", "Cuerpo negro con anillos delgados (rabo de ají)", None, "paso 0"

    # ---------- PASO 1: poco amarillo/blanco ----------
    if m["claro"] < p["claro_min"]:

        if m["rojo"] < p["rojo_poco"]:
            if m["negro"] >= p["negro_dom"]:
                return "Coral verdadera", "Poco claro, poco rojo y cuerpo negro", None, "paso 1"
            return ("Coral verdadera", "No se distinguen los colores (se trata como verdadera)",
                    None, "paso 1")

        # Mucho rojo: corona del rojo dilatado AND las otras mascaras
        if not np.isnan(rc) and rc >= p["rc"]:
            return "Coral verdadera", "Paso 1: el rojo dilatado toca sobre todo claro", 1, "paso 1"
        return "Falsa coral", "Paso 1: el rojo dilatado toca sobre todo negro (bicolor)", 1, "paso 1"

    # ---------- PASO 2: bastante amarillo/blanco ----------
    if m["rojo"] < 0.03 or np.isnan(rc):
        return (p["sin_rojo"], "Paso 2: hay claro pero casi no se ve rojo", None, "paso 2")

    negras, detalle = contar_negras(m, p)

    if negras == 3:
        return "Coral verdadera", "Paso 2: tres negras entre rojos (triada)", 3, detalle
    if negras == 2:
        return "Falsa coral", "Paso 2: dos negras entre rojos (diada)", 2, detalle
    if negras == 1 or negras is None:
        if rc >= p["rc"]:
            return "Coral verdadera", "Paso 2: una negra y el rojo toca claro (monadal)", 1, detalle
        return "Falsa coral", "Paso 2: una negra y el rojo toca negro", 1, detalle


def decidir(contactos, bloques, colores, perfil, modo_seguro=None, umbral_contacto=0.5):
    # Misma firma que antes: la celda 2 y la celda 4 funcionan sin cambios
    m = medidas_para_regla(contactos, bloques, colores, perfil)
    ULTIMAS_MEDIDAS.clear()
    ULTIMAS_MEDIDAS.update(m)
    clase, motivo, negras, detalle = regla_nueva(m)
    return clase, motivo, negras, detalle


def tratar_como(clase):
    # Sin resultado (la serpiente no se pudo separar) tambien se trata como verdadera
    if clase == "Falsa coral":
        return "Falsa coral"
    return "Coral verdadera"




# =====================================================
# PUNTO DE ENTRADA PARA LA PAGINA WEB
# =====================================================
import base64
import json

_ESTADO = {}


def _png(imagen_bgr, lado_max=640):
    alto, ancho = imagen_bgr.shape[:2]
    escala = min(1.0, lado_max / max(alto, ancho))
    if escala < 1:
        imagen_bgr = cv2.resize(imagen_bgr, (int(ancho * escala), int(alto * escala)),
                                interpolation=cv2.INTER_AREA)
    ok, datos = cv2.imencode(".png", imagen_bgr)
    return "data:image/png;base64," + base64.b64encode(datos.tobytes()).decode()


def _pct(valor):
    return f"{100 * float(valor):.0f} %"


def _num(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return "-"
    return "-" if np.isnan(valor) else f"{valor:.2f}"


def parece_serpiente(r):
    # Verificacion con PDI clasico antes de aplicar las reglas:
    # 1. Tamano: lo que se separo del fondo debe ocupar al menos el 1 % de la imagen.
    # 2. Color: al menos dos colores de coral (negro, rojo, claro) con 5 % o mas del cuerpo.
    # 3. Anillos: al menos 3 franjas negras separadas (componentes conectados) y un recorrido
    #    del eje medial que alterne colores varias veces.
    # El patron rabo de aji (cuerpo casi negro) se acepta directamente.
    cuerpo = r["cuerpo"]
    if np.count_nonzero(cuerpo) < 0.01 * cuerpo.size:
        return False, "Lo que se separó del fondo es demasiado pequeño"

    et = r["etiquetas"]
    if es_patron_rabo_de_aji(r["colores"]):
        return True, ""

    total = max(1, np.count_nonzero(cuerpo))
    negro = np.count_nonzero(et == NEGRO) / total
    rojo = np.count_nonzero(et == ROJO) / total
    claro = np.count_nonzero(np.isin(et, (AMARILLO, BLANCO))) / total
    if sum(v >= 0.05 for v in (negro, rojo, claro)) < 2:
        return False, "No se encontraron los colores de una serpiente coral"

    mascara_negra = (et == NEGRO).astype(np.uint8)
    _, _, est, _ = cv2.connectedComponentsWithStats(mascara_negra, connectivity=8)
    franjas = int(np.sum(est[1:, cv2.CC_STAT_AREA] >= (0.5 * r["escala"]) ** 2))
    if franjas < 3 or len(r.get("secuencia", "")) < 4:
        return False, "No se encontraron anillos de colores a lo largo del cuerpo"
    return True, ""


def web_preparar(rgba, ancho, alto):
    # Paso 1 de PDI: la foto llega del navegador (RGBA), se pasa a BGR y se limita el tamano.
    datos = rgba.to_py() if hasattr(rgba, "to_py") else rgba     # en el navegador llega un Uint8Array
    foto = np.frombuffer(bytes(datos), np.uint8).reshape(int(alto), int(ancho), 4)
    imagen = preparar_imagen(cv2.cvtColor(foto, cv2.COLOR_RGBA2BGR))
    _ESTADO["imagen"] = imagen
    return entrada_u2net(imagen).tobytes()            # float32 de 1x3x320x320 para onnxruntime-web


def web_analizar(prediccion):
    imagen = _ESTADO["imagen"]
    if hasattr(prediccion, "to_py"):                  # en el navegador llega un Float32Array
        prediccion = np.frombuffer(bytes(prediccion.to_py()), np.float32)
    mascara = mascara_desde_u2net(prediccion, imagen)
    ULTIMAS_MEDIDAS.clear()
    r = analizar_con_mascara(imagen, mascara)
    m = dict(ULTIMAS_MEDIDAS)
    alto, ancho = imagen.shape[:2]
    pasos = []

    pasos.append({"titulo": "Preprocesamiento",
                  "pdi": "Solo se cambia el tamaño (lado mayor de 1024 px como máximo). No se tocan los colores.",
                  "dato": f"Imagen de trabajo: {ancho} x {alto} px"})
    fraccion = np.count_nonzero(r["cuerpo"]) / r["cuerpo"].size
    pasos.append({"titulo": "Segmentación (rembg, U2-Net)",
                  "pdi": "La red da una máscara alfa; se umbraliza en 127, se recuperan los anillos claros "
                         "con un cierre morfológico y se limpia con cierre, apertura y componentes conectados.",
                  "dato": f"La serpiente ocupa el {_pct(fraccion)} de la imagen"})

    imagenes = {"original": _png(imagen)}
    vista_cuerpo = imagen.copy()
    vista_cuerpo[r["cuerpo"] == 0] = (vista_cuerpo[r["cuerpo"] == 0] * 0.25).astype(np.uint8)
    imagenes["cuerpo"] = _png(vista_cuerpo)

    if "etiquetas" not in r:
        clase = "Sin resultado"
        pasos.append({"titulo": "Resultado", "pdi": r["patron"],
                      "dato": "Por seguridad se trata como coral verdadera."})
        return json.dumps({"clase": clase, "se_trata_como": "Coral verdadera", "motivo": r["patron"],
                           "pasos": pasos, "imagenes": imagenes, "medidas": {}})

    serpiente, razon = parece_serpiente(r)
    if not serpiente:
        pasos.append({"titulo": "Resultado", "pdi": razon, "dato": "Sin resultado"})
        return json.dumps({"clase": "Sin resultado", "se_trata_como": "Sin resultado", "motivo": razon,
                           "pasos": pasos, "imagenes": imagenes, "medidas": {}})

    et = r["etiquetas"]
    total = max(1, np.count_nonzero(r["cuerpo"]))
    amarillo = np.count_nonzero(et == AMARILLO) / total
    blanco = np.count_nonzero(et == BLANCO) / total

    pasos.append({"titulo": "Corrección de color y máscaras HSV",
                  "pdi": "Balance de blancos (mundo gris) con el fondo. Cada píxel del cuerpo se clasifica en HSV: "
                         "negro (V bajo), blanco (S baja), rojo o amarillo (tono, corte de Otsu). Filtro de mayoría "
                         "y las motas negras dentro del rojo pasan a rojo.",
                  "dato": f"Negro {_pct(m['negro'])} · Rojo {_pct(m['rojo'])} · Amarillo {_pct(amarillo)} · "
                          f"Blanco {_pct(blanco)}"})

    p = PARAMS
    clase, motivo, negras, detalle = regla_nueva(m)

    pasos.append({"titulo": "Paso 0 · ¿Rabo de ají?",
                  "pdi": "Proporción de negro y ancho de franjas con la transformada de distancia: "
                         "¿cuerpo ≥ 55 % negro y anillos ≤ 0,6 veces el ancho del negro?",
                  "dato": "Sí: patrón rabo de ají (Micrurus mipartitus)" if m["rabo_de_aji"] else "No"})

    if not m["rabo_de_aji"]:
        va_paso1 = m["claro"] < p["claro_min"]
        pasos.append({"titulo": "¿Claro ≥ X?",
                      "pdi": "Claro = píxeles amarillos + blancos ÷ píxeles del cuerpo. X es el umbral fijo de la regla.",
                      "dato": f"Claro {_pct(m['claro'])} frente a X = {_pct(p['claro_min'])} → "
                              + ("PASO 1 (bicolor)" if va_paso1 else "PASO 2 (díadas / tríadas / monadal)")})
        if va_paso1:
            pasos.append({"titulo": "Paso 1 · ¿Cuánto rojo y qué lo toca?",
                          "pdi": "Proporción de rojo. Si hay mucho: dilatación elíptica del rojo, se resta el rojo "
                                 "(corona) y AND con negro y con claro.",
                          "dato": f"Rojo {_pct(m['rojo'])} (poco si < {_pct(p['rojo_poco'])}) · "
                                  f"índice rojo–claro {_num(m['rc'])} (toca claro si ≥ {_num(p['rc'])})"})
        else:
            pasos.append({"titulo": "Paso 2 · ¿Cuántas negras entre dos rojos?",
                          "pdi": "Votación de cuatro métodos: bloques (componentes conectados entre rojos), "
                                 "claros internos, perfil del esqueleto y contacto del borde negro.",
                          "dato": (f"Votos: {detalle} → {negras} negra(s)" if negras else str(detalle))
                                  + f" · índice rojo–claro {_num(m['rc'])}"})

    pasos.append({"titulo": "Resultado", "pdi": motivo, "dato": clase})

    vista_corona = r["corregida"].copy()
    vista_corona[r["corona"] > 0] = (0, 255, 0)
    imagenes["colores"] = _png(pintar_etiquetas(et))
    imagenes["corona"] = _png(vista_corona)

    medidas = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
               for k, v in m.items() if k in ("rojo", "claro", "negro", "rc", "nr")}
    medidas = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in medidas.items()}
    return json.dumps({"clase": clase, "se_trata_como": tratar_como(clase), "motivo": motivo,
                       "secuencia": r.get("secuencia", "")[:40], "pasos": pasos,
                       "imagenes": imagenes, "medidas": medidas})
