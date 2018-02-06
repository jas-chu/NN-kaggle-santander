from numpy import genfromtxt, exp, array, random, dot
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
# from bigfloat import BigFloat

# sudo apt-get install libmpfr-dev
# sudo pip install bigfloat

matriz_entrenamiento_input = []
matriz_entrenamiento_output = []
matriz_entrenamiento_cantFILs = 0
matriz_entrenamiento_cantCOLs = 0

matriz_validacion_input = []
matriz_validacion_output = []
matriz_validacion_cantFILs = 0
matriz_validacion_cantCOLs = 0

cant_iteraciones = 0


def read_entrenamiento():
    entrenamiento_csv = genfromtxt('../entrenamientoEqual.csv', delimiter=',', names=None, dtype=float)

    # Cargo info del set de entrenamiento.
    entrenamiento_labels = entrenamiento_csv[0]
    global matriz_entrenamiento_cantFILs
    matriz_entrenamiento_cantFILs = len(entrenamiento_csv)
    global matriz_entrenamiento_cantCOLs
    matriz_entrenamiento_cantCOLs = len(entrenamiento_csv[0])

    # Leo matriz de entrenamiento input sin labels y sin TARGET.
    # Desde FIL 2 (sin labels) hasta el final.
    # Desde COL 1 hasta penultima COL (sin TARGET).
    global matriz_entrenamiento_input
    matriz_entrenamiento_input = entrenamiento_csv[1:, :matriz_entrenamiento_cantCOLs - 1]

    # Leo matriz de entrenamiento output (TARGET) sin labels.
    # Desde fila 2 (sin labels) hasta el final.
    global matriz_entrenamiento_output
    matriz_entrenamiento_output = entrenamiento_csv[1:, matriz_entrenamiento_cantCOLs - 1]

    # Normalizo la matriz de entrenamiento input para que todos los valores esten entre 0 y 1.
    matriz_entrenamiento_input = normalizar(matriz_entrenamiento_input)


def read_validacion():
    validacion_csv = genfromtxt('../validacionEqual.csv', delimiter=',', names=None, dtype=float)

    validacion_labels = validacion_csv[0]
    global matriz_validacion_cantFILs
    matriz_validacion_cantFILs = len(validacion_csv)
    global matriz_validacion_cantCOLs
    matriz_validacion_cantCOLs = len(validacion_csv[0])

    # Leo matriz de validacion input sin labels y sin TARGET.
    # Desde FIL 2 (sin labels) hasta el final.
    # Desde COL 1 hasta penultima COL (sin TARGET).
    global matriz_validacion_input
    matriz_validacion_input = validacion_csv[1:, :matriz_validacion_cantCOLs - 1]
    # print matriz_validacion_input

    # Leo matriz de validacion output (TARGET) sin labels.
    # Desde fila 2 (sin labels) hasta el final.
    global matriz_validacion_output
    matriz_validacion_output = validacion_csv[1:, matriz_validacion_cantCOLs - 1]
    # print matriz_validacion_output

    # Normalizo la matriz de validacion input para que todos los valores esten entre 0 y 1.
    matriz_validacion_input = normalizar(matriz_validacion_input)


def normalizar(matriz_a_normalizar):
    matriz_normalizada = []

    # Recorro por COL.
    for indice in range(len(matriz_a_normalizar[0])):
        col_normalizada = []
        columna = matriz_a_normalizar[:, indice]
        elemento_max = 1

        # Recorro por cada elemento en COL.
        for elemento in columna:
            if (elemento >= elemento_max):
                elemento_max = elemento

        # Normalizo por el valor MAX de la COL y agrego a la matriz normalizada como FIL.
        col_normalizada = columna / elemento_max
        matriz_normalizada.append(col_normalizada)

    # Devuelvo la matriz normalizada traspuesta dado que agregue a las COLs como FILs.
    return (array(matriz_normalizada).T)


def procesar():
    cant_iguales = 0
    cant_no_iguales = 0
    cant_total = 0
    fallas_uno = 0

    set_entrenamiento_input = array(matriz_entrenamiento_input)
    set_entrenamiento_output = array([matriz_entrenamiento_output]).T
    random.seed(1)
    pesos_neuronas = 2 * random.random((matriz_entrenamiento_cantCOLs - 1, 1)) - 1

    # Calculo los pesos de las neuronas a partir de la matriz de entrenamiento.
    for i in xrange(cant_iteraciones):
        output = 1 / (1 + exp(-(dot(set_entrenamiento_input, pesos_neuronas))))
        pesos_neuronas += dot(set_entrenamiento_input.T, (set_entrenamiento_output - output) * output * (1 - output))

    print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Termino Calcular Pesos Neuronas.'

    # Evaluo el TARGET de la matriz de validacion a partir de la red neuronal.
    for indice in range(len(matriz_validacion_input)):
        target_calculado = round(1 / (1 + exp(-(dot(array(matriz_validacion_input[indice, :]), pesos_neuronas)))))
        target_real = (array([matriz_validacion_output]).T)[indice, :]

        # Verifico si la red neuronal predijo bien.
        if target_calculado == target_real:
            cant_iguales += 1
            cant_total += 1
        else:
            if target_real == 1:
                fallas_uno += 1
            cant_no_iguales += 1
            cant_total += 1

    print '--------------------------------------------------'
    print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Resultado:'
    print '	Cantidad de predicciones: ' + str(cant_iguales)
    print '	Cantidad de errores: ' + str(cant_no_iguales)
    print '	Cantidad de fallas uno: ' + str(fallas_uno)
    print '	Cantidad total: ' + str(cant_total)
    print '	Porcentaje de prediccion: ' + str(format(float(cant_iguales) / float(cant_total) * 100, '.2f')) + '%'
    print '--------------------------------------------------'
    print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Fin.'
    print '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*'


# Main()
print '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*'
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Inicio con ' + str(cant_iteraciones) + ' iteraciones.'
read_entrenamiento()
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Termino Entrenamiento.'
read_validacion()
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Termino Validacion.'
procesar()
