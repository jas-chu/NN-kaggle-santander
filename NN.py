# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt, exp, array, random, delete, savetxt, column_stack
from datetime import datetime

matriz_entrenamiento_input = []
matriz_entrenamiento_output = []
matriz_entrenamiento_cantFILs = 0
matriz_entrenamiento_cantCOLs = 0

matriz_validacion_input = []
matriz_validacion_output = []
matriz_validacion_cantFILs = 0
matriz_validacion_cantCOLs = 0

cant_iteraciones = 0

# Precision: cuantos 1s predijiste bien sobre la cantidad de 1s totales.
# Recall: relacionado a los falsos negativos.
# Hay que limitar la profundidad o el minimo de elementos de un conjunto para partirlo en dos.
# El set de validacion deberia ser mas chico que el de entrenamiento: 4k - 2k.
# Con las clases balanceadas.

def read_entrenamiento():
    entrenamiento_csv = genfromtxt('entrenamientoEqual.csv', delimiter=',', names=None, dtype=float)

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

    #matriz_entrenamiento_input, matriz_entrenamiento_output = eliminar_filas_random(matriz_entrenamiento_input, matriz_entrenamiento_output)

    # Normalizo la matriz de entrenamiento input para que todos los valores esten entre 0 y 1.
    # matriz_entrenamiento_input = normalizar(matriz_entrenamiento_input)


def read_validacion():
    validacion_csv = genfromtxt('validacionEqual.csv', delimiter=',', names=None, dtype=float)

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

    #matriz_validacion_input, matriz_validacion_output = eliminar_filas_random(matriz_validacion_input, matriz_validacion_output)

    # Normalizo la matriz de validacion input para que todos los valores esten entre 0 y 1.

    # matriz_validacion_input = normalizar(matriz_validacion_input)


def eliminar_filas_random(matriz, target):
    index = 0
    unos = 0
    eliminadas = 0
    contar_unos = True
    while len(matriz) > 2004:
        if index >= len(matriz):
            index = 0
            contar_unos = False
        if target[index] == 1:
            print "eliminadas = ", eliminadas, "INDEX = ", index, " LENGTH = ", len(matriz), "ONES = ", unos
            if contar_unos:
                unos += 1
        if random.rand() > 0.3 and target[index] != 1:
            matriz = delete(matriz, index, axis=0)
            target = delete(target, index, axis=0)
            eliminadas += 1
        else:
            index += 1

    print "len target = ", len(target), " - len matriz = ", len(matriz)
    matriz_guardar = column_stack([matriz, target])
    savetxt("validacionEqual.csv", matriz_guardar, fmt='%s', delimiter=",")
    return matriz, target


def procesar():
    # fit a CART model to the data
    model = DecisionTreeClassifier(max_depth=3)
    # Elimino los IDs
    print "len input = ", len(matriz_entrenamiento_input[:,1:]), " ; len output = ", len(matriz_entrenamiento_output)
    model.fit(matriz_entrenamiento_input[:,1:], matriz_entrenamiento_output)
    print(model)
    # make predictions
    expected = matriz_validacion_output
    predicted = model.predict(matriz_validacion_input[:,1:])
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print '\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n'
    knn = KNeighborsClassifier()
    knn.fit(matriz_entrenamiento_input[:,1:], matriz_entrenamiento_output)
    red_expected = matriz_validacion_output
    red_predicted = knn.predict(matriz_validacion_input[:,1:])
    print(metrics.classification_report(red_expected, red_predicted))
    print(metrics.confusion_matrix(red_expected, red_predicted))


# Main()
print '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*'
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Inicio'
read_entrenamiento()
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Termino Entrenamiento.'
read_validacion()
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Termino Validacion.'
procesar()
print '\n\n--------------------------------------------------'
print str(datetime.now().strftime('%Y.%m.%d %H:%M:%S')) + ' - Fin.'
print '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*'
