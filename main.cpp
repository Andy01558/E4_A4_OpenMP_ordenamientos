#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <iomanip>
#include <algorithm>

// =================================================================
// FUNCIONES AUXILIARES DE MEMORIA Y VERIFICACIÓN
// =================================================================

int* crearArreglo(int tamano) {
    return new int[tamano];
}

void liberarArreglo(int*& arr) {
    if (arr != nullptr) {
        delete[] arr;
        arr = nullptr;
    }
}

void copiarArreglo(const int* origen, int* destino, int tamano) {
    for (int i = 0; i < tamano; ++i) {
        destino[i] = origen[i];
    }
}

bool estaOrdenado(const int* arr, int tamano) {
    for (int i = 0; i < tamano - 1; ++i) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

void mostrarArreglo(const int* arr, int tamano, const char* mensaje) {
    if (tamano > 100) {
        std::cout << "\n[Aviso: El arreglo tiene " << tamano << " elementos. No se imprimira en consola por limite de tamano.]\n";
        return;
    }
    std::cout << "\n" << mensaje << " (Tamano: " << tamano << "):\n[ ";
    for (int i = 0; i < tamano; ++i) {
        std::cout << arr[i] << (i < tamano - 1 ? ", " : "");
    }
    std::cout << " ]\n";
}

// =================================================================
// 2. LLENADO DEL ARREGLO (PARALELO)
// =================================================================

void llenarArregloParalelo(int* arr, int tamano, int rangoMax) {
    #pragma omp parallel
    {
        int idHilo = omp_get_thread_num();
        unsigned int semilla = static_cast<unsigned int>(time(NULL)) ^ (idHilo + 1) ^ (idHilo * 1259);

        #pragma omp for schedule(static)
        for (int i = 0; i < tamano; ++i) {
            arr[i] = rand_r(&semilla) % (rangoMax + 1);
        }
    }
}

// =================================================================
// 3. ALGORITMO ITERATIVO: ODD-EVEN TRANSPOSITION SORT
// =================================================================

void ordenamientoIterativoSecuencial(int* arr, int n) {
    bool estaOrdenado = false;
    while (!estaOrdenado) {
        estaOrdenado = true;
        for (int i = 1; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                estaOrdenado = false;
            }
        }
        for (int i = 0; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                estaOrdenado = false;
            }
        }
    }
}

void ordenamientoIterativoParalelo(int* arr, int n) {
    #pragma omp parallel
    {
        for (int fase = 0; fase < n; ++fase) {
            if (fase % 2 == 0) {
                #pragma omp for schedule(static)
                for (int i = 0; i < n - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        std::swap(arr[i], arr[i + 1]);
                    }
                }
            } else {
                #pragma omp for schedule(static)
                for (int i = 1; i < n - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        std::swap(arr[i], arr[i + 1]);
                    }
                }
            }
            #pragma omp barrier
        }
    }
}

// =================================================================
// 4. ALGORITMO RECURSIVO: QUICK SORT
// =================================================================

int particionar(int* arr, int bajo, int alto) {
    int pivote = arr[alto];
    int i = (bajo - 1);

    for (int j = bajo; j <= alto - 1; j++) {
        if (arr[j] < pivote) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[alto]);
    return (i + 1);
}

void quickSortSecuencial(int* arr, int bajo, int alto) {
    if (bajo < alto) {
        int pi = particionar(arr, bajo, alto);
        quickSortSecuencial(arr, bajo, pi - 1);
        quickSortSecuencial(arr, pi + 1, alto);
    }
}

void quickSortParaleloRecursivo(int* arr, int bajo, int alto, int umbral) {
    if (bajo < alto) {
        int pi = particionar(arr, bajo, alto);

        if ((alto - bajo) < umbral) {
            quickSortSecuencial(arr, bajo, pi - 1);
            quickSortSecuencial(arr, pi + 1, alto);
        } else {
            #pragma omp taskgroup
            {
                #pragma omp task shared(arr) firstprivate(bajo, pi)
                quickSortParaleloRecursivo(arr, bajo, pi - 1, umbral);

                #pragma omp task shared(arr) firstprivate(alto, pi)
                quickSortParaleloRecursivo(arr, pi + 1, alto, umbral);
            }
            #pragma omp taskwait
        }
    }
}

void quickSortParalelo(int* arr, int n) {
    int umbral = 1000;
    #pragma omp parallel
    {
        #pragma omp single
        {
            quickSortParaleloRecursivo(arr, 0, n - 1, umbral);
        }
    }
}

// =================================================================
// MENÚ PRINCIPAL INTERACTIVO
// =================================================================

int main() {
    int tamanoActual = 100;
    int rangoMaxActual = 200;

    int* arrOriginal = crearArreglo(tamanoActual);
    int* arrTrabajo = crearArreglo(tamanoActual);
    bool arregloLlenado = false;

    int opcion = -1;

    do {
        std::cout << "\n======================================================\n";
        std::cout << "  MENU PRACTICA: ORDENAMIENTOS CON OPENMP\n";
        std::cout << "  Configuracion actual: " << tamanoActual << " elementos (Rango 0-" << rangoMaxActual << ")\n";
        std::cout << "======================================================\n";
        std::cout << "1. Configurar Tamano (100 o 10,000,000 elementos)\n";
        std::cout << "2. Llenar arreglo de forma paralela\n";
        std::cout << "3. Ejecutar Ordenamiento Iterativo Secuencial\n";
        std::cout << "4. Ejecutar Ordenamiento Iterativo Paralelo\n";
        std::cout << "5. Ejecutar Ordenamiento Recursivo Secuencial\n";
        std::cout << "6. Ejecutar Ordenamiento Recursivo Paralelo (Tasks)\n";
        std::cout << "7. Mostrar arreglo desordenado (solo si N=100)\n";
        std::cout << "0. Salir\n";
        std::cout << "======================================================\n";
        std::cout << "Seleccione una opcion: ";

        if (!(std::cin >> opcion)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            continue;
        }

        switch (opcion) {
            case 1: {
                std::cout << "\nSeleccione el tamano:\n1. 100 elementos (Rango 0-200)\n2. 10,000,000 elementos (Rango 0-2,000,000)\nOpcion: ";
                int subOp;
                std::cin >> subOp;
                liberarArreglo(arrOriginal);
                liberarArreglo(arrTrabajo);

                if (subOp == 2) {
                    tamanoActual = 10000000;
                    rangoMaxActual = 2000000;
                } else {
                    tamanoActual = 100;
                    rangoMaxActual = 200;
                }
                arrOriginal = crearArreglo(tamanoActual);
                arrTrabajo = crearArreglo(tamanoActual);
                arregloLlenado = false;
                std::cout << "-> Tamano actualizado a " << tamanoActual << " elementos. Debe volver a llenar el arreglo (Op 2).\n";
                break;
            }
            case 2: {
                double inicio = omp_get_wtime();
                llenarArregloParalelo(arrOriginal, tamanoActual, rangoMaxActual);
                double fin = omp_get_wtime();
                arregloLlenado = true;
                std::cout << "\n[OK] Arreglo llenado exitosamente de forma paralela.\n";
                std::cout << "-> Tiempo de llenado: " << std::fixed << std::setprecision(6) << (fin - inicio) << " segundos.\n";
                break;
            }
            case 3: {
                if (!arregloLlenado) { std::cout << "\n[!] Primero llena el arreglo usando la opcion 2.\n"; break; }
                copiarArreglo(arrOriginal, arrTrabajo, tamanoActual);
                double inicio = omp_get_wtime();
                ordenamientoIterativoSecuencial(arrTrabajo, tamanoActual);
                double fin = omp_get_wtime();
                std::cout << "\n-> Iterativo Secuencial: " << (fin - inicio) << " seg. | Estado: "
                          << (estaOrdenado(arrTrabajo, tamanoActual) ? "CORRECTO" : "INCORRECTO") << "\n";
                mostrarArreglo(arrTrabajo, tamanoActual, "Resultado Iterativo Secuencial");
                break;
            }
            case 4: {
                if (!arregloLlenado) { std::cout << "\n[!] Primero llena el arreglo usando la opcion 2.\n"; break; }
                copiarArreglo(arrOriginal, arrTrabajo, tamanoActual);
                double inicio = omp_get_wtime();
                ordenamientoIterativoParalelo(arrTrabajo, tamanoActual);
                double fin = omp_get_wtime();
                std::cout << "\n-> Iterativo Paralelo: " << (fin - inicio) << " seg. | Estado: "
                          << (estaOrdenado(arrTrabajo, tamanoActual) ? "CORRECTO" : "INCORRECTO") << "\n";
                mostrarArreglo(arrTrabajo, tamanoActual, "Resultado Iterativo Paralelo");
                break;
            }
            case 5: {
                if (!arregloLlenado) { std::cout << "\n[!] Primero llena el arreglo usando la opcion 2.\n"; break; }
                copiarArreglo(arrOriginal, arrTrabajo, tamanoActual);
                double inicio = omp_get_wtime();
                quickSortSecuencial(arrTrabajo, 0, tamanoActual - 1);
                double fin = omp_get_wtime();
                std::cout << "\n-> Recursivo Secuencial: " << (fin - inicio) << " seg. | Estado: "
                          << (estaOrdenado(arrTrabajo, tamanoActual) ? "CORRECTO" : "INCORRECTO") << "\n";
                mostrarArreglo(arrTrabajo, tamanoActual, "Resultado Recursivo Secuencial");
                break;
            }
            case 6: {
                if (!arregloLlenado) { std::cout << "\n[!] Primero llena el arreglo usando la opcion 2.\n"; break; }
                copiarArreglo(arrOriginal, arrTrabajo, tamanoActual);
                double inicio = omp_get_wtime();
                quickSortParalelo(arrTrabajo, tamanoActual);
                double fin = omp_get_wtime();
                std::cout << "\n-> Recursivo Paralelo (Tasks): " << (fin - inicio) << " seg. | Estado: "
                          << (estaOrdenado(arrTrabajo, tamanoActual) ? "CORRECTO" : "INCORRECTO") << "\n";
                mostrarArreglo(arrTrabajo, tamanoActual, "Resultado Recursivo Paralelo");
                break;
            }
            case 7: {
                if (!arregloLlenado) { std::cout << "\n[!] Primero llena el arreglo usando la opcion 2.\n"; break; }
                mostrarArreglo(arrOriginal, tamanoActual, "Arreglo Desordenado Inicial");
                break;
            }
            case 0:
                std::cout << "\nSaliendo del programa...\n";
                break;
            default:
                std::cout << "\nOpcion invalida.\n";
                break;
        }
    } while (opcion != 0);

    liberarArreglo(arrOriginal);
    liberarArreglo(arrTrabajo);
    return 0;
}
