# Práctica: Algoritmos de Ordenamiento Iterativos y Recursivos con OpenMP

## Información General
* **Materia:** Programación Paralela
* **Equipo:** Equipo 

### Integrantes (Orden Alfabético por Primer Apellido)
1. **Zárate Castillo Andrea Itzel**
2.**Marin Puga Ulises Israel]


---

## Algoritmos Seleccionados
1. **Algoritmo Iterativo:** Odd-Even Transposition Sort
   * **Versión Secuencial:** Intercambio alternado por fases pares e impares en un solo hilo.
   * **Versión Paralela:** Paralelización de bucles por fase mediante `#pragma omp parallel for` con sincronización `#pragma omp barrier`.
2. **Algoritmo Recursivo:** QuickSort
   * **Versión Secuencial:** Divide y vencerás tradicional con llamadas recursivas directas.
   * **Versión Paralela:** Generación de tareas independientes con `#pragma omp task`, coordinadas mediante `#pragma omp taskgroup` y `#pragma omp taskwait`.

---

## Directivas y Funciones de OpenMP Utilizadas
* `#pragma omp parallel`: Define regiones paralelas para ejecución multihilo.
* `#pragma omp parallel for`: Distribuye el trabajo del bucle `for` (llenado del arreglo y fases de ordenamiento) entre los hilos activos.
* `#pragma omp task`: Crea tareas asíncronas para el procesamiento de cada sub-arreglo recursivo.
* `#pragma omp taskgroup` / `#pragma omp taskwait`: Sincronizan y coordinan la finalización de las sub-tareas recursivas.
* `#pragma omp barrier`: Garantiza que todos los hilos concluyan una fase del ordenamiento par-impar antes de pasar a la siguiente.
* `#pragma omp single`: Asegura que el punto de entrada recursivo sea lanzado por un único hilo.
* `omp_get_wtime()`: Mide el tiempo exacto de ejecución en segundos para el cálculo de performance y Speedup ($S = T_s / T_p$).
* `rand_r(&semilla)`: Garantiza la generación paralela de números aleatorios thread-safe con semillas independientes.

---

## Instrucciones de Compilación y Ejecución

### Requisitos Previos
* Compilador GCC / MinGW configurado con soporte para OpenMP.
* Entorno Code::Blocks o terminal de comandos.

### Compilación desde Terminal
```bash
g++ -fopenmp main.cpp -o ordenamientos
./ordenamientos
