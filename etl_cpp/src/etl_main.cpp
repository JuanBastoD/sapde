// etl_main.cpp — ETL paralelo con OpenMP para SAPDE (OE1)
//
// Operaciones:
//   1. Limpieza de outliers (clip por rango conocido + IQR)
//   2. Decodificación de cohorte → año + semestre numérico
//   3. Feature engineering: ratio_aprobacion, deficit_creditos, score_riesgo_bruto
//
// Todas las operaciones sobre filas se paralelizan con #pragma omp parallel for.
// El resultado se imprime en formato JSON para que Python lo parse fácilmente.
//
// Compilar:
//   cd etl_cpp && mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release   (MSVC o MinGW detectado automáticamente)
//   cmake --build . --config Release

#include "csv_io.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>

// ── Reducción paralela: media ─────────────────────────────────────────────────
static double media_paralela(const std::vector<double>& v) {
    double suma = 0.0;
    const int n = static_cast<int>(v.size());
    #pragma omp parallel for reduction(+:suma) schedule(static)
    for (int i = 0; i < n; ++i) suma += v[i];
    return (n > 0) ? suma / n : 0.0;
}

// ── Cálculo de percentil (serial — requiere ordenamiento) ────────────────────
static double percentil(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * (static_cast<double>(v.size()) - 1.0);
    const size_t lo  = static_cast<size_t>(std::floor(idx));
    const size_t hi  = std::min(lo + 1, v.size() - 1);
    const double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// ── ETL principal ─────────────────────────────────────────────────────────────
static void etl_paralelo(std::vector<Registro>& datos, int n_hilos) {
    const int n = static_cast<int>(datos.size());
    omp_set_num_threads(n_hilos);

    std::cout << "  Registros: " << n
              << "  Hilos OMP: " << n_hilos << '\n';

    // ── Paso 1: Limpieza — clip por rangos de dominio ──────────────────────
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& r = datos[i];

        // Notas en escala colombiana [1.0, 5.0]
        r.promedio_semestre  = std::max(1.0, std::min(5.0, r.promedio_semestre));
        r.promedio_acumulado = std::max(1.0, std::min(5.0, r.promedio_acumulado));

        // Tasas y porcentajes [0, 1] y [0, 100]
        r.tasa_reprobacion        = std::max(0.0, std::min(1.0, r.tasa_reprobacion));
        r.porcentaje_avance_real  = std::max(0.0, std::min(100.0, r.porcentaje_avance_real));

        // Créditos aprobados no pueden superar los matriculados
        r.creditos_aprobados =
            std::max(0, std::min(r.creditos_matriculados, r.creditos_aprobados));

        // Interrupciones no negativas
        r.interrupciones_previas = std::max(0, r.interrupciones_previas);
    }

    // ── Paso 2: Clip IQR sobre num_asignaturas_reprobadas ──────────────────
    {
        std::vector<double> vals(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            vals[i] = static_cast<double>(datos[i].num_asignaturas_reprobadas);

        // El sort dentro de percentil es serial (necesario)
        const double q1  = percentil(vals, 25.0);
        const double q3  = percentil(vals, 75.0);
        const double lim = q3 + 1.5 * (q3 - q1);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (datos[i].num_asignaturas_reprobadas > static_cast<int>(lim))
                datos[i].num_asignaturas_reprobadas = static_cast<int>(lim);
        }
    }

    // ── Paso 3: Decodificación de cohorte → año + semestre (paralelo) ──────
    // Formato: "YYYY-I" (semestre 1) o "YYYY-II" (semestre 2)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const std::string& c = datos[i].cohorte;
        if (c.size() >= 6) {
            try {
                datos[i].anio_cohorte = std::stoi(c.substr(0, 4));
            } catch (...) {
                datos[i].anio_cohorte = 0;
            }
            const std::string sem = c.substr(5);
            datos[i].semestre_cohorte = (sem == "I") ? 1 : 2;
        }
    }

    // ── Paso 4: Feature engineering paralelo ───────────────────────────────

    // Media global de promedio_acumulado (para referencia en score)
    std::vector<double> notas(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) notas[i] = datos[i].promedio_acumulado;
    const double media_global = media_paralela(notas);
    (void)media_global;  // disponible para extensiones futuras

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& r = datos[i];

        // ratio_aprobacion ∈ [0, 1]: fracción de créditos superados
        const int mat = r.creditos_matriculados;
        r.ratio_aprobacion = (mat > 0)
            ? static_cast<double>(r.creditos_aprobados) / mat
            : 0.0;

        // deficit_creditos: créditos no aprobados (rezago curricular)
        r.deficit_creditos = static_cast<double>(mat - r.creditos_aprobados);

        // score_riesgo_bruto ∈ [0, 1]: mayor → más riesgo
        // Pesos calibrados según literatura de deserción estudiantil
        const double f_nota   = (5.0 - r.promedio_acumulado) / 4.0;       // bajo rendimiento
        const double f_repro  = r.tasa_reprobacion;                        // reprobación
        const double f_avance = 1.0 - r.porcentaje_avance_real / 100.0;   // rezago curricular
        const double f_interr = std::min(1.0, r.interrupciones_previas / 2.0); // discontinuidad

        r.score_riesgo_bruto =
            0.40 * f_nota   +
            0.30 * f_repro  +
            0.20 * f_avance +
            0.10 * f_interr;
    }
}

// ── Parsear argumentos CLI ────────────────────────────────────────────────────
struct Args {
    std::string input;
    std::string output;
    int n_hilos = 1;
};

static Args parsear_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--input"  && i + 1 < argc) a.input   = argv[++i];
        else if (s == "--output" && i + 1 < argc) a.output  = argv[++i];
        else if (s == "--hilos"  && i + 1 < argc) a.n_hilos = std::stoi(argv[++i]);
    }
    if (a.input.empty() || a.output.empty())
        throw std::invalid_argument(
            "Uso: etl --input <csv> --output <csv> [--hilos <n>]"
        );
    return a;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    try {
        const Args args = parsear_args(argc, argv);

        std::cout << "=== SAPDE ETL Paralelo (OpenMP) ===\n"
                  << "Entrada : " << args.input  << '\n'
                  << "Salida  : " << args.output << '\n'
                  << "Hilos   : " << args.n_hilos << '\n';

        using Clk = std::chrono::high_resolution_clock;

        // Lectura (I/O secuencial)
        auto t0   = Clk::now();
        auto datos = leer_csv(args.input);
        auto t1   = Clk::now();
        const double t_lectura = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Lectura : " << datos.size() << " registros en "
                  << t_lectura << "s\n";

        // Procesamiento paralelo
        auto t2 = Clk::now();
        etl_paralelo(datos, args.n_hilos);
        auto t3 = Clk::now();
        const double t_etl = std::chrono::duration<double>(t3 - t2).count();

        // Escritura (I/O secuencial)
        auto t4 = Clk::now();
        escribir_csv(datos, args.output);
        auto t5 = Clk::now();
        const double t_escritura = std::chrono::duration<double>(t5 - t4).count();
        const double t_total     = std::chrono::duration<double>(t5 - t0).count();

        // Resultado en JSON (parseado por Python)
        std::cout << "RESULTADO_JSON:{"
                  << "\"hilos\":"      << args.n_hilos      << ','
                  << "\"registros\":"  << datos.size()      << ','
                  << "\"t_lectura\":"  << t_lectura         << ','
                  << "\"t_etl\":"      << t_etl             << ','
                  << "\"t_escritura\":" << t_escritura      << ','
                  << "\"t_total\":"    << t_total
                  << "}\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR ETL: " << e.what() << '\n';
        return 1;
    }
}
