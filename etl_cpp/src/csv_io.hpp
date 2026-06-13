#pragma once
// csv_io.hpp — Lectura y escritura de CSV para el ETL de SAPDE

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ── Estructura de una fila del dataset SAPDE ─────────────────────────────────
struct Registro {
    // Columnas originales
    std::string student_id;
    std::string cohorte;
    int    semestre                    = 0;
    double promedio_semestre           = 0.0;
    double promedio_acumulado          = 0.0;
    double tendencia_calificaciones    = 0.0;
    int    creditos_matriculados       = 0;
    int    creditos_aprobados          = 0;
    double porcentaje_avance_real      = 0.0;
    int    num_asignaturas_reprobadas  = 0;
    double tasa_reprobacion            = 0.0;
    int    semestres_cursados          = 0;
    int    interrupciones_previas      = 0;
    int    deserta                     = 0;
    // Columnas calculadas por el ETL
    double ratio_aprobacion            = 0.0;
    double deficit_creditos            = 0.0;
    double score_riesgo_bruto          = 0.0;
    int    anio_cohorte                = 0;
    int    semestre_cohorte            = 0;  // 1 = primer semestre, 2 = segundo
};

// ── Helpers de parsing ────────────────────────────────────────────────────────
inline double parse_double(const std::string& s) {
    if (s.empty()) return 0.0;
    try { return std::stod(s); } catch (...) { return 0.0; }
}

inline int parse_int(const std::string& s) {
    if (s.empty()) return 0;
    try { return std::stoi(s); } catch (...) { return 0; }
}

// ── Leer CSV ──────────────────────────────────────────────────────────────────
// Formato esperado: 14 columnas definidas en el esquema SAPDE
inline std::vector<Registro> leer_csv(const std::string& ruta) {
    std::ifstream arch(ruta);
    if (!arch.is_open())
        throw std::runtime_error("No se puede abrir: " + ruta);

    std::vector<Registro> datos;
    std::string linea;

    std::getline(arch, linea);  // saltar encabezado

    while (std::getline(arch, linea)) {
        if (linea.empty()) continue;

        std::stringstream ss(linea);
        std::string campo;
        std::vector<std::string> f;
        f.reserve(14);

        while (std::getline(ss, campo, ','))
            f.push_back(campo);

        if (f.size() < 14) continue;

        Registro r;
        r.student_id                 = f[0];
        r.cohorte                    = f[1];
        r.semestre                   = parse_int(f[2]);
        r.promedio_semestre          = parse_double(f[3]);
        r.promedio_acumulado         = parse_double(f[4]);
        r.tendencia_calificaciones   = parse_double(f[5]);
        r.creditos_matriculados      = parse_int(f[6]);
        r.creditos_aprobados         = parse_int(f[7]);
        r.porcentaje_avance_real     = parse_double(f[8]);
        r.num_asignaturas_reprobadas = parse_int(f[9]);
        r.tasa_reprobacion           = parse_double(f[10]);
        r.semestres_cursados         = parse_int(f[11]);
        r.interrupciones_previas     = parse_int(f[12]);
        r.deserta                    = parse_int(f[13]);

        datos.push_back(std::move(r));
    }
    return datos;
}

// ── Escribir CSV ──────────────────────────────────────────────────────────────
inline void escribir_csv(const std::vector<Registro>& datos,
                          const std::string& ruta) {
    std::ofstream arch(ruta);
    if (!arch.is_open())
        throw std::runtime_error("No se puede escribir en: " + ruta);

    arch << "student_id,cohorte,semestre,promedio_semestre,promedio_acumulado,"
            "tendencia_calificaciones,creditos_matriculados,creditos_aprobados,"
            "porcentaje_avance_real,num_asignaturas_reprobadas,tasa_reprobacion,"
            "semestres_cursados,interrupciones_previas,deserta,"
            "ratio_aprobacion,deficit_creditos,score_riesgo_bruto,"
            "anio_cohorte,semestre_cohorte\n";

    arch << std::fixed;
    for (const auto& r : datos) {
        arch << r.student_id               << ','
             << r.cohorte                  << ','
             << r.semestre                 << ','
             << r.promedio_semestre        << ','
             << r.promedio_acumulado       << ','
             << r.tendencia_calificaciones << ','
             << r.creditos_matriculados    << ','
             << r.creditos_aprobados       << ','
             << r.porcentaje_avance_real   << ','
             << r.num_asignaturas_reprobadas << ','
             << r.tasa_reprobacion         << ','
             << r.semestres_cursados       << ','
             << r.interrupciones_previas   << ','
             << r.deserta                  << ','
             << r.ratio_aprobacion         << ','
             << r.deficit_creditos         << ','
             << r.score_riesgo_bruto       << ','
             << r.anio_cohorte             << ','
             << r.semestre_cohorte         << '\n';
    }
}
