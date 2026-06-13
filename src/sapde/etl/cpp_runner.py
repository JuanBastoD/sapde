"""
Compilador y ejecutor del ETL C++/OpenMP.

Intenta compilar con CMake (MSVC o MinGW) y ejecutar el binario resultante.
Si la compilación falla, lanza CppNoDisponible para que pipeline.py active el fallback.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("sapde.etl.cpp_runner")

# Ruta al directorio etl_cpp relativa a la raíz del proyecto
_RAIZ = Path(__file__).resolve().parents[4]
_ETL_CPP_DIR = _RAIZ / "etl_cpp"
_BUILD_DIR = _ETL_CPP_DIR / "build"
_EXE = _BUILD_DIR / "etl.exe"  # Windows


class CppNoDisponible(RuntimeError):
    """Se lanza cuando el ETL C++ no puede compilarse ni ejecutarse."""


@dataclass
class ResultadoETL:
    hilos: int
    registros: int
    t_lectura: float
    t_etl: float
    t_escritura: float
    t_total: float
    modo: str = "cpp"


def _cmake_disponible() -> bool:
    return shutil.which("cmake") is not None


def compilar(forzar: bool = False) -> Path:
    """
    Compila el ETL C++ con CMake.

    Parámetros
    ----------
    forzar : bool
        Si True, recompila aunque el binario ya exista.

    Retorna
    -------
    Path al ejecutable compilado.

    Lanza
    -----
    CppNoDisponible si cmake no está disponible o la compilación falla.
    """
    if _EXE.exists() and not forzar:
        logger.info("ETL C++ ya compilado: %s", _EXE)
        return _EXE

    if not _cmake_disponible():
        raise CppNoDisponible(
            "cmake no encontrado en PATH. "
            "Instala CMake desde https://cmake.org/download/ "
            "o usa el fallback Python (se activa automáticamente)."
        )

    logger.info("Compilando ETL C++ con CMake en %s ...", _BUILD_DIR)
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Paso 1: cmake configure
    cmd_configure = [
        "cmake", str(_ETL_CPP_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    resultado = subprocess.run(
        cmd_configure,
        cwd=str(_BUILD_DIR),
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        logger.error("cmake configure falló:\n%s", resultado.stderr)
        raise CppNoDisponible(
            f"cmake configure falló (ver arriba). "
            "El fallback Python se activará automáticamente."
        )

    # Paso 2: cmake build
    cmd_build = ["cmake", "--build", ".", "--config", "Release"]
    resultado = subprocess.run(
        cmd_build,
        cwd=str(_BUILD_DIR),
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        logger.error("cmake build falló:\n%s", resultado.stderr)
        raise CppNoDisponible("cmake build falló. Activando fallback Python.")

    # Buscar el ejecutable en posibles ubicaciones (MSVC vs MinGW)
    candidatos = [
        _BUILD_DIR / "etl.exe",
        _BUILD_DIR / "Release" / "etl.exe",
        _BUILD_DIR / "Debug" / "etl.exe",
    ]
    for c in candidatos:
        if c.exists():
            logger.info("ETL C++ compilado: %s", c)
            return c

    raise CppNoDisponible("Ejecutable ETL no encontrado tras la compilación.")


def ejecutar(
    exe: Path,
    input_csv: Path,
    output_csv: Path,
    n_hilos: int = 1,
) -> ResultadoETL:
    """
    Ejecuta el binario ETL C++ y parsea su salida JSON.

    Retorna ResultadoETL con los tiempos de ejecución.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        "--input",  str(input_csv),
        "--output", str(output_csv),
        "--hilos",  str(n_hilos),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_hilos)

    logger.info("Ejecutando ETL C++: hilos=%d", n_hilos)
    resultado = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if resultado.returncode != 0:
        raise CppNoDisponible(f"ETL C++ falló (código {resultado.returncode}):\n{resultado.stderr}")

    # Parsear línea JSON de resultados
    for linea in resultado.stdout.splitlines():
        if linea.startswith("RESULTADO_JSON:"):
            datos = json.loads(linea.removeprefix("RESULTADO_JSON:"))
            return ResultadoETL(**datos, modo="cpp")

    raise CppNoDisponible("No se encontró RESULTADO_JSON en la salida del ETL C++.")
