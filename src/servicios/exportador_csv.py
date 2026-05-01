import csv
from pathlib import Path

import numpy as np


class ExportadorCSV:
    def __init__(self, carpeta_salida=None):
        raiz_proyecto = Path(__file__).resolve().parents[2]
        self.carpeta_salida = Path(carpeta_salida) if carpeta_salida else raiz_proyecto / "outputs"

    def exportar_analisis(self, tablas):
        self.carpeta_salida.mkdir(parents=True, exist_ok=True)
        archivos = []
        for nombre_archivo, filas in tablas.items():
            ruta = self.carpeta_salida / nombre_archivo
            self._escribir_csv(ruta, filas)
            archivos.append(str(ruta))
        return archivos

    def _escribir_csv(self, ruta, filas):
        filas = list(filas)
        columnas = self._obtener_columnas(filas)
        with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            for fila in filas:
                escritor.writerow({
                    columna: self._normalizar_valor(fila.get(columna))
                    for columna in columnas
                })

    @staticmethod
    def _obtener_columnas(filas):
        columnas = []
        for fila in filas:
            for columna in fila.keys():
                if columna not in columnas:
                    columnas.append(columna)
        return columnas

    @staticmethod
    def _normalizar_valor(valor):
        if valor is None:
            return ""
        if isinstance(valor, (np.integer, np.floating)):
            return valor.item()
        return valor
