from collections import Counter, defaultdict
from math import log1p
import os
import re
import warnings

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NOMBRES_VARIABLES_MODELO = [
    "log_visitas",
    "log_reservas",
    "es_favorito",
    "log_reposiciones",
    "log_coincidencias_busqueda",
    "afinidad_historica_producto",
    "afinidad_promedio_cluster",
    "coincide_marca_preferida",
    "coincide_familia_preferida",
    "coincide_genero_preferido",
    "precio_normalizado",
    "stock_normalizado",
]


class RecomendadorProductos:
    def __init__(self, perfumes, interacciones, busquedas):
        self.perfumes = perfumes
        self.interacciones = interacciones
        self.busquedas = busquedas
        self.perfume_por_id = {perfume["id"]: perfume for perfume in perfumes}
        self.usuarios = sorted({usuario_id for usuario_id, _ in interacciones.keys()} | {
            busqueda["usuario_id"] for busqueda in busquedas
        })
        self.productos = [perfume["id"] for perfume in perfumes]
        self.indice_usuario = {usuario_id: indice for indice, usuario_id in enumerate(self.usuarios)}
        self.indice_producto = {producto_id: indice for indice, producto_id in enumerate(self.productos)}
        self.busquedas_por_usuario_producto = self._calcular_busquedas_por_producto()
        self.matriz_preferencias = self._construir_matriz_preferencias()
        self.clusters = self._entrenar_clusters()
        self.promedios_cluster = self._calcular_promedios_cluster()
        self.preferencias_usuario = self._calcular_preferencias_usuario()
        self.variables_entrenamiento, self.etiquetas_entrenamiento = self._crear_dataset_regresion()
        self.metricas_regresion = {}
        self.coeficientes_regresion = []
        self.modelo_compra = self._entrenar_regresion_logistica()

    def recomendar_para_usuarios(self, usuarios):
        resultados = []
        for usuario in usuarios:
            usuario_id = usuario["id"]
            producto_id, puntaje = self._seleccionar_producto(usuario_id)
            producto = self.perfume_por_id[producto_id]
            probabilidad = self._predecir_probabilidad_compra(usuario_id, producto_id)
            resultados.append({
                "usuario": usuario,
                "producto": producto,
                "cluster": int(self.clusters[self.indice_usuario[usuario_id]]) if usuario_id in self.indice_usuario else 0,
                "puntaje": round(float(puntaje), 4),
                "probabilidad_compra": round(float(probabilidad), 4),
                "razon": self._explicar_recomendacion(usuario_id, producto_id),
            })
        return resultados

    def obtener_tabla_recomendaciones(self, usuarios):
        filas = []
        for resultado in self.recomendar_para_usuarios(usuarios):
            usuario = resultado["usuario"]
            producto = resultado["producto"]
            producto_mas_reservado = self._producto_mas_reservado(usuario["id"])
            datos = self.interacciones.get((usuario["id"], producto["id"]), {})
            filas.append({
                "usuario_id": usuario["id"],
                "nombres": usuario["nombres"],
                "apellidos": usuario["apellidos"],
                "email": usuario["email"],
                "movimientos": usuario["movimientos"],
                "cluster": resultado["cluster"],
                "producto_recomendado_id": producto["id"],
                "producto_recomendado": producto["nombre"],
                "marca": producto["marca"],
                "familia": producto.get("familia"),
                "concentracion": producto.get("concentracion"),
                "genero_objetivo": producto.get("genero_objetivo"),
                "precio": producto["precio"],
                "stock": producto["stock"],
                "puntaje_compatibilidad": resultado["puntaje"],
                "probabilidad_compra": resultado["probabilidad_compra"],
                "incertidumbre_probabilidad": round(1.0 - abs(resultado["probabilidad_compra"] - 0.5) * 2.0, 4),
                "visitas_producto_recomendado": datos.get("visitas", 0),
                "reservas_producto_recomendado": datos.get("reservas", 0),
                "compras_producto_recomendado": datos.get("compras", 0),
                "favoritos_producto_recomendado": datos.get("favoritos", 0),
                "coincidencias_busqueda_producto": round(
                    self.busquedas_por_usuario_producto.get((usuario["id"], producto["id"]), 0.0),
                    4,
                ),
                "producto_mas_reservado_id": producto_mas_reservado["producto_id"],
                "producto_mas_reservado": producto_mas_reservado["producto"],
                "reservas_producto_mas_reservado": producto_mas_reservado["reservas"],
                "motivo": resultado["razon"],
            })
        return filas

    def obtener_resumen_usuarios(self, usuarios):
        filas = []
        for usuario in usuarios:
            usuario_id = usuario["id"]
            producto_mas_reservado = self._producto_mas_reservado(usuario_id)
            totales = self._totales_usuario(usuario_id)
            preferencias = self.preferencias_usuario.get(usuario_id, {})
            cluster = self.clusters[self.indice_usuario[usuario_id]] if usuario_id in self.indice_usuario else None
            filas.append({
                "usuario_id": usuario_id,
                "nombres": usuario["nombres"],
                "apellidos": usuario["apellidos"],
                "email": usuario["email"],
                "movimientos": usuario["movimientos"],
                "cluster": cluster,
                "total_visitas": totales["visitas"],
                "total_reservas": totales["reservas"],
                "total_compras": totales["compras"],
                "total_favoritos": totales["favoritos"],
                "total_reposiciones": totales["reposiciones"],
                "producto_mas_reservado_id": producto_mas_reservado["producto_id"],
                "producto_mas_reservado": producto_mas_reservado["producto"],
                "reservas_producto_mas_reservado": producto_mas_reservado["reservas"],
                "marca_preferida": preferencias.get("marca"),
                "familia_preferida": preferencias.get("familia"),
                "genero_preferido": preferencias.get("genero_objetivo"),
                "concentracion_preferida": preferencias.get("concentracion"),
            })
        return filas

    def obtener_matriz_variables_usuario_producto(self):
        filas = []
        for usuario_id in self.usuarios:
            cluster = self.clusters[self.indice_usuario[usuario_id]]
            for producto_id in self.productos:
                perfume = self.perfume_por_id[producto_id]
                datos = self.interacciones.get((usuario_id, producto_id), {})
                variables = self._crear_variables_modelo(usuario_id, producto_id)
                fila = {
                    "usuario_id": usuario_id,
                    "producto_id": producto_id,
                    "producto": perfume["nombre"],
                    "marca": perfume["marca"],
                    "familia": perfume.get("familia"),
                    "cluster": int(cluster),
                    "visitas": datos.get("visitas", 0),
                    "reservas": datos.get("reservas", 0),
                    "compras": datos.get("compras", 0),
                    "favoritos": datos.get("favoritos", 0),
                    "reposiciones": datos.get("reposiciones", 0),
                    "coincidencias_busqueda": round(
                        self.busquedas_por_usuario_producto.get((usuario_id, producto_id), 0.0),
                        4,
                    ),
                    "puntaje_compatibilidad": round(float(self._puntaje_compatibilidad(usuario_id, producto_id)), 4),
                    "probabilidad_compra": round(float(self._predecir_probabilidad_compra(usuario_id, producto_id)), 4),
                    "compra_observada": 1 if datos.get("compras", 0) > 0 else 0,
                }
                fila.update({
                    nombre: round(float(valor), 6)
                    for nombre, valor in zip(NOMBRES_VARIABLES_MODELO, variables)
                })
                filas.append(fila)
        return filas

    def obtener_metricas_modelo(self):
        return [{
            "modelo": "Regresion Logistica",
            "tipo_validacion": self.metricas_regresion.get("tipo_validacion", "no_disponible"),
            "observaciones": self.metricas_regresion.get("observaciones", 0),
            "compras_observadas": self.metricas_regresion.get("compras_observadas", 0),
            "no_compras_observadas": self.metricas_regresion.get("no_compras_observadas", 0),
            "accuracy": self.metricas_regresion.get("accuracy"),
            "precision": self.metricas_regresion.get("precision"),
            "recall": self.metricas_regresion.get("recall"),
            "f1": self.metricas_regresion.get("f1"),
            "log_loss": self.metricas_regresion.get("log_loss"),
            "brier_score": self.metricas_regresion.get("brier_score"),
            "error_clasificacion": self.metricas_regresion.get("error_clasificacion"),
            "nota_margen_error": (
                "En regresion logistica no se usa margen de error clasico; "
                "se reportan log_loss, brier_score, error de clasificacion e incertidumbre por prediccion."
            ),
        }]

    def obtener_coeficientes_regresion(self):
        return self.coeficientes_regresion

    def obtener_resumen_clusters(self):
        filas = []
        for cluster in sorted(set(self.clusters)):
            indices = np.where(self.clusters == cluster)[0]
            filas.append({
                "cluster": int(cluster),
                "cantidad_usuarios": int(len(indices)),
                "afinidad_promedio": round(float(self.matriz_preferencias[indices].mean()), 6),
                "afinidad_maxima": round(float(self.matriz_preferencias[indices].max()), 6),
            })
        return filas

    def _seleccionar_producto(self, usuario_id):
        mejores = []
        for producto_id in self.productos:
            perfume = self.perfume_por_id[producto_id]
            if perfume["stock"] <= 0:
                continue
            datos = self.interacciones.get((usuario_id, producto_id), {})
            if datos.get("compras", 0) > 0:
                continue
            mejores.append((producto_id, self._puntaje_compatibilidad(usuario_id, producto_id)))

        if not mejores:
            mejores = [(producto_id, self._puntaje_compatibilidad(usuario_id, producto_id)) for producto_id in self.productos]

        return max(mejores, key=lambda item: item[1])

    def _puntaje_compatibilidad(self, usuario_id, producto_id):
        datos = self.interacciones.get((usuario_id, producto_id), {})
        busquedas = self.busquedas_por_usuario_producto.get((usuario_id, producto_id), 0.0)
        cluster = self.clusters[self.indice_usuario[usuario_id]] if usuario_id in self.indice_usuario else 0
        indice_producto = self.indice_producto[producto_id]
        puntaje_cluster = self.promedios_cluster.get(cluster, np.zeros(len(self.productos)))[indice_producto]
        afinidad_producto = self.matriz_preferencias[self.indice_usuario[usuario_id], indice_producto] if usuario_id in self.indice_usuario else 0.0

        return (
            afinidad_producto * 0.45
            + puntaje_cluster * 0.35
            + log1p(busquedas) * 1.4
            + log1p(datos.get("visitas", 0)) * 1.2
            + datos.get("favoritos", 0) * 2.0
            + datos.get("reposiciones", 0) * 1.3
            + self._bono_atributos_preferidos(usuario_id, producto_id)
        )

    def _construir_matriz_preferencias(self):
        matriz = np.zeros((len(self.usuarios), len(self.productos)), dtype=float)
        for (usuario_id, producto_id), datos in self.interacciones.items():
            if usuario_id not in self.indice_usuario or producto_id not in self.indice_producto:
                continue
            matriz[self.indice_usuario[usuario_id], self.indice_producto[producto_id]] += self._peso_interaccion(datos)

        for (usuario_id, producto_id), total in self.busquedas_por_usuario_producto.items():
            if usuario_id in self.indice_usuario and producto_id in self.indice_producto:
                matriz[self.indice_usuario[usuario_id], self.indice_producto[producto_id]] += log1p(total) * 1.5

        maximo = matriz.max()
        if maximo > 0:
            matriz = matriz / maximo
        return matriz

    def _entrenar_clusters(self):
        if len(self.usuarios) < 2 or len(self.productos) == 0:
            return np.zeros(len(self.usuarios), dtype=int)

        cantidad_clusters = min(4, len(self.usuarios))
        modelo = KMeans(n_clusters=cantidad_clusters, random_state=42, n_init=10)
        return modelo.fit_predict(self.matriz_preferencias)

    def _calcular_promedios_cluster(self):
        promedios = {}
        for cluster in set(self.clusters):
            indices = np.where(self.clusters == cluster)[0]
            promedios[cluster] = self.matriz_preferencias[indices].mean(axis=0)
        return promedios

    def _calcular_preferencias_usuario(self):
        preferencias = {}
        for usuario_id in self.usuarios:
            contadores = {
                "marca": Counter(),
                "familia": Counter(),
                "genero_objetivo": Counter(),
                "concentracion": Counter(),
            }
            for producto_id in self.productos:
                peso = self.matriz_preferencias[self.indice_usuario[usuario_id], self.indice_producto[producto_id]]
                if peso <= 0:
                    continue
                perfume = self.perfume_por_id[producto_id]
                for atributo, contador in contadores.items():
                    valor = perfume.get(atributo)
                    if valor:
                        contador[valor] += peso
            preferencias[usuario_id] = {
                atributo: contador.most_common(1)[0][0] if contador else None
                for atributo, contador in contadores.items()
            }
        return preferencias

    def _crear_dataset_regresion(self):
        if not self.usuarios or not self.productos:
            return np.array([]), np.array([])

        x = []
        y = []
        for usuario_id in self.usuarios:
            for producto_id in self.productos:
                x.append(self._crear_variables_modelo(usuario_id, producto_id))
                datos = self.interacciones.get((usuario_id, producto_id), {})
                y.append(1 if datos.get("compras", 0) > 0 else 0)

        return np.array(x, dtype=float), np.array(y, dtype=int)

    def _entrenar_regresion_logistica(self):
        x = self.variables_entrenamiento
        y = self.etiquetas_entrenamiento
        if x.size == 0 or y.size == 0:
            self.metricas_regresion = {"tipo_validacion": "sin_datos"}
            return None

        compras = int(y.sum())
        no_compras = int(len(y) - compras)
        self.metricas_regresion = {
            "observaciones": int(len(y)),
            "compras_observadas": compras,
            "no_compras_observadas": no_compras,
        }

        if len(set(y)) < 2:
            self.metricas_regresion["tipo_validacion"] = "sin_dos_clases"
            return None

        modelo_validacion = self._crear_pipeline_regresion()
        if compras >= 2 and no_compras >= 2 and len(y) >= 8:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
            modelo_validacion.fit(x_train, y_train)
            probabilidades = modelo_validacion.predict_proba(x_test)[:, 1]
            predicciones = (probabilidades >= 0.5).astype(int)
            self.metricas_regresion.update(self._calcular_metricas(y_test, predicciones, probabilidades))
            self.metricas_regresion["tipo_validacion"] = "holdout_25_por_ciento_estratificado"
        else:
            modelo_validacion.fit(x, y)
            probabilidades = modelo_validacion.predict_proba(x)[:, 1]
            predicciones = (probabilidades >= 0.5).astype(int)
            self.metricas_regresion.update(self._calcular_metricas(y, predicciones, probabilidades))
            self.metricas_regresion["tipo_validacion"] = "entrenamiento_completo_muestra_pequena"

        modelo = self._crear_pipeline_regresion()
        modelo.fit(x, y)
        self.coeficientes_regresion = self._extraer_coeficientes(modelo)
        return modelo

    @staticmethod
    def _crear_pipeline_regresion():
        return Pipeline([
            ("escalador", StandardScaler()),
            ("regresion", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])

    @staticmethod
    def _calcular_metricas(y_real, y_predicho, probabilidades):
        etiquetas = [0, 1]
        return {
            "accuracy": round(float(accuracy_score(y_real, y_predicho)), 6),
            "precision": round(float(precision_score(y_real, y_predicho, zero_division=0)), 6),
            "recall": round(float(recall_score(y_real, y_predicho, zero_division=0)), 6),
            "f1": round(float(f1_score(y_real, y_predicho, zero_division=0)), 6),
            "log_loss": round(float(log_loss(y_real, probabilidades, labels=etiquetas)), 6),
            "brier_score": round(float(brier_score_loss(y_real, probabilidades)), 6),
            "error_clasificacion": round(float(1.0 - accuracy_score(y_real, y_predicho)), 6),
        }

    @staticmethod
    def _extraer_coeficientes(modelo):
        regresion = modelo.named_steps["regresion"]
        coeficientes = regresion.coef_[0]
        return [
            {
                "variable": nombre,
                "coeficiente": round(float(coeficiente), 6),
                "impacto": "aumenta_probabilidad" if coeficiente >= 0 else "reduce_probabilidad",
            }
            for nombre, coeficiente in zip(NOMBRES_VARIABLES_MODELO, coeficientes)
        ]

    def _predecir_probabilidad_compra(self, usuario_id, producto_id):
        variables = np.array([self._crear_variables_modelo(usuario_id, producto_id)], dtype=float)
        if self.modelo_compra is None:
            puntaje = self._puntaje_compatibilidad(usuario_id, producto_id)
            return 1.0 / (1.0 + np.exp(-puntaje))
        return self.modelo_compra.predict_proba(variables)[0][1]

    def _crear_variables_modelo(self, usuario_id, producto_id):
        datos = self.interacciones.get((usuario_id, producto_id), {})
        perfume = self.perfume_por_id[producto_id]
        cluster = self.clusters[self.indice_usuario[usuario_id]] if usuario_id in self.indice_usuario else 0
        indice_producto = self.indice_producto[producto_id]
        puntaje_cluster = self.promedios_cluster.get(cluster, np.zeros(len(self.productos)))[indice_producto]
        afinidad_producto = self.matriz_preferencias[self.indice_usuario[usuario_id], indice_producto] if usuario_id in self.indice_usuario else 0.0
        precio_maximo = max(float(p["precio"]) for p in self.perfumes) or 1.0
        stock_maximo = max(int(p["stock"]) for p in self.perfumes) or 1

        return [
            log1p(datos.get("visitas", 0)),
            log1p(datos.get("reservas", 0)),
            1.0 if datos.get("favoritos", 0) > 0 else 0.0,
            log1p(datos.get("reposiciones", 0)),
            log1p(self.busquedas_por_usuario_producto.get((usuario_id, producto_id), 0.0)),
            afinidad_producto,
            puntaje_cluster,
            self._coincide_preferencia(usuario_id, producto_id, "marca"),
            self._coincide_preferencia(usuario_id, producto_id, "familia"),
            self._coincide_preferencia(usuario_id, producto_id, "genero_objetivo"),
            float(perfume["precio"]) / precio_maximo,
            int(perfume["stock"]) / stock_maximo,
        ]

    def _calcular_busquedas_por_producto(self):
        puntajes = defaultdict(float)
        textos_producto = {
            perfume["id"]: self._texto_producto(perfume)
            for perfume in self.perfumes
        }
        for busqueda in self.busquedas:
            usuario_id = busqueda["usuario_id"]
            texto_busqueda = self._normalizar_texto(busqueda.get("texto_busqueda", ""))
            filtros = busqueda.get("filtros") or {}
            for perfume in self.perfumes:
                producto_id = perfume["id"]
                puntaje = self._puntaje_busqueda(texto_busqueda, filtros, perfume, textos_producto[producto_id])
                if puntaje > 0:
                    puntajes[(usuario_id, producto_id)] += puntaje
        return dict(puntajes)

    def _puntaje_busqueda(self, texto_busqueda, filtros, perfume, texto_producto):
        puntaje = 0.0
        terminos = [termino for termino in texto_busqueda.split() if len(termino) > 2]
        for termino in terminos:
            if termino in texto_producto:
                puntaje += 1.0

        if isinstance(filtros, dict):
            for atributo in ("marca", "familia", "genero_objetivo", "concentracion"):
                filtro = filtros.get(atributo) or filtros.get(atributo.replace("_objetivo", ""))
                if filtro and self._normalizar_texto(str(filtro)) in self._normalizar_texto(str(perfume.get(atributo, ""))):
                    puntaje += 1.5
        return puntaje

    def _explicar_recomendacion(self, usuario_id, producto_id):
        perfume = self.perfume_por_id[producto_id]
        razones = []
        preferencias = self.preferencias_usuario.get(usuario_id, {})
        for atributo, etiqueta in (
            ("familia", "familia"),
            ("marca", "marca"),
            ("genero_objetivo", "genero"),
            ("concentracion", "concentracion"),
        ):
            if preferencias.get(atributo) and preferencias[atributo] == perfume.get(atributo):
                razones.append(f"{etiqueta} {perfume.get(atributo)}")

        datos = self.interacciones.get((usuario_id, producto_id), {})
        if datos.get("visitas", 0) > 0:
            razones.append(f"{datos['visitas']} visitas previas")
        if self.busquedas_por_usuario_producto.get((usuario_id, producto_id), 0) > 0:
            razones.append("coincide con sus busquedas")

        if not razones:
            razones.append("afinidad alta en su cluster")
        return ", ".join(razones[:3])

    def _producto_mas_reservado(self, usuario_id):
        candidatos = []
        for (id_usuario, producto_id), datos in self.interacciones.items():
            if id_usuario == usuario_id and datos.get("reservas", 0) > 0:
                candidatos.append((producto_id, datos["reservas"]))

        if not candidatos:
            return {"producto_id": None, "producto": None, "reservas": 0}

        producto_id, reservas = max(candidatos, key=lambda item: item[1])
        return {
            "producto_id": producto_id,
            "producto": self.perfume_por_id.get(producto_id, {}).get("nombre"),
            "reservas": reservas,
        }

    def _totales_usuario(self, usuario_id):
        totales = {
            "visitas": 0,
            "reservas": 0,
            "compras": 0,
            "favoritos": 0,
            "reposiciones": 0,
        }
        for (id_usuario, _), datos in self.interacciones.items():
            if id_usuario != usuario_id:
                continue
            for clave in totales:
                totales[clave] += datos.get(clave, 0)
        return totales

    def _bono_atributos_preferidos(self, usuario_id, producto_id):
        return sum(
            self._coincide_preferencia(usuario_id, producto_id, atributo)
            for atributo in ("marca", "familia", "genero_objetivo", "concentracion")
        ) * 0.8

    def _coincide_preferencia(self, usuario_id, producto_id, atributo):
        preferido = self.preferencias_usuario.get(usuario_id, {}).get(atributo)
        actual = self.perfume_por_id[producto_id].get(atributo)
        return 1.0 if preferido and actual and preferido == actual else 0.0

    @staticmethod
    def _peso_interaccion(datos):
        return (
            datos.get("visitas", 0) * 1.0
            + datos.get("reservas", 0) * 4.0
            + datos.get("compras", 0) * 8.0
            + datos.get("favoritos", 0) * 5.0
            + datos.get("reposiciones", 0) * 3.0
        )

    @staticmethod
    def _texto_producto(perfume):
        partes = [
            perfume.get("nombre"),
            perfume.get("marca"),
            perfume.get("familia"),
            perfume.get("concentracion"),
            perfume.get("genero_objetivo"),
            perfume.get("descripcion"),
        ]
        return RecomendadorProductos._normalizar_texto(" ".join(str(parte or "") for parte in partes))

    @staticmethod
    def _normalizar_texto(texto):
        return re.sub(r"[^a-z0-9 ]+", " ", texto.lower()).strip()
