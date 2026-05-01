from modelos.recomendador_productos import RecomendadorProductos
from modelos.repositorio_perfumeria import RepositorioPerfumeria
from servicios.exportador_csv import ExportadorCSV
from vistas.vista_recomendaciones import VistaRecomendaciones


class ControladorRecomendaciones:
    def __init__(self):
        self.repositorio = RepositorioPerfumeria()
        self.exportador = ExportadorCSV()
        self.vista = VistaRecomendaciones()

    def ejecutar(self):
        try:
            usuarios = self.repositorio.obtener_usuarios_con_mas_movimientos(limite=5)
            todos_los_usuarios = self.repositorio.obtener_usuarios_con_movimientos()
            perfumes = self.repositorio.obtener_perfumes_activos()
            interacciones = self.repositorio.obtener_interacciones()
            busquedas = self.repositorio.obtener_busquedas()

            recomendador = RecomendadorProductos(
                perfumes=perfumes,
                interacciones=interacciones,
                busquedas=busquedas,
            )
            resultados = recomendador.recomendar_para_usuarios(usuarios)
            self.exportador.exportar_analisis({
                "recomendaciones_top_5.csv": recomendador.obtener_tabla_recomendaciones(usuarios),
                "recomendaciones_todos_usuarios.csv": recomendador.obtener_tabla_recomendaciones(todos_los_usuarios),
                "resumen_usuarios.csv": recomendador.obtener_resumen_usuarios(todos_los_usuarios),
                "matriz_usuario_producto.csv": recomendador.obtener_matriz_variables_usuario_producto(),
                "metricas_regresion_logistica.csv": recomendador.obtener_metricas_modelo(),
                "coeficientes_regresion_logistica.csv": recomendador.obtener_coeficientes_regresion(),
                "resumen_clusters.csv": recomendador.obtener_resumen_clusters(),
                "volumen_datos.csv": self._crear_volumen_datos(
                    todos_los_usuarios=todos_los_usuarios,
                    perfumes=perfumes,
                    interacciones=interacciones,
                    busquedas=busquedas,
                    recomendador=recomendador,
                ),
            })
            self.vista.imprimir_resultados(resultados)
        finally:
            self.repositorio.cerrar()

    @staticmethod
    def _crear_volumen_datos(todos_los_usuarios, perfumes, interacciones, busquedas, recomendador):
        return [
            {"dato": "usuarios_activos", "cantidad": len(todos_los_usuarios)},
            {"dato": "perfumes_activos", "cantidad": len(perfumes)},
            {"dato": "pares_usuario_producto_con_interaccion", "cantidad": len(interacciones)},
            {"dato": "busquedas_analizadas", "cantidad": len(busquedas)},
            {"dato": "usuarios_en_modelo", "cantidad": len(recomendador.usuarios)},
            {"dato": "productos_en_modelo", "cantidad": len(recomendador.productos)},
            {"dato": "observaciones_regresion_logistica", "cantidad": len(recomendador.etiquetas_entrenamiento)},
            {"dato": "compras_observadas_regresion", "cantidad": int(recomendador.etiquetas_entrenamiento.sum())},
        ]
