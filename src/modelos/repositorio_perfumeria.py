from collections import defaultdict
from decimal import Decimal

from psycopg2.extras import RealDictCursor

from modelos.conexion_bd import obtener_conexion


ESTADOS_COMPRA = ("CONFIRMADA", "ENTREGADA")


def _normalizar_valor(valor):
    if isinstance(valor, Decimal):
        return float(valor)
    return valor


def _normalizar_fila(fila):
    return {clave: _normalizar_valor(valor) for clave, valor in dict(fila).items()}


class RepositorioPerfumeria:
    def __init__(self):
        self._conexion = obtener_conexion()

    def cerrar(self):
        self._conexion.close()

    def obtener_usuarios_con_mas_movimientos(self, limite=5):
        return self.obtener_usuarios_con_movimientos(limite=limite, solo_con_movimientos=True)

    def obtener_usuarios_con_movimientos(self, limite=None, solo_con_movimientos=False):
        filtro_movimientos = "HAVING COALESCE(SUM(m.cantidad), 0) > 0" if solo_con_movimientos else ""
        limite_sql = "LIMIT %s" if limite else ""
        consulta = """
            WITH movimientos AS (
                SELECT usuario_id, COUNT(*) AS cantidad
                FROM busquedas
                WHERE usuario_id IS NOT NULL
                GROUP BY usuario_id

                UNION ALL
                SELECT usuario_id, COUNT(*) AS cantidad
                FROM visitas_producto
                WHERE usuario_id IS NOT NULL
                GROUP BY usuario_id

                UNION ALL
                SELECT usuario_id, COUNT(*) AS cantidad
                FROM reservas
                GROUP BY usuario_id

                UNION ALL
                SELECT usuario_id, COUNT(*) AS cantidad
                FROM favoritos
                GROUP BY usuario_id

                UNION ALL
                SELECT usuario_id, COUNT(*) AS cantidad
                FROM intereses_reposicion
                WHERE usuario_id IS NOT NULL
                GROUP BY usuario_id
            )
            SELECT
                u.id,
                u.nombres,
                u.apellidos,
                u.email::text AS email,
                COALESCE(SUM(m.cantidad), 0)::int AS movimientos
            FROM usuarios u
            LEFT JOIN movimientos m ON u.id = m.usuario_id
            WHERE u.activo = true
            GROUP BY u.id, u.nombres, u.apellidos, u.email
            {filtro_movimientos}
            ORDER BY movimientos DESC, u.id
            {limite_sql};
        """.format(filtro_movimientos=filtro_movimientos, limite_sql=limite_sql)
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            parametros = (limite,) if limite else ()
            cursor.execute(consulta, parametros)
            return [_normalizar_fila(fila) for fila in cursor.fetchall()]

    def obtener_perfumes_activos(self):
        consulta = """
            SELECT
                p.id,
                p.codigo,
                p.nombre,
                p.descripcion,
                p.genero_objetivo,
                p.volumen_ml,
                p.precio,
                p.stock,
                p.activo,
                m.nombre AS marca,
                f.nombre AS familia,
                c.nombre AS concentracion
            FROM perfumes p
            JOIN marcas m ON m.id = p.marca_id
            LEFT JOIN familias_olfativas f ON f.id = p.familia_id
            LEFT JOIN concentraciones c ON c.id = p.concentracion_id
            WHERE p.activo = true
            ORDER BY p.id;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta)
            return [_normalizar_fila(fila) for fila in cursor.fetchall()]

    def obtener_interacciones(self):
        interacciones = defaultdict(self._crear_interaccion)
        self._sumar_visitas(interacciones)
        self._sumar_reservas(interacciones)
        self._sumar_favoritos(interacciones)
        self._sumar_intereses_reposicion(interacciones)
        return dict(interacciones)

    def obtener_busquedas(self):
        consulta = """
            SELECT usuario_id, texto_busqueda, filtros, fecha_busqueda
            FROM busquedas
            WHERE usuario_id IS NOT NULL
            ORDER BY fecha_busqueda;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta)
            return [_normalizar_fila(fila) for fila in cursor.fetchall()]

    @staticmethod
    def _crear_interaccion():
        return {
            "visitas": 0,
            "reservas": 0,
            "compras": 0,
            "favoritos": 0,
            "reposiciones": 0,
        }

    def _sumar_visitas(self, interacciones):
        consulta = """
            SELECT usuario_id, perfume_id, COUNT(*)::int AS total
            FROM visitas_producto
            WHERE usuario_id IS NOT NULL
            GROUP BY usuario_id, perfume_id;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta)
            for fila in cursor.fetchall():
                clave = (fila["usuario_id"], fila["perfume_id"])
                interacciones[clave]["visitas"] += fila["total"]

    def _sumar_reservas(self, interacciones):
        consulta = """
            SELECT
                usuario_id,
                perfume_id,
                COUNT(*)::int AS reservas,
                COUNT(*) FILTER (WHERE estado IN %s)::int AS compras
            FROM reservas
            GROUP BY usuario_id, perfume_id;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta, (ESTADOS_COMPRA,))
            for fila in cursor.fetchall():
                clave = (fila["usuario_id"], fila["perfume_id"])
                interacciones[clave]["reservas"] += fila["reservas"]
                interacciones[clave]["compras"] += fila["compras"]

    def _sumar_favoritos(self, interacciones):
        consulta = """
            SELECT usuario_id, perfume_id, COUNT(*)::int AS total
            FROM favoritos
            GROUP BY usuario_id, perfume_id;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta)
            for fila in cursor.fetchall():
                clave = (fila["usuario_id"], fila["perfume_id"])
                interacciones[clave]["favoritos"] += fila["total"]

    def _sumar_intereses_reposicion(self, interacciones):
        consulta = """
            SELECT usuario_id, perfume_id, COUNT(*)::int AS total
            FROM intereses_reposicion
            WHERE usuario_id IS NOT NULL
            GROUP BY usuario_id, perfume_id;
        """
        with self._conexion.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(consulta)
            for fila in cursor.fetchall():
                clave = (fila["usuario_id"], fila["perfume_id"])
                interacciones[clave]["reposiciones"] += fila["total"]
