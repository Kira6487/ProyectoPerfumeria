class VistaRecomendaciones:
    def imprimir_resultados(self, resultados):
        print("Recomendaciones para los 5 usuarios con mas movimientos")
        print("=" * 72)
        for posicion, resultado in enumerate(resultados, start=1):
            usuario = resultado["usuario"]
            producto = resultado["producto"]
            probabilidad = resultado["probabilidad_compra"] * 100
            print(f"{posicion}. Usuario: {usuario['nombres']} {usuario['apellidos']} ({usuario['email']})")
            print(f"   Movimientos: {usuario['movimientos']} | Cluster: {resultado['cluster']}")
            print(
                "   Producto recomendado: "
                f"{producto['nombre']} - {producto['marca']} "
                f"({producto.get('familia') or 'Sin familia'}, {producto.get('concentracion') or 'Sin concentracion'})"
            )
            print(f"   Precio: S/ {producto['precio']:.2f} | Stock: {producto['stock']}")
            print(f"   Compatibilidad: {resultado['puntaje']:.4f}")
            print(f"   Probabilidad de compra: {probabilidad:.2f}%")
            print(f"   Motivo: {resultado['razon']}")
            if posicion != len(resultados):
                print("-" * 72)
