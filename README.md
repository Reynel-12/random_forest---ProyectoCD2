# Proyecto Random Forest - CD2

Este repositorio contiene un script en Python que utiliza un modelo de regresión con `RandomForestRegressor` para predecir la demanda de productos con base en históricos de ventas, el stock actual y los tiempos de reposición de proveedores. Con la predicción se calcula una sugerencia de compra para optimizar inventarios.

## Descripción
El script principal `random_forest.py` realiza los siguientes pasos:

1. **Carga y procesamiento de datos:**
   - Lee los archivos CSV: `proveedores.csv`, `stock_estado.csv`, `ventas_historico.csv`.
   - Calcula el `lead_time_dias` real de cada proveedor a partir de los días de reposición.
   - Une la información de stock con los plazos de entrega.

2. **Ingeniería de características:**
   - Agrupa las ventas por semana y genera variables como ventas previas, tendencia y mes.
   - Prepara un dataset final combinando ventas con datos de stock y proveedores.

3. **Entrenamiento de modelo:**
   - Entrena un `RandomForestRegressor` para predecir la demanda de la próxima semana.
   - Evalúa el modelo utilizando MAE y R2.

4. **Predicciones y lógica de compra:**
   - Calcula la demanda pronosticada de la última semana disponible.
   - Sugiere cantidades a comprar considerando el stock actual, el punto de reorden y un margen de seguridad ajustado por el tiempo de espera.
   - Genera reportes de resumen por proveedor y guarda resultados en CSV (`pedido_total_por_proveedor.csv`, `detalle_productos_a_comprar.csv`).

## Archivos clave

- `random_forest.py`: Script principal con toda la lógica.
- `proveedores.csv`: Datos de proveedores y sus días de reposición.
- `stock_estado.csv`: Inventario actual de productos.
- `ventas_historico.csv`: Histórico de ventas diarias.
- `pedido_total_por_proveedor.csv`: Salida generada con las compras sugeridas por proveedor.
- `detalle_productos_a_comprar.csv`: Detalle por producto de la cantidad a adquirir.

## Requisitos

El proyecto requiere Python 3.9+ y las siguientes librerías:

- pandas
- numpy
- scikit-learn

Puedes instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca los archivos CSV mencionados en la misma carpeta que el script.
2. Ejecuta el script:
   ```bash
   python random_forest.py
   ```
3. El script imprimirá resultados de evaluación y generará los archivos de reporte en la carpeta actual.

> Asegúrate de tener los datos de entrada correctamente formateados y de que los `codigo_producto` sean homogéneos (cadenas sin espacios). El script maneja algunas correcciones básicas de limpieza.

## Notas

- Si no existe información de `lead_time_dias` para un proveedor, se asume un plazo por defecto de 7 días.
- La función `calcular_lead_time_real` interpreta los días de reposición (p.ej. "lunes, martes") y devuelve el mayor hueco entre entregas.

---

*Proyecto desarrollado para el Curso CD2 utilizando Random Forest en Python.*
