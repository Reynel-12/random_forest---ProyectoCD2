import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def calcular_lead_time_real(dias_str):
    # Diccionario para convertir nombres a número de día
    mapa_dias = {
        'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 
        'viernes': 4, 'sabado': 5, 'domingo': 6
    }
    
    # Limpiar y convertir a lista de números (ej: [0, 1] para lunes, martes)
    dias = [mapa_dias[d.strip().lower()] for d in str(dias_str).split(',') if d.strip().lower() in mapa_dias]
    
    if not dias:
        return 7  # Si no hay datos, asumimos una semana
    
    dias.sort()
    
    # Calcular los huecos entre días (incluyendo el salto del domingo al lunes)
    huecos = []
    for i in range(len(dias)):
        if i < len(dias) - 1:
            huecos.append(dias[i+1] - dias[i])
        else:
            # Hueco desde el último día de la semana hasta el primero de la siguiente
            huecos.append(7 - dias[i] + dias[0])
            
    # RETORNAMOS EL HUECO MÁS LARGO
    # Porque el stock debe ser suficiente para sobrevivir el intervalo más largo sin camión.
    return max(huecos)

# En la lógica de compra sugerida:
def sugerir_compra_profesional(row):
    # Ahora 'lead_time_dias' es el hueco más largo sin camión
    factor_espera = row["lead_time_dias"] / 7
    
    # Demanda mínima para no quebrar stock en el hueco más largo
    demanda_critica = row["demanda_predicha_7dias"] * factor_espera
    
    # Stock de seguridad (20% de la demanda + margen de error del modelo)
    # Lo ajustamos por el tiempo de espera
    seguridad = (demanda_critica * 0.20) + (mae * factor_espera)
    
    punto_reorden = demanda_critica + seguridad
    
    if row["stock_actual"] < punto_reorden:
        # Compramos para cubrir el hueco y quedar con un remanente
        cantidad = (punto_reorden * 1.2) - row["stock_actual"]
        return max(0, round(cantidad))
    return 0



########################


# ==========================
# 1. CARGA Y PROCESAMIENTO DE PROVEEDORES
# ==========================
# Cargamos proveedores y calculamos cuántos días pasan entre entregas (frecuencia)
proveedores = pd.read_csv("proveedores.csv")

proveedores['lead_time_dias'] = proveedores['dias_reposicion'].apply(calcular_lead_time_real)
# Promediamos el lead time por proveedor por si aparece repetido
lead_time_final = proveedores.groupby('proveedor_id')['lead_time_dias'].mean().reset_index()

# ==========================
# 2. CARGA DE STOCK Y VENTAS (CON CORRECCIONES)
# ==========================
estado_stock = pd.read_csv("stock_estado.csv", on_bad_lines='skip')
estado_stock.columns = ["codigo_producto", "stock_actual", "stock_minimo", "proveedor_id"]

# Unimos stock con el lead time de sus proveedores
estado_stock = estado_stock.merge(lead_time_final, on="proveedor_id", how="left")
estado_stock['lead_time_dias'] = estado_stock['lead_time_dias'].fillna(7) # Default: 1 semana si no hay info

ventas = pd.read_csv("ventas_historico.csv")
ventas["fecha"] = pd.to_datetime(ventas["fecha"])

# ==========================
# 3. FEATURE ENGINEERING (MEMORIA)
# ==========================
ventas_semanales = ventas.groupby(['codigo_producto', pd.Grouper(key='fecha', freq='W')]).agg({
    'cantidad_vendida': 'sum'
}).reset_index().sort_values(['codigo_producto', 'fecha'])

ventas_semanales['venta_semana_anterior'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(1)
ventas_semanales['venta_hace_2_semanas'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(2)
ventas_semanales['tendencia'] = ventas_semanales['cantidad_vendida'] - ventas_semanales['venta_semana_anterior']
ventas_semanales['mes'] = ventas_semanales['fecha'].dt.month
ventas_semanales['demanda_proxima_semana'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(-1)

data_entrenamiento = ventas_semanales.dropna()

# Limpieza de IDs y unión final
for df in [data_entrenamiento, estado_stock]:
    df["codigo_producto"] = df["codigo_producto"].astype(str).str.strip()

data_final = data_entrenamiento.merge(estado_stock, on="codigo_producto", how="left").fillna(0)

# ==========================
# 4. ENTRENAMIENTO
# ==========================
# Agregamos 'lead_time_dias' a las variables que el modelo analiza
features = ["cantidad_vendida", "venta_semana_anterior", "venta_hace_2_semanas", "tendencia", "mes", "stock_minimo", "lead_time_dias"]
X = data_final[features]
y = data_final["demanda_proxima_semana"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_demanda = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42)
modelo_demanda.fit(X_train, y_train)

mae = mean_absolute_error(y_test, modelo_demanda.predict(X_test))

# ==========================
# 5. PREDICCIÓN Y LÓGICA DE COMPRA (LEAD TIME DINÁMICO)
# ==========================
ultima_semana = ventas_semanales.groupby('codigo_producto').last().reset_index()
ultima_semana["codigo_producto"] = ultima_semana["codigo_producto"].astype(str).str.strip()
ultima_semana = ultima_semana.merge(estado_stock, on="codigo_producto", how="left").fillna(0)

X_input = ultima_semana[features]
ultima_semana["demanda_predicha_7dias"] = modelo_demanda.predict(X_input)

ultima_semana["cantidad_a_comprar"] = ultima_semana.apply(sugerir_compra_profesional, axis=1)

print(f"R2 Final: {modelo_demanda.score(X_test, y_test):.2f} | MAE: {mae:.2f}")
print("\n--- RESULTADOS CON LEAD TIME ---")
print(ultima_semana[["codigo_producto", "lead_time_dias", "stock_actual", "demanda_predicha_7dias", "cantidad_a_comprar"]].head(10))


# Unir con la información de proveedores para tener los nombres
resumen_compras = ultima_semana[ultima_semana['cantidad_a_comprar'] > 0].copy()

# Agrupar por proveedor
reporte_proveedores = resumen_compras.groupby('proveedor_id').agg({
    'codigo_producto': 'count',
    'cantidad_a_comprar': 'sum'
}).rename(columns={'codigo_producto': 'variedad_productos', 'cantidad_a_comprar': 'unidades_totales'})

print("\n--- RESUMEN PARA EL DEPARTAMENTO DE COMPRAS ---")
print(reporte_proveedores)

# Exportar a CSV para abrir en Excel
reporte_proveedores.to_csv("pedido_total_por_proveedor.csv")
ultima_semana.to_csv("detalle_productos_a_comprar.csv", index=False)