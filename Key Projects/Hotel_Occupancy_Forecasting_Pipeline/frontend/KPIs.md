KPI / Métrica	Descripción	Gráfica sugerida
1. Volumen de reservas (Booking Volume)	Número total de reservas en el periodo seleccionado (día/semana/mes).	Line chart (series temporales)
2. Noches habitación vendidas (Room Nights)	Suma de h_num_noc × h_tot_hab; mide la demanda en noches-habitación.	Área o línea apilada
3. Tasa de ocupación (Occupancy Rate)	(Room Nights vendidas / Habitaciones_totales disponibles) × 100.	
Se calculan diario o mensual, según ID_empresa y su Habitaciones_tot.	Gauge / indicador numérico + serie temporal	
4. Ingreso Promedio Diario (ADR)	Total ingresos (h_tfa_total) / Room Nights vendidas.	Gauge / barra simple
5. Ingresos por habitación disponible (RevPAR)	ADR × Occupancy Rate. Refleja ingresos ajustados por disponibilidad.	Gauge / serie temporal
6. Anticipación de reserva (Lead Time)	Distribución de días entre fecha de reserva (h_res_fec_ok) y fecha de check-in (h_fec_lld_ok).	Histograma
7. Duración promedio de estancia	Promedio de h_num_noc por reserva.	Box plot o barra
8. Tasa de cancelación y no-show	% de reservas con ID_estatus_reservaciones = “Cancelada” o “No-show” frente al total.	Barras comparativas / línea