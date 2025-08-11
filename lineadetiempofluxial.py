# app.py
# Requisitos: streamlit, pandas, plotly, numpy, networkx, python-dateutil, openpyxl
# pip install streamlit pandas plotly numpy networkx python-dateutil openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import date
from dateutil import parser
from pathlib import Path

st.set_page_config(page_title="Línea de Tiempo Proyecto Eólico", layout="wide")

# -------- Helpers --------
DATE_COL_START = "Inicio (AAAA-MM-DD)"
DATE_COL_END_PLAN = "Fin plan (AAAA-MM-DD)"
DATE_COL_END_REAL = "Fin real"

def parse_dependencies(s):
    """Convierte '8,14' -> [8,14] (robusto ante NaN, int, str)."""
    if pd.isna(s) or s in ["—", "-", ""]:
        return []
    if isinstance(s, (int, float)) and not pd.isna(s):
        return [int(s)]
    parts = str(s).replace(" ", "").split(",")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def status_color(s):
    s = (s or "").lower()
    if "complet" in s:
        return "#2ca02c"  # verde
    if "curso" in s:
        return "#1f77b4"  # azul
    if "planific" in s or "recurrente" in s:
        return "#9467bd"  # violeta
    if "pend" in s:
        return "#ff7f0e"  # naranja
    return "#7f7f7f"      # gris

def infer_piloto(row):
    """
    Si existe columna 'Piloto' en el Excel se respeta.
    Si el texto menciona 55/55kW/55 k -> Piloto 55 kW.
    Si ID está en {24,25} -> Piloto 55 kW (según tu plan).
    En otro caso -> Piloto 10 kW.
    """
    val = str(row.get("Piloto", "")).strip()
    if val:
        return val

    texto = " ".join([
        str(row.get("Fase", "")),
        str(row.get("Línea", "")),
        str(row.get("Tarea / Entregable", "")),
        str(row.get("Método", "")),
    ]).lower()

    if "55" in texto or "55kw" in texto or "55 k" in texto or "55 k" in texto:
        return "Piloto 55 kW"

    try:
        if int(row.get("ID", 0)) in {24, 25}:
            return "Piloto 55 kW"
    except Exception:
        pass

    return "Piloto 10 kW"

def compute_risk(row, today):
    estado = str(row.get("Estado", "")).lower()
    fin_plan = row.get(DATE_COL_END_PLAN)  # Timestamp (gracias a process_df)
    today_ts = pd.to_datetime(today)
    days_left = (fin_plan - today_ts).days if pd.notna(fin_plan) else 9999

    if "complet" in estado:
        prob = 1
    else:
        if days_left <= 7:
            prob = 3
        elif days_left <= 14:
            prob = 2
        else:
            prob = 1

    impact = row.get("_impact_tmp", 2)
    return prob, impact

def gantt(df, date_mode="Plan", color_by="Estado"):
    start = df[DATE_COL_START]
    end = df[DATE_COL_END_REAL].fillna(df[DATE_COL_END_PLAN]) if date_mode == "Real" else df[DATE_COL_END_PLAN]

    color_arg = color_by if color_by in df.columns else "Estado"
    color_map = {s: status_color(s) for s in df["Estado"].dropna().unique()} if color_by == "Estado" else None

    fig = px.timeline(
        df,
        x_start=start,
        x_end=end,
        y="Tarea / Entregable",
        color=color_arg,
        color_discrete_map=color_map,
        hover_data=["ID","Fase","Línea","Responsable","Ubicación","%","Depende de","Hito (S/N)","Riesgo clave","Piloto"]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=600, legend_title=color_by, xaxis_title="Fecha", yaxis_title=None)
    return fig

# -------- Carga de datos (sin uploader) --------
DEFAULT_XLSX = Path(__file__).parent / "linea_tiempo_proyecto_eolico.xlsx"
FALLBACK_XLSX = Path.home() / "Desktop" / "Linea de tiempo" / "linea_tiempo_proyecto_eolico.xlsx"

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    # IDs
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

    # Fechas a datetime
    for c in [DATE_COL_START, DATE_COL_END_PLAN, DATE_COL_END_REAL]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.NaT

    # 'Depende de' seguro para Arrow/pyarrow (no mezclar int/str)
    if "Depende de" in df.columns:
        df["Depende de"] = df["Depende de"].apply(lambda v: "" if pd.isna(v) else (",".join(map(str, parse_dependencies(v)))))
        df["_deps"] = df["Depende de"].apply(parse_dependencies)
    else:
        df["Depende de"] = ""
        df["_deps"] = [[] for _ in range(len(df))]

    # Duraciones
    df["DuraciónPlan(d)"] = (df[DATE_COL_END_PLAN] - df[DATE_COL_START]).dt.days
    df["DuraciónReal(d)"] = (df[DATE_COL_END_REAL] - df[DATE_COL_START]).dt.days
    df["DuraciónPlan(d)"] = pd.to_numeric(df["DuraciónPlan(d)"], errors="coerce")
    df["DuraciónReal(d)"] = pd.to_numeric(df["DuraciónReal(d)"], errors="coerce")

    # Piloto (10 kW / 55 kW)
    df["Piloto"] = df.apply(infer_piloto, axis=1)

    return df

@st.cache_data
def load_data_from_path(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)  # requiere openpyxl
    return process_df(df)

def load_with_fallback(path_text: str) -> pd.DataFrame:
    p = Path(path_text).expanduser()
    if p.exists():
        return load_data_from_path(p)
    for alt in [DEFAULT_XLSX, FALLBACK_XLSX]:
        if alt.exists():
            st.sidebar.info(f"Usando archivo: {alt}")
            return load_data_from_path(alt)
    st.error(f"No encuentro el Excel.\nIntentado:\n- {p}\n- {DEFAULT_XLSX}\n- {FALLBACK_XLSX}")
    st.stop()

# -------- Sidebar --------
st.sidebar.title("⚙️ Control")
today = st.sidebar.date_input("Fecha de referencia (hoy)", value=date.today(), key="today")
st.sidebar.markdown("**Excel de línea de tiempo (ruta):**")
custom_path = st.sidebar.text_input("Ruta del Excel", value=str(DEFAULT_XLSX), help="Si no está aquí, se intenta en ~/Desktop/Linea de tiempo/...")
use_manual_upload = st.sidebar.checkbox("Usar carga manual (uploader)", value=False)

if use_manual_upload:
    uploaded = st.sidebar.file_uploader("Sube el Excel", type=["xlsx"])
    if uploaded is None:
        st.stop()
    df = process_df(pd.read_excel(uploaded))
else:
    df = load_with_fallback(custom_path)

# -------- Filtros (incluye Piloto) --------
cols_filter = st.sidebar.container()
fases = sorted(df["Fase"].dropna().unique().tolist()) if "Fase" in df.columns else []
lineas = sorted(df["Línea"].dropna().unique().tolist()) if "Línea" in df.columns else []
resp = sorted(df["Responsable"].dropna().unique().tolist()) if "Responsable" in df.columns else []
estado = sorted(df["Estado"].dropna().unique().tolist()) if "Estado" in df.columns else []
pilotos = sorted(df["Piloto"].dropna().unique().tolist()) if "Piloto" in df.columns else []

sel_proy = cols_filter.multiselect("Piloto", pilotos, default=pilotos)
sel_fase = cols_filter.multiselect("Fase", fases, default=fases)
sel_linea = cols_filter.multiselect("Línea", lineas, default=lineas)
sel_resp = cols_filter.multiselect("Responsable", resp, default=resp)
sel_estado = cols_filter.multiselect("Estado", estado, default=estado)

mask = (
    df["Piloto"].isin(sel_proy)
    & df["Fase"].isin(sel_fase)
    & df["Línea"].isin(sel_linea)
    & df["Responsable"].isin(sel_resp)
    & df["Estado"].isin(sel_estado)
)
fdf = df[mask].copy()

# -------- KPIs --------
st.title("🚀 Tablero Proyecto Eólico")
st.caption("Línea de tiempo, KPIs, ruta crítica y riesgos")

total = len(fdf)
done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
pending = (fdf["Estado"].str.contains("Pend", case=False, na=False)).sum()
late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(st.session_state.today))
        & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
milestones = (fdf["Hito (S/N)"].astype(str).str.upper() == "S").sum() if "Hito (S/N)" in fdf.columns else 0
progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Tareas totales", total)
col2.metric("Completadas", done)
col3.metric("En curso", in_course)
col4.metric("Planificadas", planned)
col5.metric("Pendientes", pending)
col6.metric("Atrasadas", late)
if not np.isnan(progress_avg):
    st.progress(float(progress_avg)/100.0, text=f"Avance promedio: {progress_avg:.0f}%")

# -------- Tabs --------
tab_timeline, tab_burnup, tab_crit, tab_risk, tab_table = st.tabs(
    ["📅 Línea de tiempo", "📈 Burn‑up / Avance", "🧩 Ruta crítica", "⚠️ Riesgos", "📋 Tabla"]
)

with tab_timeline:
    st.subheader("Gantt")
    colopt1, colopt2 = st.columns([1,2])
    with colopt1:
        mode = st.radio("Fechas", ["Plan", "Real"], horizontal=True)
    with colopt2:
        color_by = st.radio("Color por", ["Estado", "Piloto"], horizontal=True)
    fig_gantt = gantt(fdf, date_mode=mode, color_by=color_by)
    st.plotly_chart(fig_gantt, use_container_width=True)

with tab_burnup:
    st.subheader("Burn‑up de tareas completadas")
    df_burn = fdf.copy()
    df_burn["_fecha_done"] = df_burn[DATE_COL_END_REAL].fillna(df_burn[DATE_COL_END_PLAN])
    df_burn = df_burn.dropna(subset=["_fecha_done"]).sort_values("_fecha_done")
    if not df_burn.empty:
        daily = df_burn.groupby("_fecha_done")["ID"].count().rename("Completadas_día")
        cum = daily.cumsum().rename("Completadas_acum").to_frame()
        cum["Totales"] = total
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum["Completadas_acum"], mode="lines+markers", name="Completadas acum"))
        fig.add_trace(go.Scatter(x=cum.index, y=cum["Totales"], mode="lines", name="Total tareas", line=dict(dash="dash")))
        fig.update_layout(height=450, margin=dict(l=5, r=5, t=30, b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay tareas con fecha de término para graficar el burn‑up.")

def build_graph(df):
    G = nx.DiGraph()
    for _, r in df.iterrows():
        dur = r.get("DuraciónPlan(d)")
        dur = int(dur) if pd.notna(dur) else 1
        G.add_node(int(r["ID"]), duration=max(1, dur))
    for _, r in df.iterrows():
        i = int(r["ID"])
        for d in r["_deps"]:
            if d in G.nodes:
                G.add_edge(int(d), i)
    return G

def longest_path_by_duration(G):
    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        H.add_node(n, duration=data.get("duration", 1))
    for u, v in G.edges():
        H.add_edge(u, v, w=G.nodes[v].get("duration", 1))
    try:
        order = list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible:
        return [], 0
    dist = {n: H.nodes[n].get("duration", 1) for n in H.nodes()}
    prev = {n: None for n in H.nodes()}
    for n in order:
        for _, v, data in H.out_edges(n, data=True):
            w = data["w"]
            if dist[n] + w > dist[v]:
                dist[v] = dist[n] + w
                prev[v] = n
    end = max(dist, key=dist.get)
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[end]

with tab_crit:
    st.subheader("Ruta crítica (por duración planificada)")
    G = build_graph(fdf)
    path, total_dur = longest_path_by_duration(G)
    if path:
        df_path = fdf.set_index("ID").loc[path, ["Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN, "DuraciónPlan(d)"]].reset_index()
        df_path["Orden"] = range(1, len(df_path)+1)
        colA, colB = st.columns([2,1])
        with colA:
            st.markdown("**Secuencia (ID → Tarea)**")
            st.table(df_path[["Orden","ID","Tarea / Entregable","DuraciónPlan(d)"]])
        with colB:
            st.metric("Duración total ruta crítica (días)", int(total_dur))
        fig_cp = px.timeline(df_path, x_start=DATE_COL_START, x_end=DATE_COL_END_PLAN, y="Tarea / Entregable",
                             color_discrete_sequence=["#d62728"])
        fig_cp.update_yaxes(autorange="reversed")
        fig_cp.update_layout(height=350, margin=dict(l=5, r=5, t=30, b=5), showlegend=False)
        st.plotly_chart(fig_cp, use_container_width=True)
    else:
        st.warning("No se pudo calcular la ruta crítica (revisa que no existan ciclos en dependencias).")

with tab_risk:
    st.subheader("Matriz de riesgos (probabilidad vs impacto)")
    out_degree = dict(G.out_degree())
    fdf["_impact_tmp"] = fdf["ID"].map(
        lambda i: 3 if (out_degree.get(int(i),0)>=2 or str(fdf.set_index('ID').loc[int(i), "Hito (S/N)"]).upper()=="S")
        else (2 if out_degree.get(int(i),0)==1 else 1)
    )
    probs, imps = [], []
    for _, r in fdf.iterrows():
        p, im = compute_risk(r, st.session_state.today)
        probs.append(p)
        imps.append(im)
    fdf["Probabilidad(1-3)"] = probs
    fdf["Impacto(1-3)"] = imps
    fdf["Severidad"] = fdf["Probabilidad(1-3)"] * fdf["Impacto(1-3)"]

    fig_r = go.Figure()
    for i in range(1,4):
        for j in range(1,4):
            fig_r.add_shape(type="rect", x0=i-0.5, x1=i+0.5, y0=j-0.5, y1=j+0.5,
                            line=dict(width=1, color="#cccccc"), fillcolor="rgba(240,240,240,0.4)")
    fig_r.add_trace(go.Scatter(
        x=fdf["Probabilidad(1-3)"], y=fdf["Impacto(1-3)"], mode="markers+text",
        text=fdf["ID"].astype(str), textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="#333"), color=fdf["Severidad"]),
        hovertext=fdf["Tarea / Entregable"], hoverinfo="text"
    ))
    fig_r.update_xaxes(range=[0.5,3.5], dtick=1, title="Probabilidad")
    fig_r.update_yaxes(range=[0.5,3.5], dtick=1, title="Impacto")
    fig_r.update_layout(height=450, margin=dict(l=5, r=5, t=30, b=5), coloraxis_showscale=False)
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("**Top 10 por severidad**")
    top_r = fdf.sort_values("Severidad", ascending=False)[[
        "ID","Tarea / Entregable","Estado","Piloto","Probabilidad(1-3)","Impacto(1-3)","Severidad","Riesgo clave","Mitigación breve",
        DATE_COL_END_PLAN, "%"
    ]].head(10)
    st.dataframe(top_r, use_container_width=True)

with tab_table:
    st.subheader("Tabla filtrada")
    show_cols = ["ID","Piloto","Fase","Línea","Tarea / Entregable","Método","Ubicación","Responsable","Estado",
                 DATE_COL_START, DATE_COL_END_PLAN, DATE_COL_END_REAL, "%","Depende de","Hito (S/N)",
                 "DuraciónPlan(d)","DuraciónReal(d)"]
    show_cols = [c for c in show_cols if c in fdf.columns]
    st.dataframe(fdf[show_cols].sort_values([DATE_COL_START, DATE_COL_END_PLAN]), use_container_width=True)

with st.expander("🧠 Recomendaciones de uso"):
    st.markdown("""
- **Piloto**: filtra por *Piloto 10 kW* o *Piloto 55 kW* desde la barra lateral. Si no viene en el Excel, se infiere automáticamente.
- **Depende de**: usa IDs separados por coma (ej. `8,14`).
- **Fechas reales**: al completar `Fin real`, el burn‑up y el modo “Real” del Gantt reflejan avance efectivo.
- **Campos extra**: `Prioridad`, `NivelRiesgo`, `CostoPlan`, `CostoReal`, `CriterioAceptacion`, `EvidenciaURL`.
    """)
