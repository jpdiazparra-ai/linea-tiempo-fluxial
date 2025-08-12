# app.py
# Requisitos base: streamlit, pandas, plotly, numpy, networkx, python-dateutil
# Sugeridos para exportar: reportlab o fpdf2, y kaleido
# pip install streamlit pandas plotly numpy networkx python-dateutil reportlab fpdf2 kaleido

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date
from io import BytesIO
import tempfile, os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Línea de Tiempo Proyecto Eólico", layout="wide")

# Fallback para 'today' si corres en bare mode (sin contexto Streamlit)
if "today" not in st.session_state:
    st.session_state["today"] = date.today()

# -------- Config fuente de datos (Google Sheets CSV) --------
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3l1P_rQzwYqDter4G4tuM4z7ZvoOGruhh__QfIigyQeDNgJqF-qDYs0z7zjjL-mwblzejzotblwtr/pub?gid=0&single=true&output=csv"

# -------- Helpers --------
DATE_COL_START = "Inicio (AAAA-MM-DD)"
DATE_COL_END_PLAN = "Fin plan (AAAA-MM-DD)"
DATE_COL_END_REAL = "Fin real"

def parse_dependencies(s):
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
        return "#de1a3b"  # rojo
    if "planific" in s:
        return "#1778e1"  # azul
    if "recurrente" in s:
        return "#C5C213"  # amarillo
    if "pend" in s:
        return "#ff7f0e"  # naranja
    return "#7f7f7f"      # gris

def infer_piloto(row):
    val = str(row.get("Piloto", "")).strip()
    if val:
        return val
    texto = " ".join([
        str(row.get("Fase", "")),
        str(row.get("Línea", "")),
        str(row.get("Tarea / Entregable", "")),
        str(row.get("Método", "")),
    ]).lower()
    if "55" in texto or "55kw" in texto or "55 k" in texto:
        return "Piloto 55 kW"
    try:
        if int(row.get("ID", 0)) in {24, 25}:
            return "Piloto 55 kW"
    except Exception:
        pass
    return "Piloto 10 kW"

def compute_risk(row, today):
    estado = str(row.get("Estado", "")).lower()
    fin_plan = row.get(DATE_COL_END_PLAN)
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

def build_burnup_fig(df):
    df_burn = df.copy()
    df_burn["_fecha_done"] = df_burn[DATE_COL_END_REAL].fillna(df_burn[DATE_COL_END_PLAN])
    df_burn = df_burn.dropna(subset=["_fecha_done"]).sort_values("_fecha_done")
    if df_burn.empty:
        return None
    daily = df_burn.groupby("_fecha_done")["ID"].count().rename("Completadas_día")
    cum = daily.cumsum().rename("Completadas_acum").to_frame()
    cum["Totales"] = len(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum["Completadas_acum"], mode="lines+markers", name="Completadas acum"))
    fig.add_trace(go.Scatter(x=cum.index, y=cum["Totales"], mode="lines", name="Total tareas", line=dict(dash="dash")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# -------- Exportadores --------
def make_pdf_summary(df, fdf, fig_gantt=None, fig_burn=None, project_title="Tablero Proyecto Eólico"):
    """Genera PDF con reportlab; si falla, intenta fpdf2; si falta kaleido, omite imágenes."""
    total = len(fdf)
    done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
    in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
    planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + \
              (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
    today_value = st.session_state.get("today", date.today())
    late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(today_value))
            & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
    progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

    today_ts = pd.to_datetime(today_value)
    soon = fdf[(fdf[DATE_COL_START] >= today_ts) &
               (fdf[DATE_COL_START] <= today_ts + pd.Timedelta(days=60))] \
            .sort_values(DATE_COL_START) \
            .head(6)[["ID","Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN]].copy()

    # Top riesgos simple
    finp = fdf[DATE_COL_END_PLAN]
    days_left = (finp - today_ts).dt.days
    prob = np.where(fdf["Estado"].str.contains("complet", case=False, na=False), 1,
                    np.where(days_left<=7,3, np.where(days_left<=14,2,1)))
    sev = prob * 1  # impacto básico=1 si no calculas con grafo
    rtop = fdf.assign(Severidad=sev).sort_values("Severidad", ascending=False) \
              .loc[:, ["ID","Tarea / Entregable","Severidad"]].head(6)

    # ---- Intento 1: reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

        def _img(fig, wcm, hcm):
            if fig is None: return None
            try:
                png = fig.to_image(format="png", scale=2, width=1000, height=400)  # requiere kaleido
                return Image(BytesIO(png), width=wcm*cm, height=hcm*cm)
            except Exception:
                return None

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.2*cm, bottomMargin=1.2*cm,
                                leftMargin=1.4*cm, rightMargin=1.4*cm)
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceAfter=6)
        p  = styles["BodyText"]

        elements = [
            Paragraph(project_title, h1),
            Paragraph(f"Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}", p),
            Spacer(1, 0.4*cm)
        ]

        kpi_data = [
            ["Tareas totales", total, "Completadas", done, "En curso", in_course],
            ["Planificadas", planned, "Atrasadas", late, "Avance prom.", f"{0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%"],
        ]
        kpi_tbl = Table(kpi_data, colWidths=[3.2*cm, 2.2*cm, 3.2*cm, 2.2*cm, 3.2*cm, 2.2*cm])
        kpi_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
            ("BOX",(0,0),(-1,-1), 0.5, colors.grey),
            ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ]))
        elements += [kpi_tbl, Spacer(1, 0.4*cm)]

        g_img = _img(fig_gantt, 17, 6)
        if g_img: elements += [Paragraph("Línea de tiempo (snapshot)", h2), g_img, Spacer(1, 0.3*cm)]
        b_img = _img(fig_burn, 17, 5)
        if b_img: elements += [Paragraph("Burn-up (completadas acumuladas)", h2), b_img]

        elements += [PageBreak(), Paragraph("Próximos hitos (60 días)", h2)]
        if not soon.empty:
            s = soon.copy()
            s[DATE_COL_START]    = s[DATE_COL_START].dt.strftime("%Y-%m-%d")
            s[DATE_COL_END_PLAN] = s[DATE_COL_END_PLAN].dt.strftime("%Y-%m-%d")
            soon_tbl = Table([["ID","Tarea","Inicio","Fin plan"]] + s.values.tolist(),
                             colWidths=[1.5*cm, 9.0*cm, 3.5*cm, 3.5*cm])
            soon_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
                                          ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            elements.append(soon_tbl)
        else:
            elements.append(Paragraph("No hay hitos en las próximas 8 semanas.", p))

        elements += [Spacer(1, 0.3*cm), Paragraph("Top riesgos por severidad", h2)]
        if not rtop.empty:
            rt_tbl = Table([["ID","Tarea","Severidad"]] + rtop.values.tolist(),
                           colWidths=[1.5*cm, 12.0*cm, 3.0*cm])
            rt_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.whitesmoke),
                                        ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            elements.append(rt_tbl)
        else:
            elements.append(Paragraph("No se identifican riesgos destacados.", p))

        doc.build(elements)
        return buf.getvalue()

    except Exception:
        # ---- Intento 2: fpdf2
        try:
            from fpdf import FPDF
        except Exception:
            return None  # sin libs → fallback HTML

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 8, project_title)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 6, f"Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}", ln=1)
        kpis = f"Tareas: {total} | Completadas: {done} | En curso: {in_course} | Planif.: {planned} | Atrasadas: {late} | Avance prom.: {0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%"
        pdf.multi_cell(0, 6, kpis)
        pdf.ln(2)

        def _add_fig(fig, w=190, h=80):
            if fig is None: return
            try:
                png = fig.to_image(format="png", scale=2, width=1000, height=400)  # kaleido
            except Exception:
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png); path = tmp.name
            try:
                pdf.image(path, w=w, h=h); pdf.ln(2)
            finally:
                try: os.remove(path)
                except: pass

        _add_fig(fig_gantt, 190, 80)
        _add_fig(fig_burn,  190, 70)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, "Próximos hitos (60 días)", ln=1)
        pdf.set_font("Helvetica", size=10)
        if not soon.empty:
            for _, r in soon.iterrows():
                s_ini = r[DATE_COL_START].strftime("%Y-%m-%d")
                s_fin = r[DATE_COL_END_PLAN].strftime("%Y-%m-%d")
                pdf.multi_cell(0, 5, f"#{int(r['ID'])} | {r['Tarea / Entregable']} | {s_ini} → {s_fin}")
        else:
            pdf.multi_cell(0, 5, "No hay hitos próximos.")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, "Top riesgos", ln=1)
        pdf.set_font("Helvetica", size=10)
        if not rtop.empty:
            for _, r in rtop.iterrows():
                pdf.multi_cell(0, 5, f"#{int(r['ID'])} | {r['Tarea / Entregable']} | Sev: {int(r['Severidad'])}")
        else:
            pdf.multi_cell(0, 5, "Sin riesgos destacados.")

        out = BytesIO()
        pdf.output(out, dest="S")
        return out.getvalue()

def make_html_summary(df, fdf, fig_gantt=None, fig_burn=None, project_title="Tablero Proyecto Eólico"):
    total = len(fdf)
    done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
    in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
    planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + \
              (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
    today_value = st.session_state.get("today", date.today())
    late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(today_value))
            & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
    progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

    today_ts = pd.to_datetime(today_value)
    soon = fdf[(fdf[DATE_COL_START] >= today_ts) &
               (fdf[DATE_COL_START] <= today_ts + pd.Timedelta(days=60))] \
            .sort_values(DATE_COL_START) \
            .head(6)[["ID","Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN]].copy()
    if not soon.empty:
        soon[DATE_COL_START]    = soon[DATE_COL_START].dt.strftime("%Y-%m-%d")
        soon[DATE_COL_END_PLAN] = soon[DATE_COL_END_PLAN].dt.strftime("%Y-%m-%d")

    g_html = pio.to_html(fig_gantt, full_html=False, include_plotlyjs='cdn') if fig_gantt else ""
    b_html = pio.to_html(fig_burn,  full_html=False, include_plotlyjs=False) if fig_burn else ""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{project_title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
.kpis {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 8px 0 16px; }}
.kpis div {{ background:#f6f6f6; padding:8px 10px; border:1px solid #ddd; border-radius:8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border:1px solid #ddd; padding:6px 8px; text-align:left; }}
th {{ background:#fafafa; }}
</style></head><body>
<h1>{project_title}</h1>
<p>Fecha de generación: {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
<div class="kpis">
  <div><b>Tareas</b><br>{total}</div>
  <div><b>Completadas</b><br>{done}</div>
  <div><b>En curso</b><br>{in_course}</div>
  <div><b>Planificadas</b><br>{planned}</div>
  <div><b>Atrasadas</b><br>{late}</div>
  <div><b>Avance prom.</b><br>{0 if np.isnan(progress_avg) else round(float(progress_avg),1)}%</div>
</div>
<h2>Línea de tiempo (snapshot)</h2>{g_html}
<h2>Burn-up</h2>{b_html}
<h2>Próximos hitos (60 días)</h2>
{soon.to_html(index=False) if not soon.empty else "<p>No hay hitos en las próximas 8 semanas.</p>"}
</body></html>"""
    return html.encode("utf-8")

# -------- Visual (Gantt) --------
def gantt(df, date_mode="Plan", color_by="Estado"):
    dfp = df.copy()
    dfp["_start"]     = pd.to_datetime(dfp.get(DATE_COL_START),     errors="coerce")
    dfp["_end_plan"]  = pd.to_datetime(dfp.get(DATE_COL_END_PLAN),  errors="coerce")
    dfp["_end_real"]  = pd.to_datetime(dfp.get(DATE_COL_END_REAL),  errors="coerce")

    dfp["_start"] = dfp["_start"].fillna(dfp["_end_real"]).fillna(dfp["_end_plan"])
    dfp["_end"] = dfp["_end_real"] if date_mode == "Real" else dfp["_end_plan"]
    dfp["_end"] = dfp["_end"].fillna(dfp["_end_plan"]).fillna(dfp["_end_real"]).fillna(dfp["_start"])

    bad_dur = dfp["_end"] <= dfp["_start"]
    dfp.loc[bad_dur, "_end"] = dfp.loc[bad_dur, "_start"] + pd.Timedelta(days=1)

    valid_mask = dfp["_start"].notna() & dfp["_end"].notna()
    dfp = dfp[valid_mask].copy()

    dfp = dfp.sort_values(["_start", "_end", "ID"], ascending=[False, False, True])
    cat_order = pd.unique(dfp["Tarea / Entregable"])

    color_arg = color_by if color_by in dfp.columns else "Estado"
    color_map = {s: status_color(s) for s in dfp["Estado"].dropna().unique()} if color_by == "Estado" else None

    fig = px.timeline(
        dfp,
        x_start="_start",
        x_end="_end",
        y="Tarea / Entregable",
        color=color_arg,
        color_discrete_map=color_map,
        category_orders={"Tarea / Entregable": cat_order},
        hover_data=["ID","Fase","Línea","Responsable","Ubicación","%","Depende de","Hito (S/N)","Riesgo clave","Piloto"]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=600, legend_title=color_by,
                      xaxis_title="Fecha", yaxis_title=None)
    st.session_state["gantt_adjusted"] = int(bad_dur.sum())
    st.session_state["gantt_omitted"]  = 0
    return fig

def tune_gantt_for_export(fig):
    """Copia la figura de Gantt y la ajusta para exportación (más margen/altura)."""
    f2 = go.Figure(fig)  # copia, no modifica la que se ve en pantalla
    labels = []
    for tr in f2.data:
        y = getattr(tr, "y", None)
        if y is not None:
            labels.extend([str(v) for v in y])
    uniq = list(dict.fromkeys(labels))
    n_rows = len(uniq)
    max_len = max((len(s) for s in uniq), default=10)
    left = min(40 + max_len * 6, 360)       # margen izquierdo amplio para etiquetas
    height = min(max(420, 26 * n_rows), 1600)
    f2.update_yaxes(automargin=True)
    f2.update_layout(margin=dict(l=left, r=24, t=40, b=24), height=height)
    return f2

# -------- ETL --------
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    for c in [DATE_COL_START, DATE_COL_END_PLAN, DATE_COL_END_REAL]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.NaT
    if "Depende de" in df.columns:
        df["Depende de"] = df["Depende de"].apply(lambda v: "" if pd.isna(v) else (",".join(map(str, parse_dependencies(v)))))
        df["_deps"] = df["Depende de"].apply(parse_dependencies)
    else:
        df["Depende de"] = ""
        df["_deps"] = [[] for _ in range(len(df))]
    df["DuraciónPlan(d)"] = (df[DATE_COL_END_PLAN] - df[DATE_COL_START]).dt.days
    df["DuraciónReal(d)"] = (df[DATE_COL_END_REAL] - df[DATE_COL_START]).dt.days
    df["DuraciónPlan(d)"] = pd.to_numeric(df["DuraciónPlan(d)"], errors="coerce")
    df["DuraciónReal(d)"] = pd.to_numeric(df["DuraciónReal(d)"], errors="coerce")
    df["Piloto"] = df.apply(infer_piloto, axis=1)
    return df

@st.cache_data
def load_from_gsheet_csv(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url, encoding="utf-8-sig")
    return process_df(df)

# -------- Sidebar --------
st.sidebar.title("⚙️ Control")
st.sidebar.text_input("Fuente de datos (Google Sheets CSV):", GSHEET_CSV_URL, disabled=True)
today_widget = st.sidebar.date_input("Fecha de referencia (hoy)", value=st.session_state["today"], key="today")

try:
    df = load_from_gsheet_csv(GSHEET_CSV_URL)
except Exception as e:
    st.error(f"No pude leer el CSV de Google Sheets.\nDetalles: {e}")
    st.stop()

if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if st.sidebar.button("🔄 Actualizar datos (limpiar caché)"):
    st.session_state.refresh_key += 1
    st.cache_data.clear()
    st.toast("Datos actualizados desde la fuente")
    st.rerun()
st.sidebar.caption(f"Última actualización: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

# -------- Filtros --------
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

# -------- Header / KPIs --------
st.title("🚀 Tablero Proyecto Eólico")
st.caption("Línea de tiempo, KPIs, ruta crítica y riesgos")

total = len(fdf)
done = (fdf["Estado"].str.contains("Complet", case=False, na=False)).sum()
in_course = (fdf["Estado"].str.contains("curso", case=False, na=False)).sum()
planned = (fdf["Estado"].str.contains("Planific", case=False, na=False)).sum() + (fdf["Estado"].str.contains("Recurrente", case=False, na=False)).sum()
pending = (fdf["Estado"].str.contains("Pend", case=False, na=False)).sum()
late = ((fdf[DATE_COL_END_PLAN] < pd.to_datetime(st.session_state["today"]))
        & (~fdf["Estado"].str.contains("Complet", case=False, na=False))).sum()
milestones = (fdf["Hito (S/N)"].astype(str).str.upper() == "S").sum() if "Hito (S/N)" in fdf.columns else 0
progress_avg = np.nanmean(pd.to_numeric(fdf["%"], errors="coerce")) if "%" in fdf.columns else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Tareas totales", total)
c2.metric("Completadas", done)
c3.metric("En curso", in_course)
c4.metric("Planificadas", planned)
c5.metric("Pendientes", pending)
c6.metric("Atrasadas", late)
if not np.isnan(progress_avg):
    st.progress(float(progress_avg)/100.0, text=f"Avance promedio: {progress_avg:.0f}%")

# -------- Dependencias / Grafo --------
def build_graph(df):
    try:
        import networkx as nx
    except ImportError:
        class DummyGraph:
            def __len__(self): return 0
            def nodes(self, data=False): return []
            def edges(self, data=False): return []
            def out_degree(self): return []
            def in_degree(self): return []
        return DummyGraph()

    G = nx.DiGraph()
    for _, r in df.iterrows():
        if pd.isna(r.get("ID")): continue
        dur = r.get("DuraciónPlan(d)")
        dur = int(dur) if pd.notna(dur) else 1
        G.add_node(int(r["ID"]), duration=max(1, dur))
    for _, r in df.iterrows():
        if pd.isna(r.get("ID")): continue
        i = int(r["ID"])
        for d in r.get("_deps", []):
            if d in G.nodes:
                G.add_edge(int(d), i)
    return G

def longest_path_by_duration(G):
    try:
        import networkx as nx
    except ImportError:
        return [], 0
    try:
        if G is None or len(G) == 0:
            return [], 0
    except Exception:
        return [], 0
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
            w = data.get("w", 1)
            if dist[n] + w > dist[v]:
                dist[v] = dist[n] + w
                prev[v] = n
    if not dist: return [], 0
    end = max(dist, key=dist.get)
    path, cur = [], end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path, dist[end]

def critical_path_fallback(df: pd.DataFrame):
    rows = df.dropna(subset=["ID"]).copy()
    if rows.empty: return [], 0, []
    rows["ID"] = rows["ID"].astype(int)
    dur = rows.set_index("ID")["DuraciónPlan(d)"].fillna(1).clip(lower=1).astype(int).to_dict()
    adj = {i: [] for i in dur}
    indeg = {i: 0 for i in dur}
    invalid_edges = []
    for _, r in rows.iterrows():
        i = int(r["ID"])
        deps = r.get("_deps", []) or []
        for d in deps:
            if d in dur:
                adj[d].append(i); indeg[i] += 1
            else:
                invalid_edges.append((i, d))
    from collections import deque
    q = deque([n for n in dur if indeg[n] == 0])
    dist = {n: dur[n] for n in dur}
    prev = {n: None for n in dur}
    visited = 0
    while q:
        n = q.popleft(); visited += 1
        for v in adj[n]:
            if dist[n] + dur[v] > dist[v]:
                dist[v] = dist[n] + dur[v]; prev[v] = n
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    if visited == 0 or visited < len(dur):
        return [], 0, invalid_edges
    end = max(dist, key=dist.get)
    path = []; cur = end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path, dist[end], invalid_edges

G = build_graph(fdf)

# -------- Tabs --------
tab_timeline, tab_burnup, tab_crit, tab_risk, tab_table = st.tabs(
    ["📅 Línea de tiempo", "📈 Burn-up / Avance", "🧩 Ruta crítica", "⚠️ Riesgos", "📋 Tabla"]
)

with tab_timeline:
    st.subheader("Gantt")
    colopt1, colopt2 = st.columns([1,2])
    with colopt1:
        mode = st.radio("Fechas", ["Plan", "Real"], horizontal=True, key="mode_radio")
    with colopt2:
        color_by = st.radio("Color por", ["Estado", "Piloto"], horizontal=True, key="color_radio")

    fig_gantt = gantt(fdf, date_mode=mode, color_by=color_by)
    st.plotly_chart(fig_gantt, use_container_width=True, key="gantt_display")

    # Aviso bajo el gráfico
    adj = st.session_state.get("gantt_adjusted", 0)
    om  = st.session_state.get("gantt_omitted", 0)
    if adj or om:
        st.caption(f"🔎 Ajustadas: {adj} tareas con duración 0/negativa · Omitidas: {om} sin fechas válidas.")

    # --- Exportación (PDF/HTML) usando una copia ajustada del Gantt ---
    fig_gantt_export = tune_gantt_for_export(fig_gantt)   # copia con más margen/altura
    fig_burn_pdf = build_burnup_fig(fdf)                  # usa filtro aplicado

    pdf_bytes = make_pdf_summary(
        df, fdf,
        fig_gantt=fig_gantt_export,
        fig_burn=fig_burn_pdf,
        project_title="Tablero Proyecto Eólico"
    )
    if pdf_bytes:
        st.download_button(
            "📄 Descargar resumen PDF",
            data=pdf_bytes,
            file_name=f"Resumen_proyecto_{pd.Timestamp.now():%Y%m%d}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf"
        )
    else:
        html_bytes = make_html_summary(
            df, fdf,
            fig_gantt=fig_gantt_export,
            fig_burn=fig_burn_pdf,
            project_title="Tablero Proyecto Eólico"
        )
        st.download_button(
            "🌐 Descargar resumen HTML (imprimir → PDF)",
            data=html_bytes,
            file_name=f"Resumen_proyecto_{pd.Timestamp.now():%Y%m%d}.html",
            mime="text/html",
            use_container_width=True,
            key="dl_html"
        )

with tab_burnup:
    st.subheader("Burn-up de tareas completadas")
    df_burn = fdf.copy()
    df_burn["_fecha_done"] = df_burn[DATE_COL_END_REAL].fillna(df_burn[DATE_COL_END_PLAN])
    df_burn = df_burn.dropna(subset=["_fecha_done"]).sort_values("_fecha_done")
    if not df_burn.empty:
        daily = df_burn.groupby("_fecha_done")["ID"].count().rename("Completadas_día")
        cum = daily.cumsum().rename("Completadas_acum").to_frame()
        cum["Totales"] = len(fdf)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum["Completadas_acum"], mode="lines+markers", name="Completadas acum"))
        fig.add_trace(go.Scatter(x=cum.index, y=cum["Totales"], mode="lines", name="Total tareas", line=dict(dash="dash")))
        fig.update_layout(height=450, margin=dict(l=5, r=5, t=30, b=5))
        st.plotly_chart(fig, use_container_width=True, key="burnup_chart")
    else:
        st.info("Aún no hay tareas con fecha de término para graficar el burn-up.")

with tab_crit:
    st.subheader("Ruta crítica (por duración planificada)")
    try:
        graf_vacio = (G is None or len(G) == 0)
    except Exception:
        graf_vacio = True

    if graf_vacio:
        path, total_dur, invalid = critical_path_fallback(fdf)
    else:
        path, total_dur = longest_path_by_duration(G)
        invalid = []

    if path:
        df_unique = fdf.dropna(subset=["ID"]).drop_duplicates(subset=["ID"], keep="first").copy()
        df_path = df_unique[df_unique["ID"].astype(int).isin(path)].copy()
        df_path["ID"] = df_path["ID"].astype(int)
        df_path["__order"] = pd.Categorical(df_path["ID"], categories=path, ordered=True)
        df_path = df_path.sort_values("__order")

        cols_needed = ["Tarea / Entregable", DATE_COL_START, DATE_COL_END_PLAN, "DuraciónPlan(d)"]
        cols_needed = [c for c in cols_needed if c in df_path.columns]
        df_path = df_path[["ID"] + cols_needed].reset_index(drop=True)
        df_path["Orden"] = range(1, len(df_path)+1)

        colA, colB = st.columns([2,1])
        with colA:
            st.markdown("**Secuencia (ID → Tarea)**")
            st.table(df_path[["Orden","ID","Tarea / Entregable","DuraciónPlan(d)"]])
        with colB:
            st.metric("Duración total ruta crítica (días)", int(total_dur))

        fig_cp = px.timeline(
            df_path, x_start=DATE_COL_START, x_end=DATE_COL_END_PLAN, y="Tarea / Entregable",
            color_discrete_sequence=["#d62728"]
        )
        fig_cp.update_yaxes(autorange="reversed")
        fig_cp.update_layout(height=350, margin=dict(l=5, r=5, t=30, b=5), showlegend=False)
        st.plotly_chart(fig_cp, use_container_width=True, key="critical_path_chart")

        if invalid:
            missing = sorted(set(d for _, d in invalid))
            st.caption(f"⚠️ Dependencias ignoradas hacia IDs inexistentes: {missing}")
    else:
        msg = "Ruta crítica no disponible (sin dependencias válidas o hay ciclos)."
        if graf_vacio:
            msg += " También puede faltar 'networkx' si deseas usar el modo completo."
        st.info(msg)

with tab_risk:
    st.subheader("Matriz de riesgos (probabilidad vs impacto)")
    # out_degree (si no hay networkx, queda vacío)
    try:
        out_degree = dict(G.out_degree()) if (G is not None and len(G) > 0) else {}
    except Exception:
        out_degree = {}

    def _impact_for_id(i):
        try:
            is_hito = str(fdf.set_index("ID").loc[int(i), "Hito (S/N)"]).upper() == "S"
        except Exception:
            is_hito = False
        deg = out_degree.get(int(i), 0)
        return 3 if (deg >= 2 or is_hito) else (2 if deg == 1 else 1)

    fdf["_impact_tmp"] = fdf["ID"].map(lambda i: _impact_for_id(i) if pd.notna(i) else 1)

    today_value = st.session_state.get("today", date.today())
    probs, imps = [], []
    for _, r in fdf.iterrows():
        p, im = compute_risk(r, today_value)
        probs.append(p); imps.append(im)
    fdf["Probabilidad(1-3)"] = probs
    fdf["Impacto(1-3)"] = imps
    fdf["Severidad"] = fdf["Probabilidad(1-3)"] * fdf["Impacto(1-3)"]

    fig_r = go.Figure()
    for i in range(1,4):
        for j in range(1,4):
            fig_r.add_shape(type="rect", x0=i-0.5, x1=i+0.5, y0=j-0.5, y1=j+0.5,
                            line=dict(width=1, color="#cccccc"),
                            fillcolor="rgba(240,240,240,0.4)")
    fig_r.add_trace(go.Scatter(
        x=fdf["Probabilidad(1-3)"], y=fdf["Impacto(1-3)"], mode="markers+text",
        text=fdf["ID"].fillna("").astype(str), textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="#333"), color=fdf["Severidad"]),
        hovertext=fdf["Tarea / Entregable"], hoverinfo="text"
    ))
    fig_r.update_xaxes(range=[0.5,3.5], dtick=1, title="Probabilidad")
    fig_r.update_yaxes(range=[0.5,3.5], dtick=1, title="Impacto")
    fig_r.update_layout(height=450, margin=dict(l=5, r=5, t=30, b=5), coloraxis_showscale=False)
    st.plotly_chart(fig_r, use_container_width=True, key="risk_matrix_chart")

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
- **Fuente**: esta app lee automáticamente el CSV público de Google Sheets configurado arriba.
- **Depende de**: usa IDs separados por coma (ej. `8,14`).
- **Fechas reales**: al completar `Fin real`, el burn-up y el modo “Real” del Gantt reflejan avance efectivo.
- **Campos extra sugeridos**: `Prioridad`, `NivelRiesgo`, `CostoPlan`, `CostoReal`, `CriterioAceptacion`, `EvidenciaURL`.
    """)
