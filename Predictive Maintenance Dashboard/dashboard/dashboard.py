import os
import sys
import time
from collections import deque

import pandas as pd
from queue import Queue, Empty
import plotly.graph_objects as go
import streamlit as st

# Ensure src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulator import RealTimeSimulator
from predict import predict_failure
from explain import get_prediction_explanation


st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide', page_icon='⚙️')

st.markdown("""
<style>
body { background-color: #0e1117; }
.stApp { color-scheme: dark; }
</style>
""", unsafe_allow_html=True)

# Load Font Awesome for professional icons (fallback to CDN)
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-pO1m8R+YkqK1s8eZPp4rjvQm2FQeKx3a5+g0J4h3QzxQ6qVQe6Vf3p1Q1vJ6Qe6vZK5s1K6eQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)
# KPI card styles
st.markdown("""
<style>
.kpi-row { display: flex; gap: 12px; }
.kpi-card { background: #0f1720; border-radius: 8px; padding: 18px; width:100%; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
.kpi-title { color: #9aa4ad; font-size:14px; margin-bottom:6px; }
.kpi-value { font-size:28px; font-weight:700; color: #ffffff; }
.kpi-icon { font-size:22px; margin-right:8px; }
.kpi-row .card-inner { display:flex; align-items:center; justify-content:space-between; }
.kpi-badge { padding:6px 10px; border-radius:6px; color:#0a0a0a; font-weight:600; }
.kpi-green { background:#2ecc71; }
.kpi-yellow { background:#f1c40f; }
.kpi-red { background:#e74c3c; }
.kpi-gray { background:#7f8c8d; }

/* Machine Failure Analysis styling */
.mfa-title { font-size:28px; font-weight:700; color:#ffffff; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:12px; }
.mfa-block { background:transparent; padding:8px 0 16px 0; }
.mfa-row { display:flex; gap:12px; align-items:center; margin-bottom:12px; }
.mfa-field-label { font-size:16px; font-weight:600; color:#9aa4ad; margin-bottom:4px; }
.mfa-field-value { font-size:20px; font-weight:700; color:#ffffff; }
.mfa-risk { font-size:26px; font-weight:700; color:#ffffff; margin-left:8px; }
.mfa-status-badge { display:inline-block; padding:6px 10px; border-radius:6px; color:#0a0a0a; font-weight:700; font-size:16px; }
.mfa-subheading { font-size:20px; font-weight:600; color:#ffffff; padding-top:12px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.04); margin-bottom:8px; }
.mfa-feature-list { list-style:disc; margin-left:18px; color:#c9d1d9; font-size:16px; line-height:1.6; }
.mfa-feature-list li { margin-bottom:6px; }
.mfa-section { padding:8px 0 16px 0; }
</style>
""", unsafe_allow_html=True)


def kpi_card(title, value, delta=None):
    st.metric(label=title, value=value, delta=delta)


def init_session():
    if 'history' not in st.session_state:
        st.session_state.history = deque(maxlen=300)
    if 'counts' not in st.session_state:
        st.session_state.counts = {'total': 0, 'healthy': 0, 'warning': 0, 'critical': 0}
    # compatibility: keep old history key as alias
    if 'simulator_running' not in st.session_state:
        st.session_state.simulator_running = False
    if 'sensor_queue' not in st.session_state:
        st.session_state.sensor_queue = None
    if 'sensor_history' not in st.session_state:
        # keep as deque for efficient appends and maxlen
        st.session_state.sensor_history = deque(maxlen=300)


def update_state(sample, result):
    ts = sample.get('timestamp', time.time())
    entry = {**sample, **result, 'ts': ts}
    st.session_state.sensor_history.append(entry)
    # also maintain legacy history alias
    st.session_state.history.append(entry)
    st.session_state.counts['total'] = st.session_state.counts.get('total', 0) + 1
    status = result.get('status', 'Healthy')
    key = status.lower()
    if key in st.session_state.counts:
        st.session_state.counts[key] += 1


def build_time_series():
    df = pd.DataFrame(list(st.session_state.sensor_history))
    if df.empty:
        return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    return df


def main():
    st.title('Predictive Maintenance — Industrial Dashboard')
    init_session()

    # KPI cards using containers and styled HTML for a modern industrial look
    k1, k2, k3, k4 = st.columns(4)

    total = st.session_state.counts.get('total', 0)
    healthy = st.session_state.counts.get('healthy', 0)
    warning = st.session_state.counts.get('warning', 0)
    critical = st.session_state.counts.get('critical', 0)

    with k1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='card-inner'>
                        <div>
                            <div class='kpi-title'><i class='fa-solid fa-cpu' style='margin-right:8px;color:#9aa4ad'></i>Total Machines</div>
                            <div class='kpi-value'>{total}</div>
                        </div>
                        <div><div class='kpi-badge kpi-gray'>ALL</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with k2:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='card-inner'>
                        <div>
                            <div class='kpi-title'><i class='fa-solid fa-circle-check' style='margin-right:8px;color:#2ecc71'></i>Healthy Machines</div>
                            <div class='kpi-value' style='color:#2ecc71'>{healthy}</div>
                        </div>
                        <div><div class='kpi-badge kpi-green'>OK</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with k3:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='card-inner'>
                        <div>
                            <div class='kpi-title'><i class='fa-solid fa-triangle-exclamation' style='margin-right:8px;color:#f1c40f'></i>Warning Machines</div>
                            <div class='kpi-value' style='color:#f1c40f'>{warning}</div>
                        </div>
                        <div><div class='kpi-badge kpi-yellow'>WARN</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with k4:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='card-inner'>
                        <div>
                            <div class='kpi-title'><i class='fa-solid fa-circle-exclamation' style='margin-right:8px;color:#e74c3c'></i>Critical Machines</div>
                            <div class='kpi-value' style='color:#e74c3c'>{critical}</div>
                        </div>
                        <div><div class='kpi-badge kpi-red'>CRIT</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Real-time simulator controls
    st.sidebar.header('Simulator')
    start_clicked = st.sidebar.button('Start Simulator')
    stop_clicked = st.sidebar.button('Stop Simulator')

    if 'sim' not in st.session_state:
        st.session_state.sim = None

    # Create a thread-safe queue for communication between simulator thread and Streamlit
    if st.session_state.sensor_queue is None:
        st.session_state.sensor_queue = Queue()

    if start_clicked and not st.session_state.simulator_running:
        # start simulator thread; it will put samples into sensor_queue
        st.session_state.sim = RealTimeSimulator(interval=1.0)
        st.session_state.sim.start(queue=st.session_state.sensor_queue)
        st.session_state.simulator_running = True
        st.sidebar.success('Simulator started')

    if stop_clicked and st.session_state.simulator_running:
        if st.session_state.sim:
            st.session_state.sim.stop()
            st.session_state.sim = None
        st.session_state.simulator_running = False
        st.sidebar.info('Simulator stopped')

    # Consume any queued sensor samples produced by the simulator thread
    queued = 0
    q = st.session_state.sensor_queue
    if q is not None:
        while True:
            try:
                sample = q.get_nowait()
            except Empty:
                break
            # For each sample, compute prediction in main thread and update state
            try:
                res = predict_failure(
                    sample['Type'], sample['Air_temperature_K'], sample['Process_temperature_K'],
                    sample['Rotational_speed_rpm'], sample['Torque_Nm'], sample['Tool_wear_min']
                )
            except Exception:
                res = {'risk_percent': 0.0, 'status': 'Healthy', 'prediction': 0, 'recommendations': []}
            update_state(sample, res)
            queued += 1

    # If we processed new samples, request a rerun once to refresh UI (safe use)
    if queued > 0:
        try:
            st.rerun()
        except Exception:
            # In some Streamlit versions rerun may not be allowed at this point; ignore
            pass

    # Historical trends
    df = build_time_series()
    st.subheader('Real-Time Sensor Monitoring')
    if df is None or df.empty:
        st.warning('No data yet. Start the simulator to populate streams.')
    else:
        # Latest readings (most recent sample)
        latest = df.iloc[-1]

        # KPI cards and status distribution
        kpi_col, donut_col = st.columns([3, 1])
        with kpi_col:
            kp1, kp2, kp3, kp4, kp5 = st.columns(5)
            kp1.markdown(f"<div style='background:#0f1720;padding:10px;border-radius:8px;text-align:center'><div style='color:#9aa4ad;font-size:12px'>Process Temp (K)</div><div style='font-size:20px;font-weight:700;color:#ffffff'>{latest.get('Process_temperature_K',0):.2f}</div></div>", unsafe_allow_html=True)
            kp2.markdown(f"<div style='background:#0f1720;padding:10px;border-radius:8px;text-align:center'><div style='color:#9aa4ad;font-size:12px'>Air Temp (K)</div><div style='font-size:20px;font-weight:700;color:#ffffff'>{latest.get('Air_temperature_K',0):.2f}</div></div>", unsafe_allow_html=True)
            kp3.markdown(f"<div style='background:#0f1720;padding:10px;border-radius:8px;text-align:center'><div style='color:#9aa4ad;font-size:12px'>RPM</div><div style='font-size:20px;font-weight:700;color:#ffffff'>{latest.get('Rotational_speed_rpm',0):.1f}</div></div>", unsafe_allow_html=True)
            kp4.markdown(f"<div style='background:#0f1720;padding:10px;border-radius:8px;text-align:center'><div style='color:#9aa4ad;font-size:12px'>Torque (Nm)</div><div style='font-size:20px;font-weight:700;color:#ffffff'>{latest.get('Torque_Nm',0):.2f}</div></div>", unsafe_allow_html=True)
            kp5.markdown(f"<div style='background:#0f1720;padding:10px;border-radius:8px;text-align:center'><div style='color:#9aa4ad;font-size:12px'>Tool Wear (min)</div><div style='font-size:20px;font-weight:700;color:#ffffff'>{latest.get('Tool_wear_min',0):.2f}</div></div>", unsafe_allow_html=True)

        with donut_col:
            counts = st.session_state.get('counts', {'healthy':0,'warning':0,'critical':0})
            labels = ['Healthy','Warning','Critical']
            values = [counts.get('healthy',0), counts.get('warning',0), counts.get('critical',0)]
            fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=['#2ecc71','#f1c40f','#e74c3c']))])
            fig_donut.update_layout(template='plotly_dark', showlegend=True, margin=dict(t=10,b=10,l=10,r=10), height=200)
            st.plotly_chart(fig_donut, use_container_width=True)

        # ----- Live Data Table (last 20 readings) -----
        st.markdown('**Live Data (last 20 readings)**')
        table_df = df.tail(20).copy()
        if not table_df.empty:
            table_df['datetime'] = pd.to_datetime(table_df['ts'], unit='s')
            table_df = table_df.sort_values('datetime', ascending=False)

            # Machine selector above analysis
            product_ids = [str(x) for x in table_df['Product_ID'].tolist()]
            # default to previously selected machine if present and in list, else choose latest
            default_idx = 0
            prev = st.session_state.get('selected_machine')
            if prev in product_ids:
                try:
                    default_idx = product_ids.index(prev)
                except Exception:
                    default_idx = 0

            selected_pid = st.selectbox('Select Machine', options=product_ids, index=default_idx, key='machine_selector')

            # find matching entry in sensor_history for the selected pid
            entries = list(st.session_state.sensor_history)
            found = None
            for e in reversed(entries):
                if str(e.get('Product_ID')) == str(selected_pid):
                    found = e
                    break
            if found:
                st.session_state['selected_machine'] = selected_pid
                st.session_state['selected_entry'] = found

            cols = ['Product_ID', 'datetime', 'Process_temperature_K', 'Air_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'status', 'risk_percent']
            tbl_html = """
            <div style='max-height:360px;overflow:auto;border-radius:8px;padding:6px;background:#0b0f14'>
            <table style='width:100%;border-collapse:collapse'>
              <thead>
                <tr>
            """
            for c in cols:
                tbl_html += f"<th style='text-align:left;padding:8px;border-bottom:1px solid #222;color:#9aa4ad'>{c}</th>"
            tbl_html += "</tr></thead><tbody>"

            sel = st.session_state.get('selected_machine')
            for _, r in table_df.iterrows():
                status = str(r.get('status', '')).lower()
                if status == 'healthy':
                    bg = '#07260a'
                    color = '#2ecc71'
                elif status == 'warning':
                    bg = '#2b2200'
                    color = '#f1c40f'
                elif status == 'critical':
                    bg = '#2b0a0a'
                    color = '#e74c3c'
                else:
                    bg = '#071018'
                    color = '#95a5a6'

                pid = str(r.get('Product_ID', ''))
                # selected row highlight
                if sel and pid == str(sel):
                    row_bg = '#12314a'
                    row_style = f"background:{row_bg};border-left:4px solid #3aa0ff;"
                else:
                    row_style = f"background:{bg};"

                dt = r.get('datetime')
                dt_str = dt.strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(dt) else ''
                tbl_html += f"<tr style='{row_style}'>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#9fb3ff'>{pid}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{dt_str}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('Process_temperature_K',0)):.2f}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('Air_temperature_K',0)):.2f}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('Rotational_speed_rpm',0)):.1f}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('Torque_Nm',0)):.2f}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('Tool_wear_min',0)):.2f}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:{color};font-weight:700'>{r.get('status','')}</td>"
                tbl_html += f"<td style='padding:8px;border-bottom:1px solid #111;color:#c9d1d9'>{float(r.get('risk_percent',0)):.2f}%</td>"
                tbl_html += "</tr>"

            tbl_html += "</tbody></table></div>"
            st.markdown(tbl_html, unsafe_allow_html=True)
        else:
            st.info('No recent readings to display in Live Data table.')

        # ----- Machine Failure Analysis (below table) -----
        selected_pid = st.session_state.get('selected_machine')
        selected_entry = st.session_state.get('selected_entry')
        if selected_pid and selected_entry is not None:
            risk = selected_entry.get('risk_percent', None)
            status = selected_entry.get('status', 'Unknown')

            # Determine badge color (preserve existing colors)
            if str(status).lower() == 'healthy':
                badge_bg = '#2ecc71'
                badge_text = '#01240b'
            elif str(status).lower() == 'warning':
                badge_bg = '#f1c40f'
                badge_text = '#2f2600'
            elif str(status).lower() == 'critical':
                badge_bg = '#e74c3c'
                badge_text = '#2a0b0b'
            else:
                badge_bg = '#95a5a6'
                badge_text = '#0a0a0a'

            # Title and primary fields (styled)
            st.markdown(f"""
            <div class='mfa-block'>
              <div class='mfa-title'>Machine Failure Analysis</div>
              <div class='mfa-row'>
                <div>
                  <div class='mfa-field-label'>Machine ID</div>
                  <div class='mfa-field-value'>{selected_pid}</div>
                </div>
                <div>
                  <div class='mfa-field-label'>Failure Probability</div>
                  <div class='mfa-risk'>{(float(risk) if risk is not None else 0.0):.2f}%</div>
                </div>
                <div>
                  <div class='mfa-field-label'>Status</div>
                  <div><span class='mfa-status-badge' style='background:{badge_bg}; color:#ffffff'>{status}</span></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Build input dataframe for SHAP explanation using latest sensor values
            try:
                tval = selected_entry.get('Type')
                type_num = 0 if tval == 'L' else (1 if tval == 'M' else 2)
                inp_df = pd.DataFrame([{
                    'Type': type_num,
                    'Air_temperature_K': float(selected_entry.get('Air_temperature_K', 0)),
                    'Process_temperature_K': float(selected_entry.get('Process_temperature_K', 0)),
                    'Rotational_speed_rpm': float(selected_entry.get('Rotational_speed_rpm', 0)),
                    'Torque_Nm': float(selected_entry.get('Torque_Nm', 0)),
                    'Tool_wear_min': float(selected_entry.get('Tool_wear_min', 0)),
                    'temp_diff': float(selected_entry.get('Process_temperature_K', 0)) - float(selected_entry.get('Air_temperature_K', 0)),
                    'power': float(selected_entry.get('Rotational_speed_rpm', 0)) * float(selected_entry.get('Torque_Nm', 0)),
                    'stress': float(selected_entry.get('Torque_Nm', 0)) * float(selected_entry.get('Tool_wear_min', 0))
                }])
                explanation = get_prediction_explanation(inp_df)
                feats = explanation.get('top_features', [])

                # Subheading for top factors
                st.markdown("<div class='mfa-subheading'>Top factors affecting prediction</div>", unsafe_allow_html=True)

                # Feature list (styled)
                if feats:
                    items_html = "<ul class='mfa-feature-list'>"
                    for f in feats:
                        name = f.get('feature')
                        sv = f.get('shap_value', 0.0)
                        impact = f.get('impact')
                        text = 'increases risk' if impact == 'increases_risk' else ('decreases risk' if impact == 'decreases_risk' else 'neutral')
                        items_html += f"<li><strong style='display:inline-block;width:160px'>{name}</strong> <span style='color:#ffffff'>{sv:.4f}</span> <span style='color:#9aa4ad'>({text})</span></li>"
                    items_html += "</ul>"
                    st.markdown(items_html, unsafe_allow_html=True)

                # Plotly signed bar chart (unchanged)
                st.plotly_chart(explanation['plotly_figure'], use_container_width=True)
            except Exception as e:
                st.write('Explain failed:', e)

            # Recommendations (keep content unchanged)
            recs = selected_entry.get('recommendations', [])
            st.markdown("<div style='padding-top:8px'><strong>Recommended Actions:</strong></div>", unsafe_allow_html=True)
            if recs:
                for r in recs:
                    st.write('-', r)
            else:
                st.write('- Routine inspection recommended')
        else:
            st.info('Select a machine from the live table to inspect its prediction and explanation.')

        # ----- Charts grid (2x2) below analysis -----
        top1, top2 = st.columns(2)
        bot1, bot2 = st.columns(2)

        # helper: prepare critical markers
        crit_mask = False
        if 'status' in df.columns:
            try:
                crit_mask = df['status'].astype(str).str.lower() == 'critical'
            except Exception:
                crit_mask = df['status'] == 'Critical'

        # Temperature Trend (Process and Air) with thresholds
        with top1:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=df['datetime'], y=df['Process_temperature_K'], mode='lines', name='Process Temp', line=dict(color='#ff7f0e')))
            fig_temp.add_trace(go.Scatter(x=df['datetime'], y=df['Air_temperature_K'], mode='lines', name='Air Temp', line=dict(color='#1f77b4')))
            # critical points
            if isinstance(crit_mask, pd.Series) and crit_mask.any():
                fig_temp.add_trace(go.Scatter(x=df.loc[crit_mask,'datetime'], y=df.loc[crit_mask,'Process_temperature_K'], mode='markers', name='Critical', marker=dict(color='#e74c3c',size=8)))
            # Thresholds
            fig_temp.add_hline(y=305, line=dict(color='#2ecc71',dash='dash'), annotation_text='Temp Normal', annotation_position='bottom right', opacity=0.6)
            fig_temp.add_hline(y=312, line=dict(color='#f1c40f',dash='dash'), annotation_text='Temp Warning', annotation_position='bottom right', opacity=0.6)
            fig_temp.add_hline(y=316, line=dict(color='#e74c3c',dash='dash'), annotation_text='Temp Critical', annotation_position='bottom right', opacity=0.6)
            fig_temp.update_layout(template='plotly_dark', title='Temperature (K)', height=300, legend=dict(orientation='h'))
            st.plotly_chart(fig_temp, use_container_width=True)

        # RPM Trend
        with top2:
            fig_rpm = go.Figure()
            fig_rpm.add_trace(go.Scatter(x=df['datetime'], y=df['Rotational_speed_rpm'], mode='lines+markers', name='RPM', line=dict(color='#00cc96')))
            if isinstance(crit_mask, pd.Series) and crit_mask.any():
                fig_rpm.add_trace(go.Scatter(x=df.loc[crit_mask,'datetime'], y=df.loc[crit_mask,'Rotational_speed_rpm'], mode='markers', name='Critical', marker=dict(color='#e74c3c',size=8)))
            fig_rpm.update_layout(template='plotly_dark', title='RPM', height=300, legend=dict(orientation='h'))
            st.plotly_chart(fig_rpm, use_container_width=True)

        # Torque Trend with thresholds
        with bot1:
            fig_torque = go.Figure()
            fig_torque.add_trace(go.Scatter(x=df['datetime'], y=df['Torque_Nm'], mode='lines', name='Torque', line=dict(color='#ab63fa')))
            if isinstance(crit_mask, pd.Series) and crit_mask.any():
                fig_torque.add_trace(go.Scatter(x=df.loc[crit_mask,'datetime'], y=df.loc[crit_mask,'Torque_Nm'], mode='markers', name='Critical', marker=dict(color='#e74c3c',size=8)))
            fig_torque.add_hline(y=60, line=dict(color='#2ecc71',dash='dash'), annotation_text='Torque Normal', annotation_position='bottom right', opacity=0.6)
            fig_torque.add_hline(y=70, line=dict(color='#f1c40f',dash='dash'), annotation_text='Torque Warning', annotation_position='bottom right', opacity=0.6)
            fig_torque.add_hline(y=70.0001, line=dict(color='#e74c3c',dash='dash'), annotation_text='Torque Critical', annotation_position='bottom right', opacity=0.6)
            fig_torque.update_layout(template='plotly_dark', title='Torque (Nm)', height=300, legend=dict(orientation='h'))
            st.plotly_chart(fig_torque, use_container_width=True)

        # Tool Wear Trend with thresholds
        with bot2:
            fig_tool = go.Figure()
            fig_tool.add_trace(go.Scatter(x=df['datetime'], y=df['Tool_wear_min'], mode='lines', name='Tool Wear', line=dict(color='#ff9896')))
            if isinstance(crit_mask, pd.Series) and crit_mask.any():
                fig_tool.add_trace(go.Scatter(x=df.loc[crit_mask,'datetime'], y=df.loc[crit_mask,'Tool_wear_min'], mode='markers', name='Critical', marker=dict(color='#e74c3c',size=8)))
            fig_tool.add_hline(y=120, line=dict(color='#2ecc71',dash='dash'), annotation_text='Tool Normal', annotation_position='bottom right', opacity=0.6)
            fig_tool.add_hline(y=180, line=dict(color='#f1c40f',dash='dash'), annotation_text='Tool Warning', annotation_position='bottom right', opacity=0.6)
            fig_tool.add_hline(y=180.0001, line=dict(color='#e74c3c',dash='dash'), annotation_text='Tool Critical', annotation_position='bottom right', opacity=0.6)
            fig_tool.update_layout(template='plotly_dark', title='Tool Wear (min)', height=300, legend=dict(orientation='h'))
            st.plotly_chart(fig_tool, use_container_width=True)

        # Failure Risk Trend (last 50 readings)
        df_risk = pd.DataFrame(list(st.session_state.sensor_history))
        if not df_risk.empty and 'risk_percent' in df_risk.columns:
            df_risk = df_risk.tail(50)
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(x=pd.to_datetime(df_risk['ts'], unit='s'), y=df_risk['risk_percent'], mode='lines+markers', line=dict(color='#ffa15a')))
            fig_risk.update_layout(template='plotly_dark', title='Failure Risk Trend (last 50)', height=300, yaxis=dict(range=[0,100]), legend=dict(orientation='h'))
            st.plotly_chart(fig_risk, use_container_width=True)

        # (Duplicate live-data block removed; table and selector above are authoritative)

    st.subheader('Failure Prediction')
    with st.form('predict_form'):
        c1, c2, c3 = st.columns(3)
        Type = c1.selectbox('Type', ['L', 'M', 'H'])
        Air_temperature_K = c2.number_input('Air Temperature (K)', value=295.0)
        Process_temperature_K = c3.number_input('Process Temperature (K)', value=305.0)
        rpm = c1.number_input('RPM', value=1500.0)
        torque = c2.number_input('Torque (Nm)', value=40.0)
        tool = c3.number_input('Tool wear (min)', value=50.0)
        submitted = st.form_submit_button('Predict')

    if submitted:
        try:
            out = predict_failure(Type, Air_temperature_K, Process_temperature_K, rpm, torque, tool)
            st.metric('Failure Probability', f"{out['risk_percent']}%")
            st.write('Status:', out['status'])
            st.write('Recommendations:')
            for r in out['recommendations']:
                st.write('-', r)

            # Explain
            # Build single-row DataFrame expected by explain module
            inp_df = pd.DataFrame([{
                'Type': 0 if Type == 'L' else (1 if Type == 'M' else 2),
                'Air_temperature_K': Air_temperature_K,
                'Process_temperature_K': Process_temperature_K,
                'Rotational_speed_rpm': rpm,
                'Torque_Nm': torque,
                'Tool_wear_min': tool,
                'temp_diff': Process_temperature_K - Air_temperature_K,
                'power': rpm * torque,
                'stress': torque * tool
            }])

            explanation = get_prediction_explanation(inp_df)
            st.subheader('Why did the model predict this?')
            # Show signed SHAP impacts (positive increases risk, negative decreases)
            feats = explanation.get('top_features', [])
            if feats:
                with st.expander('Top 5 contributing features'):
                    for f in feats:
                        feat = f.get('feature')
                        sv = f.get('shap_value', 0.0)
                        impact = f.get('impact', 'neutral')
                        text = 'increases risk' if impact == 'increases_risk' else ('decreases risk' if impact == 'decreases_risk' else 'neutral')
                        st.write(f"{feat}: {sv:.4f} ({text})")
                # Plotly signed bar chart
                st.plotly_chart(explanation['plotly_figure'], use_container_width=True)
            else:
                st.info('No SHAP explanation available for this prediction.')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

    st.subheader('Maintenance Recommendations (rules)')
    st.write('- If temperature high → Inspect cooling system')
    st.write('- If torque high → Inspect motor load')
    st.write('- If tool wear high → Replace tool')
    st.write('- If stress high → Check bearings')


if __name__ == '__main__':
    main()
