import streamlit as st
import pandas as pd
import re
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Cytek Log Parser v.1.0.0",
    layout="wide"
)

# ---- Helper Functions ----

def strip_timestamp(s):
    s = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', s)
    s = re.sub(r'\b\d{4}-\d{1,2}-\d{1,2}\b', '', s)
    s = re.sub(r'\b\d{4}/\d{1,2}/\d{1,2}\b', '', s)
    s = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(am|pm)?\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d{1,2}\s*(am|pm)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(am|pm)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[-:\.,/]{2,}', ' ', s)
    s = re.sub(r'[\s:\-\[\],\.]+', ' ', s)
    return s.strip()

def parse_log_file(file):
    errors = []
    lines = []
    file.seek(0)
    for i, line in enumerate(file):
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='ignore')
        lines.append(line.rstrip('\n'))
        if "error" in line.lower():
            errors.append((i, line.rstrip('\n')))
    return lines, errors

def aggregate_errors(logs_errors):
    summary = {}
    total = 0
    for errors in logs_errors.values():
        for _, err in errors:
            err_stripped = err.strip()
            no_time = strip_timestamp(err_stripped)
            no_time_lower = no_time.lower()
            # Remove extra symbols and 'Error' prefix for grouping
            new_no_time = re.sub(r'^error(\s*\([^\)]*\))?[\s:\-\[\]\(\)\.]*', '', no_time, flags=re.IGNORECASE)
            new_no_time = re.sub(r'^[^a-zA-Z]+', '', new_no_time)
            if new_no_time:
                new_no_time = new_no_time[0].upper() + new_no_time[1:]
            # Group certain Cytek errors
            if "cytek device readafpgaregister" in no_time_lower:
                key = "Cytek device readafpgaregister"
            elif "cytek device readmfpgaregisterrmt" in no_time_lower:
                key = "Cytek device readmfpgaregisterrmt"
            elif "cytek device writemfpgaregisterrmt" in no_time_lower:
                key = "Cytek device writemfpgaregisterrmt"
            else:
                key = new_no_time
            summary[key] = summary.get(key, 0) + 1
            total += 1
    # Sort errors descending
    display_summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
    percentages = {k: f"{int(v)} ({(100*v/total):.1f}%)" for k, v in display_summary.items()} if total else {}
    return display_summary, percentages

def pareto_chart(df):
    fig, ax = plt.subplots(figsize=(10, max(4, len(df)*0.45)))
    bar_vals = [int(p.split()[0]) for p in df["Count (Percent)"]]
    bar_containers = ax.barh(
        df["Error"], bar_vals, color="#3399FF", edgecolor="#2176ae", height=0.5, zorder=2
    )
    ax.invert_yaxis()
    ax.set_title("Errors (Pareto)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Count", fontsize=11)
    ax.set_ylabel("Error", fontsize=11)
    ax.grid(True, axis='x', color="#d9e8f6", linestyle="--", linewidth=0.8, zorder=1)
    max_bar = max(bar_vals) if bar_vals else 1
    ax.set_xlim(0, max_bar * 1.15)
    for i, txt in enumerate(df["Count (Percent)"]):
        bar_end = bar_vals[i]
        chart_max = max_bar * 1.15
        label_x = bar_end + max_bar * 0.02
        text_width = len(txt) * 0.015 * chart_max
        if label_x + text_width > chart_max:
            label_x = chart_max - text_width - max_bar * 0.02
        ax.text(label_x, i, txt, va='center', fontsize=10, color="#2176ae", fontweight='bold')
    plt.subplots_adjust(left=0.32, right=0.98)
    fig.tight_layout()
    return fig

# ---- Streamlit App ----

st.title("Cytek Flow Cytometer Log Parser v.1.0.0")
st.markdown("**Upload your log files (can select multiple):**")

uploaded_files = st.file_uploader(
    "Upload .txt log files",
    type=["txt"], accept_multiple_files=True, key="file_uploader"
)

# -- File select list (checkboxes) --
if uploaded_files:
    # Use a stable name key for each file
    files_metadata = [
        {
            "name": file.name,
            "mtime": file.file_id if hasattr(file, "file_id") else 0,  # Streamlit doesn't provide mtime, but name is enough
            "idx": idx
        }
        for idx, file in enumerate(uploaded_files)
    ]
    # Find latest ApplicationLog
    app_logs = [f for f in files_metadata if "applicationlog" in f["name"].lower()]
    if app_logs:
        default_idx = max(app_logs, key=lambda f: f["name"])[
            "idx"
        ]  # Use "latest" by name sort (approximation for upload)
    else:
        default_idx = 0

    # --- Sidebar for file selection ---
    st.sidebar.subheader("Select log files for analysis (Pareto):")
    file_checks = []
    for idx, meta in enumerate(files_metadata):
        checked = True
        file_checks.append(
            st.sidebar.checkbox(meta["name"], value=checked, key=f"chk_{idx}")
        )
    # Bulk select/deselect/app/setupengine toggles
    cols = st.sidebar.columns(2)
    if cols[0].button("Select All"):
        for idx in range(len(file_checks)):
            st.session_state[f"chk_{idx}"] = True
    if cols[1].button("Deselect All"):
        for idx in range(len(file_checks)):
            st.session_state[f"chk_{idx}"] = False
    cols2 = st.sidebar.columns(2)
    if cols2[0].button("Select ApplicationLog"):
        for idx, meta in enumerate(files_metadata):
            st.session_state[f"chk_{idx}"] = "applicationlog" in meta["name"].lower()
    if cols2[1].button("Select SetupEngineLog"):
        for idx, meta in enumerate(files_metadata):
            st.session_state[f"chk_{idx}"] = "setupenginelog" in meta["name"].lower()

    # -- Main file selector: always highlight latest ApplicationLog by default
    st.subheader("View a Log File")
    selected_file = st.selectbox(
        "Choose a log file to view its content:",
        [meta["name"] for meta in files_metadata],
        index=default_idx
    )

    # Parse all files once and store in dict
    parsed_files = {}
    all_errors = {}
    for idx, file in enumerate(uploaded_files):
        lines, errors = parse_log_file(file)
        parsed_files[file.name] = lines
        all_errors[file.name] = errors

    # -- Log content search --
    log_search = st.text_input("Search log content:")
    error_search = st.text_input("Search errors:")

    # --- Log content and error display (side by side) ---
    left, right = st.columns(2)
    # --- Log content ---
    with left:
        st.markdown("#### Log Content")
        lines = parsed_files[selected_file]
        if log_search:
            filtered = [
                (i, line)
                for i, line in enumerate(lines)
                if log_search.lower() in line.lower()
            ]
        else:
            filtered = list(enumerate(lines))
        highlighted = set(i for i, line in all_errors[selected_file])
        # Show as code for monospace
        log_disp = [
            f"{i+1:>5}: " +
            (f"**:red[{line}]**" if i in highlighted else line)
            for i, line in filtered
        ]
        st.code("\n".join(log_disp), language="")

    # --- Error List ---
    with right:
        st.markdown("#### Error Entries")
        errors = all_errors[selected_file]
        if error_search:
            err_filtered = [
                (i, err)
                for i, err in errors
                if error_search.lower() in err.lower()
            ]
        else:
            err_filtered = errors
        error_disp = [
            f"Line {i+1}: {err}" for i, err in err_filtered
        ]
        st.code("\n".join(error_disp), language="")

    # --- Pareto chart and summary (selected files only) ---
    st.divider()
    st.subheader("Errors (Pareto)")
    selected_names = [
        meta["name"] for idx, meta in enumerate(files_metadata)
        if st.session_state.get(f"chk_{idx}", True)
    ]
    sel_errors = {name: all_errors[name] for name in selected_names}
    display_summary, percentages = aggregate_errors(sel_errors)
    df = pd.DataFrame([(k, percentages[k]) for k in display_summary.keys()],
                      columns=["Error", "Count (Percent)"])
    if not df.empty:
        fig = pareto_chart(df)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        st.pyplot(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            # PNG Export
            st.download_button(
                label="Export Pareto PNG",
                data=buf.getvalue(),
                file_name=f"Pareto {datetime.now().strftime('%m-%d %H-%M')}.png",
                mime="image/png"
            )
        with col2:
            # CSV Export
            csv_buf = StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Export Pareto CSV",
                data=csv_buf.getvalue(),
                file_name=f"Pareto {datetime.now().strftime('%m-%d %H-%M')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No error data in the selected files.")

else:
    st.info("Upload at least one .txt log file to get started.")

