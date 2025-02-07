import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import webbrowser
import signal
import sys
from types import FrameType
from flask import Flask
from utils.logging import logger

# Load Data Before Startup
df_filtered = pd.read_parquet("./data/TN_900.parquet")

df_filtered["embeddings"] = df_filtered["embeddings"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
)
matrix = np.array(df_filtered["embeddings"].to_list())

# Apply Clustering
# n_clusters = 4
# df_filtered["Cluster"] = KMeans(
#     n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10
# ).fit_predict(matrix)

# t-SNE for Visualization
tsne = TSNE(
    n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
)
vis_dims = tsne.fit_transform(matrix)
df_filtered["x"], df_filtered["y"] = vis_dims[:, 0], vis_dims[:, 1]

df_filtered["case_link"] = "https://app.compfox.io/" + df_filtered["id"]

# Initialize Dash App
app = dash.Dash(__name__)
server = app.server  # Flask server for deployment

app.layout = html.Div(
    [
        dcc.Dropdown(
            id="current-id-dropdown",
            options=[
                {"label": row["name"], "value": row["id"]}
                for _, row in df_filtered.iterrows()
            ],
            placeholder="Select a case to highlight...",
            multi=False,
        ),
        dcc.Graph(
            id="scatter-plot",
            config={
                "scrollZoom": False,
                "displayModeBar": False,
            },
        ),
    ]
)


@app.callback(Output("scatter-plot", "figure"), [Input("current-id-dropdown", "value")])
def update_plot(current_id):
    fig = px.scatter(
        df_filtered,
        x="x",
        y="y",
        color=df_filtered["Cluster"].astype(str),
        hover_data={"name": True, "id": True, "case_link": False},
        title="Legal Case Clusters Tennesse (t-SNE + K-Means)",
        labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2", "color": "Cluster"},
        opacity=0.7,
        size_max=10,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    if current_id:
        selected_df = df_filtered[df_filtered["id"] == current_id]
        fig.add_trace(
            px.scatter(
                selected_df,
                x="x",
                y="y",
                text="name",
                size=np.full(len(selected_df), 10),
                color_discrete_sequence=["red"],
            ).data[0]
        )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
        hovertemplate="<b>%{customdata[0]}</b><br>Case ID: %{customdata[1]}<br><a href='https://app.compfox.io/%{customdata[1]}' target='_blank'>Open Case</a><extra></extra>",
        customdata=df_filtered[["name", "id"]].values,
    )
    return fig


@app.callback(Output("scatter-plot", "clickData"), [Input("scatter-plot", "clickData")])
def open_case(clickData):
    if clickData:
        case_id = clickData["points"][0]["customdata"][1]
        case_url = f"https://app.compfox.io/{case_id}"
        webbrowser.open_new_tab(case_url)
    return None


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")
    from utils.logging import flush

    flush()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    app.run_server(debug=True, host="localhost", port=8080)
