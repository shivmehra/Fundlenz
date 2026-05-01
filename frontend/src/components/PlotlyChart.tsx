import Plot from "react-plotly.js";
import type { ChartPayload } from "../types";

export default function PlotlyChart({ payload }: { payload: ChartPayload }) {
  return (
    <Plot
      data={payload.chart_spec.data as never}
      layout={{
        ...payload.chart_spec.layout,
        paper_bgcolor: "#0d1117",
        plot_bgcolor: "#0d1117",
        font: { color: "#e6edf3" },
        autosize: true,
      }}
      style={{ width: "100%", height: "320px" }}
      config={{ responsive: true, displaylogo: false }}
    />
  );
}
