import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ErrorBar, ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";

const CATEGORY_COLORS = {
  feature: "#6366f1",
  epoch: "#10b981",
  deep: "#f59e0b",
};

function AccuracyChart({ results, chance, bestName }) {
  const data = results
    .filter((r) => r.accuracy_mean !== null)
    .sort((a, b) => b.accuracy_mean - a.accuracy_mean)
    .map((r) => ({
      name: r.classifier,
      accuracy: r.accuracy_mean * 100,
      err: r.accuracy_std * 100,
      category: r.category,
    }));

  if (data.length === 0) return null;

  return (
    <div className="chart-block">
      <h3 className="subheading">Accuracy Comparison</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis dataKey="name" angle={-35} textAnchor="end" height={70}
                 tick={{ fontSize: 11 }} interval={0} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 11 }}
                 label={{ value: "Accuracy (%)", angle: -90, position: "insideLeft",
                          style: { fontSize: 11, fill: "#6b7280" } }} />
          <Tooltip
            formatter={(v, key) => [`${v.toFixed(1)}%`, key === "accuracy" ? "Accuracy" : key]}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          {/* Chance depends on class count — 25% for 4-class, not a fixed 50% */}
          <ReferenceLine y={chance * 100} stroke="#ef4444" strokeDasharray="4 4"
            label={{ value: `chance ${(chance * 100).toFixed(0)}%`,
                     position: "right", style: { fontSize: 10, fill: "#ef4444" } }} />
          <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.name}
                    fill={CATEGORY_COLORS[d.category] || "#9ca3af"}
                    fillOpacity={d.name === bestName ? 1 : 0.65} />
            ))}
            {/* Error bars make the "these differences aren't meaningful" point visually */}
            <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="#374151" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="chart-legend">
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <span key={cat}>
            <i style={{ background: color }} />
            {cat === "feature" ? "Classical" : cat === "epoch" ? "Riemannian" : "Deep"}
          </span>
        ))}
      </div>
    </div>
  );
}

export default AccuracyChart;