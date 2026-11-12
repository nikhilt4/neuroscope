import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceArea, ResponsiveContainer,
} from "recharts";

const BAND_COLORS = {
  delta: "#f3f4f6", theta: "#eef2ff", alpha: "#e0e7ff",
  beta: "#f3f4f6", gamma: "#fafafa",
};

const CLASS_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

function PSDChart({ psd }) {
  if (!psd) return null;


  
  const names = Object.keys(psd.curves);
  // recharts wants row-per-x, so transpose from {class: [values]} to
  // [{freq, class1, class2, ...}]


  const [lo, hi] = psd.passband || [psd.freqs[0], psd.freqs[psd.freqs.length - 1]];


  const data = psd.freqs
    .map((f, i) => {
      const row = { freq: Number(f.toFixed(1)) };
      names.forEach((n) => { row[n] = psd.curves[n][i]; });
      return row;
    })
    .filter((row) => row.freq >= lo && row.freq <= hi);

  
  

  return (
    <div className="chart-block">
      <h3 className="subheading">Power Spectral Density</h3>
      <p className="group-hint">
        Average power at each frequency, per class. Separation between curves
        means the classes differ in frequency content — which is what the
        band-power features pick up.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          {psd.bands && Object.entries(psd.bands)
            .filter(([, [bLo, bHi]]) => bHi > lo && bLo < hi)
            .map(([band, [bLo, bHi]]) => (
          <ReferenceArea key={band} x1={Math.max(bLo, lo)} x2={Math.min(bHi, hi)}
                fill={BAND_COLORS[band]} fillOpacity={0.6}
                label={{ value: band, position: "insideTop",
                  style: { fontSize: 9, fill: "#9ca3af" } }} />
          ))}
          <XAxis dataKey="freq" type="number" domain={["dataMin", "dataMax"]}
                 tick={{ fontSize: 11 }}
                 label={{ value: "Frequency (Hz)", position: "insideBottom", offset: -5,
                          style: { fontSize: 11, fill: "#6b7280" } }} />
          <YAxis tick={{ fontSize: 11 }}
                 label={{ value: "log₁₀ power", angle: -90, position: "insideLeft",
                          style: { fontSize: 11, fill: "#6b7280" } }} />
          <Tooltip formatter={(v) => v.toFixed(2)}
                   labelFormatter={(f) => `${f} Hz`}
                   contentStyle={{ fontSize: 12, borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />
          {names.map((n, i) => (
            <Line key={n} type="monotone" dataKey={n} dot={false} strokeWidth={1.8}
                  stroke={CLASS_COLORS[i % CLASS_COLORS.length]} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PSDChart;