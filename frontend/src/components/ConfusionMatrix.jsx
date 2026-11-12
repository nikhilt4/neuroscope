function ConfusionMatrix({ matrix, classNames }) {
  if (!matrix || matrix.length === 0) return null;

  // Row-normalised: each cell is "of the trials that truly were X, what
  // fraction were predicted as Y". Raw counts mislead when classes are
  // unbalanced, so colour by proportion and show the count as text.
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0) || 1);

  const shade = (value, total) => {
    const p = value / total;
    return { background: `rgba(99, 102, 241, ${p * 0.85})`,
             color: p > 0.5 ? "white" : "#1f2937" };
  };

  return (
    <div className="confusion-wrap">
      <table className="confusion-table">
        <thead>
          <tr>
            <th className="cm-corner">true \ predicted</th>
            {classNames.map((c) => <th key={c}>{c}</th>)}
            <th className="cm-recall">recall</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classNames[i]}>
              <th>{classNames[i]}</th>
              {row.map((v, j) => (
                <td key={j} style={shade(v, rowTotals[i])}
                    title={`${v} of ${rowTotals[i]} ${classNames[i]} trials predicted ${classNames[j]}`}>
                  {v}
                </td>
              ))}
              <td className="cm-recall">
                {((row[i] / rowTotals[i]) * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ConfusionMatrix;