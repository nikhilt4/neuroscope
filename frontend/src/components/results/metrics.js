/**
 * Metrics derived from a confusion matrix (rows = true class, columns = predicted).
 * Shared by the landing-page preview and the live Results step, so both show
 * identical numbers for identical input.
 */

/** "left_hand" -> "Left hand" */
export const prettyClass = (c) => {
  const s = String(c).replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
};

export const FAMILY = { feature: "classical", epoch: "riemannian", deep: "deep" };

/** Per-class F1 = 2PR / (P + R). Classes never predicted get precision 0, not NaN. */
export function f1Scores(matrix) {
  const n = matrix.length;
  return matrix.map((row, i) => {
    const tp = row[i];
    const actual = row.reduce((a, b) => a + b, 0);
    const predicted = matrix.reduce((a, r) => a + r[i], 0);
    const p = predicted ? tp / predicted : 0;
    const r = actual ? tp / actual : 0;
    return p + r ? (2 * p * r) / (p + r) : 0;
  }).slice(0, n);
}

/** Unweighted mean of per-class F1 — what sklearn calls f1_score(average="macro"). */
export const macroF1 = (matrix) => {
  const f = f1Scores(matrix);
  return f.reduce((a, b) => a + b, 0) / f.length;
};

/**
 * One plain-English sentence about the matrix: the best-recognised class and the
 * most-confused pair (largest off-diagonal share, in either direction).
 */
export function describeConfusion(matrix, classNames) {
  if (!matrix?.length || matrix.length < 2) return null;
  const names = classNames.map(prettyClass);
  const recall = matrix.map((row, i) => row[i] / (row.reduce((a, b) => a + b, 0) || 1));
  const bestIdx = recall.indexOf(Math.max(...recall));

  let worst = { share: 0, a: 0, b: 1 };
  matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((v, j) => {
      if (i !== j && v / total > worst.share) worst = { share: v / total, a: i, b: j };
    });
  });

  const best = `${names[bestIdx]} is recognised best (${(recall[bestIdx] * 100).toFixed(0)}% correct).`;
  if (worst.share < 0.05) return `${best} Very few trials are mixed up between classes.`;
  return `${best} The most common mistake is ${names[worst.a].toLowerCase()} predicted as ${names[worst.b].toLowerCase()} (${(worst.share * 100).toFixed(0)}% of those trials).`;
}
