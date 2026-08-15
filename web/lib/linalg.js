// Cholesky solve for the ridge normal equations.
//
// Port of ranksim/linalg.py. The design matrices are (alliance-appearances x
// teams) with ~30 columns, so a plain O(n^3) Cholesky is instant even in a
// phone's JS engine. tests/test_parity.py asserts this agrees with the Python.

export function cholesky(a) {
  const n = a.length;
  const lower = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += lower[i][k] * lower[j][k];
      if (i === j) {
        let d = a[i][i] - s;
        // Ridge should prevent this; nudge rather than explode.
        if (d <= 0) d = 1e-9;
        lower[i][j] = Math.sqrt(d);
      } else {
        lower[i][j] = (a[i][j] - s) / lower[j][j];
      }
    }
  }
  return lower;
}

export function cholSolve(lower, b) {
  const n = lower.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < i; k++) s += lower[i][k] * y[k];
    y[i] = (b[i] - s) / lower[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = 0;
    for (let k = i + 1; k < n; k++) s += lower[k][i] * x[k];
    x[i] = (y[i] - s) / lower[i][i];
  }
  return x;
}

export function cholInvDiag(lower) {
  const n = lower.length;
  const diag = new Float64Array(n);
  for (let col = 0; col < n; col++) {
    const e = new Float64Array(n);
    e[col] = 1;
    diag[col] = cholSolve(lower, e)[col];
  }
  return diag;
}

// Build (X'X + ridge*I, X'y) from sparse rows of column indices. Every design
// row is an alliance: three 1s and the rest zeros.
export function normalEquations(rows, y, nCols, ridge) {
  const ata = Array.from({ length: nCols }, () => new Float64Array(nCols));
  const aty = new Float64Array(nCols);
  for (let r = 0; r < rows.length; r++) {
    const cols = rows[r];
    const target = y[r];
    for (const i of cols) {
      aty[i] += target;
      for (const j of cols) ata[i][j] += 1;
    }
  }
  for (let i = 0; i < nCols; i++) ata[i][i] += ridge;
  return { ata, aty };
}

export function mean(values) {
  if (!values.length) return 0;
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

export function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
