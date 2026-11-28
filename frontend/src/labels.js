/**
 * Display names for class / event labels. The data keeps its own names (they're what the
 * backend and exports use); this only makes common codes readable on screen. A name the
 * user typed themselves (PreprocessingConfig.class_names) arrives already readable and
 * passes through unchanged apart from underscores.
 */
const FRIENDLY = {
  // Sleep-EDF after the AASM mapping in mne_fetcher.prepare_sleep_annotations
  wake: "Wake",
  W: "Wake",
  N1: "N1 (light sleep)",
  N2: "N2",
  N3: "N3 (deep sleep)",
  REM: "REM sleep",
  // PhysioNet motor imagery's raw codes, seen before the preset renames them
  T0: "Rest",
};

/** "left_hand" -> "Left hand", "N3" -> "N3 (deep sleep)". */
export function prettyLabel(name) {
  const s = String(name);
  if (FRIENDLY[s]) return FRIENDLY[s];
  const spaced = s.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
