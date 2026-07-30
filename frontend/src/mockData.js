export const MOCK_UPLOAD_DATA = {
  session_id: "preview-session",
  format: "edf",
  n_channels: 22,
  channels: ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "Fz", "Cz", "Pz"],
  sampling_rate: 250,
  duration_seconds: 272.4,
  events_found: true,
  event_classes: ["left_hand", "right_hand", "feet", "tongue"],
  available_modes: ["motor_imagery", "spectral", "basic"],
  recommended_mode: "motor_imagery",
  needs_column_picker: false,
  needs_variable_picker: false,
  needs_event_setup: false,
  is_ready: true,
  source: "upload",
  motor_channels: {
    motor_imagery_possible: true,
    primary_motor: ["c3", "c4", "cz"],
    extended_motor: ["cp3", "cp4", "fc3", "fc4"],
    frontal: ["f3", "f4", "f7", "f8", "fz"],
    parietal: ["p3", "p4", "pz"],
    occipital: ["o1", "o2"],
    temporal: [],
  },
};

export const MOCK_RESULTS = {
  dataset_summary: {
    file: "subject01.edf",
    duration: "4m 32s",
    channels: 22,
    sampling_rate: 250,
    epochs: 144,
  },
  classifiers: [
    { name: "LDA + CSP", accuracy: 78.3, std: 4.2, time_ms: 12 },
    { name: "SVM + CSP", accuracy: 81.1, std: 3.8, time_ms: 45 },
    { name: "Random Forest", accuracy: 74.6, std: 5.1, time_ms: 28 },
    { name: "EEGNet (CNN)", accuracy: 83.4, std: 3.1, time_ms: 210 },
  ],
};