# SonicMind-AI
Real Time Live Sound Engineering Assistant

{
  "capture_id": "roomscan_2025-09-13T14:06:00Z",
  "timestamp": "2025-09-13T14:06:00Z",
  "confidence_overall": 0.83,

  "dimensions_m": {
    "length": 4.8,
    "width": 3.6,
    "height": 2.6,
    "confidence": 0.60
  },

  "surfaces": {
    "floor": {
      "materials": [
        { "label": "wood",         "coverage": 0.65 },
        { "label": "rug_medium",   "coverage": 0.35 }
      ]
    },
    "walls": [
      { "label": "drywall_painted", "coverage": 0.70 },
      { "label": "window_glass",    "coverage": 0.20 },
      { "label": "curtain_heavy",   "coverage": 0.10 }
    ],
    "ceiling": {
      "materials": [
        { "label": "gypsum_board", "coverage": 1.00 }
      ]
    }
  },

  "objects": [
    { "label": "sofa_fabric" },
    { "label": "bookshelf_full" }
  ],

  "people_count": 0,
  "soft_mass_index": 0.55,

  "live_dead_class": { "label": "controlled", "confidence": 0.70 },

  "rt60_s_by_octave": {
    "125": 0.55,
    "250": 0.45,
    "500": 0.40,
    "1000": 0.38,
    "2000": 0.36,
    "4000": 0.35
  },

  "reflection_risk": {
    "early": "med",
    "flutter": "low"
  },

  "large_openings": {
    "window_area_m2": 2.4,
    "door_area_m2": 1.9
  },

  "speaker_mic_layout_guess": {
    "mic_distance_from_wall_m": 0.4,
    "expected_early_reflection_risk_sidewalls": "med"
  }
}
