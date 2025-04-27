conflicts = [

  # Head-on conflict
  {"aircraft": [{"x": 19, "y": 20, "level": 140, "target_level": 140, "heading": 90},
                {"x": 31, "y": 20, "level": 140, "target_level": 140, "heading": 270}]},

  # Crossing at 90 degrees
  {"aircraft": [{"x": 20, "y": 23, "level": 300, "target_level": 300, "heading": 360},
                {"x": 27, "y": 30, "level": 300, "target_level": 300, "heading": 270}]},

  # Climbing vs level conflict
  {"aircraft": [{"x": 12, "y": 12, "level": 150, "target_level": 150, "heading": 45},
                {"x": 20, "y": 20, "level": 130, "target_level": 150, "heading": 225}]},

  # Turn-in conflict
  {"aircraft": [{"x": 20, "y": 10, "level": 290, "target_level": 290, "heading": 45},
                {"x": 25, "y": 15, "level": 270, "target_level": 290, "heading": 225}]},

  # Descending vs crossing
  {"aircraft": [{"x": 25, "y": 25, "level": 250, "target_level": 230, "heading": 180},
                {"x": 30, "y": 20, "level": 230, "target_level": 230, "heading": 270}]},

  # 3 aircraft converging
  {"aircraft": [{"x": 11, "y": 11, "level": 250, "target_level": 250, "heading": 45},
                {"x": 11, "y": 23, "level": 250, "target_level": 250, "heading": 135},
                {"x": 23, "y": 17, "level": 250, "target_level": 250, "heading": 270}]},

  # 4-way intersection
  {"aircraft": [{"x": 15, "y": 8, "level": 180, "target_level": 180, "heading": 360},
                {"x": 15, "y": 22, "level": 180, "target_level": 180, "heading": 180},
                {"x": 8, "y": 15, "level": 180, "target_level": 180, "heading": 90},
                {"x": 22, "y": 15, "level": 180, "target_level": 180, "heading": 270}]},

  # Climb into crossing path
  {"aircraft": [{"x": 10, "y": 23, "level": 300, "target_level": 300, "heading": 0},
                {"x": 17, "y": 30, "level": 290, "target_level": 300, "heading": 270}]},

  # Crossing at acute angle
  {"aircraft": [{"x": 8, "y": 8, "level": 330, "target_level": 330, "heading": 45},
                {"x": 18, "y": 16, "level": 330, "target_level": 330, "heading": 225}]},

  # 3-aircraft sandwich
  {"aircraft": [{"x": 8.5, "y": 10, "level": 260, "target_level": 260, "heading": 90},
                {"x": 22.5, "y": 10, "level": 260, "target_level": 260, "heading": 270},
                {"x": 15.5, "y": 19, "level": 260, "target_level": 260, "heading": 180}]},

  # 4 aircraft converging toward same central point
  {"aircraft": [{"x": 9, "y": 9, "level": 270, "target_level": 270, "heading": 45},
                {"x": 21, "y": 9, "level": 270, "target_level": 270, "heading": 315},
                {"x": 21, "y": 21, "level": 270, "target_level": 270, "heading": 225},
                {"x": 9, "y": 21, "level": 270, "target_level": 270, "heading": 135}]
  },

  # 5 aircraft diamond pattern, center descending
  {"aircraft": [{"x": 15, "y": 8, "level": 330, "target_level": 330, "heading": 0},
                {"x": 15, "y": 22, "level": 330, "target_level": 330, "heading": 180},
                {"x": 8, "y": 15, "level": 320, "target_level": 320, "heading": 90},
                {"x": 20, "y": 15, "level": 320, "target_level": 320, "heading": 270},
                {"x": 10, "y": 10, "level": 350, "target_level": 350, "heading": 45}]
  },

  # 4 aircraft: 2 descending, 2 level
  {"aircraft": [{"x": 8, "y": 10, "level": 330, "target_level": 290, "heading": 70},
                {"x": 29, "y": 7, "level": 290, "target_level": 290, "heading": 270},
                {"x": 16, "y": 22, "level": 330, "target_level": 330, "heading": 180},
                {"x": 16, "y": 7, "level": 290, "target_level": 290, "heading": 90}]
  },

  # 5 aircraft: two climbing into 310, others intersecting
  {"aircraft": [{"x": 8, "y": 15, "level": 280, "target_level": 300, "heading": 90},
                {"x": 22, "y": 15, "level": 280, "target_level": 300, "heading": 270},
                {"x": 15, "y": 8, "level": 330, "target_level": 330, "heading": 0},
                {"x": 15, "y": 23, "level": 330, "target_level": 330, "heading": 180},
                {"x": 10, "y": 9, "level": 310, "target_level": 330, "heading": 40}]
  },

  # 4 aircraft: 2 parallel, 2 perpendicular crossing
  {"aircraft": [{"x": 10, "y": 15, "level": 250, "target_level": 250, "heading": 90},
                {"x": 10, "y": 21, "level": 250, "target_level": 250, "heading": 90},
                {"x": 19, "y": 29, "level": 250, "target_level": 250, "heading": 180},
                {"x": 19, "y": 8, "level": 260, "target_level": 260, "heading": 0}]
  },

  # 5 aircraft with one resolving (descending)
  {"aircraft": [{"x": 9, "y": 15, "level": 300, "target_level": 300, "heading": 90},
                {"x": 21, "y": 15, "level": 310, "target_level": 310, "heading": 270},
                {"x": 15, "y": 10, "level": 280, "target_level": 280, "heading": 0},
                {"x": 15, "y": 21, "level": 320, "target_level": 320, "heading": 180},
                {"x": 10, "y": 10, "level": 320, "target_level": 320, "heading": 45}]
  }
]