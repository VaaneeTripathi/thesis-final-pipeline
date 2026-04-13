# Stage 0: ROI
ROI_MODE = "auto"  # "auto" | "manual"
ROI_MANUAL = None  # (x, y, w, h) if manual

# Stage 1: Temporal segmentation
GMM_HISTORY = 300
GMM_VAR_THRESHOLD = 16
SSIM_SAMPLE_INTERVAL = 3        # compute SSIM every 3rd frame
ACTIVITY_THRESHOLD_HIGH = 0.02  # 2% foreground pixels = activity
ACTIVITY_THRESHOLD_LOW = 0.003  # 0.3% = idle
STABILITY_WINDOW = 15           # consecutive idle frames = pen-lift (~0.5s at 30fps)

# Stage 1: CV detection
MIN_CONTOUR_AREA = 1500
MAX_CONTOUR_AREA = 200000
MIN_NODE_WIDTH = 40
MIN_NODE_HEIGHT = 70
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# Stage 1: SoM
MARK_FONT_SCALE = 0.8
MARK_NODE_COLOR = (0, 255, 0)       # green
MARK_CONNECTION_COLOR = (255, 0, 0)  # blue
MARK_DELTA_HIGHLIGHT = (0, 255, 255) # yellow tint
CENTROID_MATCH_THRESHOLD = 50   # px — max distance to match element across frames

# Stage 2: VLM
VLM_MODEL = "gemini-2.5-flash"
VLM_TIMEOUT = 1200

# Stage 3: Mealy
IDLE_TIMEOUT = 30.0  # seconds with no activity → emit τ