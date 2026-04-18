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
MIN_CONTOUR_AREA = 3000         
MAX_CONTOUR_AREA = 200000
MIN_NODE_WIDTH = 40
MIN_NODE_HEIGHT = 40            
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# Stage 1: CV quality filters
MIN_SOLIDITY = 0.4              # contour_area / convex_hull_area — filters noise blobs
EDGE_MARGIN = 15                # px — ignore contours whose bbox touches this close to frame edge
MAX_MARKS = 20                  # cap: keep only the top-N detections by contour area per frame

# Stage 1: SoM keyframe overlay 
MARK_FONT_SCALE = 0.8
MARK_NODE_COLOR = (0, 200, 0)       # green
MARK_CONNECTION_COLOR = (200, 0, 0) # red
CENTROID_MATCH_THRESHOLD = 50   # px — max distance to match element across frames

# Stage 2: VLM
VLM_MODEL = "gemini-2.5-flash"
VLM_TIMEOUT = 1200

# Stage 3: Mealy
IDLE_TIMEOUT = 30.0  # seconds with no activity → emit τ