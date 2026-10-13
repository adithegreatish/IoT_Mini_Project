# Camera Calibration & Parking Bay Setup

## 1. Physical Placement Considerations
- **Height & Angle:** Mount the Zebronics webcam at an elevated position (2.2m to 3.5m above ground) angled downward between 25° and 45°. This reduces occlusion between adjacent vehicles while preserving license plate visibility.
- **Lighting:** Avoid facing directly toward east/west horizons to minimize blinding solar glare during sunrise/sunset.
- **Distance:** For license plate reading, plates must occupy at least 60 horizontal pixels in the image frame. For occupancy-only bays, vehicles can be positioned further back.

## 2. Interactive Polygon Calibration Workflow
1. Navigate to the **Parking Calibration** page in the dashboard (`http://127.0.0.1:8000/calibration`).
2. Click **Capture Reference Frame** to grab an empty or baseline parking lot frame.
3. Click on the canvas to place polygon vertices defining each parking bay.
4. Complete the polygon by clicking the initial vertex (minimum 3 vertices, 4 recommended).
5. Specify:
   - **Slot ID:** e.g., `S01`, `S02`, `S03`
   - **Display Label:** e.g., "Bay 1 (Reserved)", "Bay 2 (Compact)"
   - **Slot Type:** `car`, `motorcycle`, `bus`, `truck`
   - **Enabled Status:** Toggle active/inactive
6. The system automatically validates:
   - Minimum area threshold (rejects tiny accidental clicks)
   - Self-intersection (prevents bow-tie / figure-eight polygons)
   - Unwanted overlaps with neighbouring slots
7. Click **Save Calibration** to persist definitions to `data/calibration.json`.

## 3. Resolution Independence & Coordinate Normalization
- Coordinates are stored as normalized ratios `[x/W, y/H]` in the range `[0.0, 1.0]`.
- When switching camera capture resolutions (e.g., from 720p to 1080p), slot polygons automatically scale accurately without requiring re-calibration.
