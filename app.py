from pathlib import Path
from datetime import date
import tempfile

import streamlit as st
import cv2
import numpy as np
import pandas as pd


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NEUROPULSE | BIOPTIQ",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "profile_name": "",
    "profile_dob": date.today(),
    "profile_notes": "",
    "sessions": [],
    "video_inspection": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown("# NEUROPULSE")

st.sidebar.caption("A BIOPTIQ PRODUCT")

st.sidebar.markdown("---")

st.sidebar.markdown("### WORKSPACE")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Pupil Check",
        "Sessions",
        "My Profile",
        "Research",
        "About"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### PLATFORM NOTE")

st.sidebar.caption(
    "NEUROPULSE is an evolving healthcare technology "
    "prototype exploring visual assessment, vision wellness "
    "intelligence, and neurological signal analysis."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### DEVELOPMENT")

st.sidebar.caption("Research and Development")

st.sidebar.caption("Version 0.2 — Vision Wellness Intelligence")

st.sidebar.markdown("---")

st.sidebar.markdown("**BIOPTIQ**")

st.sidebar.caption("The Future of Vision.")


# ==================================================
# BRAND HEADER
# ==================================================

brand_left, brand_right = st.columns([3, 1])

with brand_left:
    st.title("NEUROPULSE")
    st.caption("SEE THE SIGNAL. UNLOCK BETTER CARE.")

with brand_right:
    st.markdown("### BIOPTIQ")
    st.caption("The Future of Vision.")

st.divider()


# ==================================================
# HELPER FUNCTIONS & VISION WELLNESS ENGINE
# ==================================================

def calculate_age(dob):
    today = date.today()
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


def analyze_frame_metrics(frame_rgb):
    """
    Deterministic frame-level evaluation separating:
    1. Recording Quality Score (illumination, contrast, sharpness)
    2. Pupil Detection & Angle Assessment (circle geometry, offset angle, annotated frame)
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # --- 1. Recording Quality Metrics ---
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Ideal brightness centered around ~115 (range 55-185)
    brightness_score = max(0.0, 100.0 - abs(brightness - 115.0) * 1.1)
    # Ideal contrast >= 40
    contrast_score = min(100.0, (contrast / 45.0) * 100.0)
    # Ideal sharpness >= 120
    sharpness_score = min(100.0, (sharpness / 140.0) * 100.0)

    recording_quality_score = round(
        0.35 * brightness_score + 0.30 * contrast_score + 0.35 * sharpness_score,
        1
    )

    # --- 2. Pupil Detection & Angle Assessment ---
    # Focus on central region of interest to minimize background false positives
    roi_x1, roi_y1 = int(w * 0.15), int(h * 0.15)
    roi_x2, roi_y2 = int(w * 0.85), int(h * 0.85)
    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)
    min_r = max(8, int(min(h, w) * 0.03))
    max_r = max(20, int(min(h, w) * 0.22))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min(h, w) * 0.15)),
        param1=60,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r
    )

    annotated = frame_rgb.copy()
    frame_cx, frame_cy = w / 2.0, h / 2.0

    # Draw subtle alignment crosshair at frame center
    cv2.line(annotated, (int(frame_cx) - 12, int(frame_cy)), (int(frame_cx) + 12, int(frame_cy)), (140, 160, 180), 1)
    cv2.line(annotated, (int(frame_cx), int(frame_cy) - 12), (int(frame_cx), int(frame_cy) + 12), (140, 160, 180), 1)

    pupil_detected = False
    detection_score = 0.0
    pupil_radius_px = 0.0
    pupil_ratio_pct = 0.0
    gaze_angle_deg = 0.0

    if circles is not None:
        candidates = np.round(circles[0, :]).astype("int")
        # Select candidate closest to center with darkest interior intensity (pupil characteristic)
        best_candidate = None
        best_rank = -1e9

        for (cx_roi, cy_roi, r) in candidates:
            cx = cx_roi + roi_x1
            cy = cy_roi + roi_y1

            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(2, int(r * 0.75)), 255, -1)
            ring_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(ring_mask, (cx, cy), int(r * 1.45), 255, -1)
            ring_mask = cv2.subtract(ring_mask, mask)

            inner_intensity = float(cv2.mean(gray, mask=mask)[0])
            outer_intensity = float(cv2.mean(gray, mask=ring_mask)[0])
            darkness_delta = outer_intensity - inner_intensity

            dist_from_center = np.hypot(cx - frame_cx, cy - frame_cy)
            norm_dist = dist_from_center / max(1.0, np.hypot(frame_cx, frame_cy))

            rank = darkness_delta * 1.5 - norm_dist * 45.0
            if rank > best_rank:
                best_rank = rank
                best_candidate = (cx, cy, r, darkness_delta, norm_dist)

        if best_candidate is not None:
            cx, cy, r, darkness_delta, norm_dist = best_candidate
            pupil_detected = True
            pupil_radius_px = float(r)
            pupil_ratio_pct = round((r / max(1.0, min(h, w))) * 100.0, 2)

            # Compute angular deviation vector from optical frame center (degrees)
            dx = cx - frame_cx
            dy = frame_cy - cy  # Invert y so upward gaze is positive
            # Approximate angular displacement assuming ~45 deg horizontal FOV
            fov_deg = 45.0
            radial_px = np.hypot(dx, dy)
            gaze_angle_deg = round((radial_px / max(1.0, w)) * fov_deg, 2)

            # Detection confidence score based on pupil-iris contrast delta & centrality
            contrast_conf = min(100.0, max(20.0, (darkness_delta / 45.0) * 100.0))
            centrality_conf = max(25.0, 100.0 - norm_dist * 80.0)
            detection_score = round(0.65 * contrast_conf + 0.35 * centrality_conf, 1)

            # Annotate detected pupil and alignment vector
            cv2.circle(annotated, (cx, cy), r, (0, 220, 150), 2)
            cv2.circle(annotated, (cx, cy), 3, (255, 90, 90), -1)
            cv2.line(annotated, (int(frame_cx), int(frame_cy)), (cx, cy), (255, 200, 0), 2)
    else:
        # Fallback dark-centroid estimation if Hough circle threshold is missed
        thresh_val = max(25, int(np.percentile(roi_gray, 8)))
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [
            c for c in contours
            if (np.pi * (min_r ** 2)) <= cv2.contourArea(c) <= (np.pi * (max_r ** 2))
        ]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            (x_f, y_f), r_f = cv2.minEnclosingCircle(largest)
            area = cv2.contourArea(largest)
            circularity = area / max(1.0, np.pi * (r_f ** 2))
            if circularity >= 0.55:
                cx, cy, r = int(x_f + roi_x1), int(y_f + roi_y1), int(r_f)
                pupil_detected = True
                pupil_radius_px = float(r)
                pupil_ratio_pct = round((r / max(1.0, min(h, w))) * 100.0, 2)
                dx = cx - frame_cx
                dy = frame_cy - cy
                gaze_angle_deg = round((np.hypot(dx, dy) / max(1.0, w)) * 45.0, 2)
                detection_score = round(min(78.0, circularity * 85.0), 1)

                cv2.circle(annotated, (cx, cy), r, (255, 180, 0), 2)
                cv2.circle(annotated, (cx, cy), 3, (255, 90, 90), -1)
                cv2.line(annotated, (int(frame_cx), int(frame_cy)), (cx, cy), (255, 200, 0), 1)

    return {
        "annotated_frame": annotated,
        "recording_quality_score": recording_quality_score,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "pupil_detected": pupil_detected,
        "detection_score": detection_score,
        "pupil_radius_px": pupil_radius_px,
        "pupil_ratio_pct": pupil_ratio_pct,
        "gaze_angle_deg": gaze_angle_deg
    }


def generate_vision_wellness_intelligence(frame_metrics_list, profile_age=None):
    """
    Aggregates frame-level metrics into Vision Wellness Intelligence:
    - Separated Recording Quality vs. Pupil Detection Scores
    - Pupillary dynamics & angular stability
    - Actionable Vision Wellness Recommendations
    - Tiered Referral Guidance
    """
    if not frame_metrics_list:
        return None

    rec_scores = [m["recording_quality_score"] for m in frame_metrics_list]
    avg_rec_quality = round(float(np.mean(rec_scores)), 1)

    detected_frames = [m for m in frame_metrics_list if m["pupil_detected"]]
    detection_rate_pct = round((len(detected_frames) / len(frame_metrics_list)) * 100.0, 1)

    if detected_frames:
        avg_detection_score = round(
            float(np.mean([m["detection_score"] for m in detected_frames])) * (len(detected_frames) / len(frame_metrics_list)),
            1
        )
        radii_ratios = [m["pupil_ratio_pct"] for m in detected_frames]
        angles = [m["gaze_angle_deg"] for m in detected_frames]

        mean_pupil_ratio = round(float(np.mean(radii_ratios)), 2)
        pupil_variability = round(float(np.std(radii_ratios)), 2)
        mean_angle_deg = round(float(np.mean(angles)), 2)
        angle_drift_deg = round(float(np.max(angles) - np.min(angles)), 2) if len(angles) > 1 else 0.0
    else:
        avg_detection_score = 0.0
        mean_pupil_ratio = 0.0
        pupil_variability = 0.0
        mean_angle_deg = 0.0
        angle_drift_deg = 0.0

    # Composite Vision Wellness Index (weighted toward stable angle & reliable detection)
    angle_stability_score = max(0.0, 100.0 - (mean_angle_deg * 3.5 + angle_drift_deg * 2.5))
    dynamics_balance_score = max(0.0, 100.0 - abs(pupil_variability - 0.8) * 18.0)

    if detected_frames:
        wellness_index = round(
            0.45 * angle_stability_score +
            0.35 * dynamics_balance_score +
            0.20 * avg_detection_score,
            1
        )
    else:
        wellness_index = 0.0

    # Quality Tier Classification
    if avg_rec_quality >= 75:
        quality_tier = "Optimal Capture Quality"
    elif avg_rec_quality >= 50:
        quality_tier = "Moderate Capture Quality"
    else:
        quality_tier = "Low Capture Quality — Recapture Recommended"

    # Detection Reliability Tier
    if avg_detection_score >= 70:
        detection_tier = "High Detection Confidence"
    elif avg_detection_score >= 40:
        detection_tier = "Moderate Detection Confidence"
    else:
        detection_tier = "Low / Unreliable Pupil Lock"

    # Referral Guidance & Wellness Recommendations
    recommendations = []
    if avg_rec_quality < 60:
        recommendations.append(
            "Improve diffuse front-facing illumination and stabilize camera focus at eye level before re-testing."
        )
    if mean_angle_deg > 8.0 or angle_drift_deg > 5.0:
        recommendations.append(
            "Maintain steady central fixation on a single target during capture; elevated angular offset or drift was observed."
        )
    if pupil_variability > 2.2:
        recommendations.append(
            "High inter-frame pupil size fluctuation detected—avoid variable screen glare or shifting ambient light during recording."
        )

    # Standard preventive ocular wellness guidance
    recommendations.append(
        "Follow the 20-20-20 visual recovery rule (every 20 minutes, look 20 feet away for 20 seconds) to reduce accommodative fatigue."
    )
    if profile_age is not None and profile_age >= 40:
        recommendations.append(
            "At age 40+, schedule routine comprehensive dilated eye exams to monitor presbyopia, intraocular pressure, and retinal health."
        )
    else:
        recommendations.append(
            "Maintain consistent ambient lighting during prolonged near-work and track baseline recordings under identical lighting conditions."
        )

    # Referral Triage Logic (strictly gated by Recording Quality & Detection Confidence)
    if avg_rec_quality < 45 or avg_detection_score < 35:
        referral_level = "Recapture Required Before Guidance"
        referral_summary = (
            "Signal reliability is below threshold due to capture lighting, blur, or low pupil lock. "
            "Repeat recording in balanced ambient light before interpreting visual metrics."
        )
        referral_urgency = "low"
    elif mean_angle_deg > 12.0 or angle_drift_deg > 7.5 or pupil_variability > 3.0:
        referral_level = "Optometric / Ophthalmologic Evaluation Recommended"
        referral_summary = (
            "Notable gaze-axis offset, fixation drift, or pupillary variability was detected under adequate capture quality. "
            "Consider scheduling an in-person vision or neuro-ophthalmic screening, especially if accompanied by headaches, diplopia, or visual strain."
        )
        referral_urgency = "high"
    elif mean_angle_deg > 7.5 or angle_drift_deg > 4.5 or wellness_index < 65:
        referral_level = "Routine Optometric Check & Follow-Up Tracking"
        referral_summary = (
            "Mild angular offset or pupillary variability observed. Log a follow-up NEUROPULSE session within 7–14 days "
            "and discuss findings at your next routine eye exam."
        )
        referral_urgency = "moderate"
    else:
        referral_level = "Routine Self-Monitoring (Baseline Optimal)"
        referral_summary = (
            "Pupil lock and gaze-axis alignment fall within expected baseline wellness parameters. "
            "Continue periodic self-monitoring to track longitudinal visual stability."
        )
        referral_urgency = "normal"

    return {
        "recording_quality_score": avg_rec_quality,
        "quality_tier": quality_tier,
        "detection_score": avg_detection_score,
        "detection_tier": detection_tier,
        "detection_rate_pct": detection_rate_pct,
        "wellness_index": wellness_index,
        "mean_pupil_ratio": mean_pupil_ratio,
        "pupil_variability": pupil_variability,
        "mean_angle_deg": mean_angle_deg,
        "angle_drift_deg": angle_drift_deg,
        "recommendations": recommendations,
        "referral_level": referral_level,
        "referral_summary": referral_summary,
        "referral_urgency": referral_urgency
    }


def inspect_video(video_bytes, filename, profile_age=None):
    temporary_path = None
    capture = None

    try:
        suffix = Path(filename).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)

        if not capture.isOpened():
            return None, "Unable to open the uploaded video."

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        frames = []
        annotated_frames = []
        frame_metrics = []

        if frame_count > 0:
            positions = np.linspace(
                0,
                frame_count - 1,
                min(6, frame_count),
                dtype=int
            )

            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                success, frame = capture.read()

                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                    metrics = analyze_frame_metrics(frame_rgb)
                    frame_metrics.append(metrics)
                    annotated_frames.append(metrics["annotated_frame"])

        wellness_report = generate_vision_wellness_intelligence(
            frame_metrics,
            profile_age=profile_age
        )

        result = {
            "filename": filename,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "frames": frames,
            "annotated_frames": annotated_frames,
            "frame_metrics": frame_metrics,
            "wellness_report": wellness_report
        }

        return result, None

    except Exception as error:
        return None, str(error)

    finally:
        if capture is not None:
            capture.release()

        if temporary_path and Path(temporary_path).exists():
            Path(temporary_path).unlink()


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("Overview")

    st.write(
        "A visual assessment, Vision Wellness Intelligence, and "
        "neurological signal research platform."
    )

    st.markdown("### Workspace Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Recorded Sessions",
            len(st.session_state.sessions)
        )

    with col2:
        st.metric(
            "Video Inspection",
            "Completed" if st.session_state.video_inspection else "Pending"
        )

    with col3:
        latest_wellness = (
            f"{st.session_state.video_inspection['wellness_index']}/100"
            if st.session_state.video_inspection and st.session_state.video_inspection.get("wellness_report")
            else "—"
        )
        if st.session_state.video_inspection and st.session_state.video_inspection.get("wellness_report"):
            latest_wellness = f"{st.session_state.video_inspection['wellness_report']['wellness_index']}/100"
        st.metric(
            "Latest Wellness Index",
            latest_wellness
        )

    with col4:
        st.metric(
            "Platform Version",
            "0.2"
        )

    st.divider()

    st.markdown("### Explore NEUROPULSE")

    left, right = st.columns(2)

    with left:
        st.subheader("Pupil Check & Vision Wellness")
        st.write(
            "Upload a video to evaluate recording quality separately from "
            "pupil detection, measure gaze-axis angles, and generate Vision "
            "Wellness Intelligence and referral guidance."
        )

    with right:
        st.subheader("Assessment & Follow-Up Comparison")
        st.write(
            "Record baseline and follow-up assessment sessions, compare longitudinal "
            "changes across visits, and export structured CSV reports."
        )

    st.divider()

    st.markdown("### Current Development")

    st.write(
        "The current prototype supports video inspection, separated recording-quality "
        "and pupil-detection scoring, angular alignment estimation, Vision Wellness "
        "Intelligence, referral guidance, and follow-up session comparison."
    )

    st.caption(
        "Research prototype: outputs support wellness tracking and educational "
        "exploration and do not constitute clinical diagnosis."
    )


# ==================================================
# PUPIL CHECK
# ==================================================

elif page == "Pupil Check":

    st.title("Pupil Check")

    st.write(
        "Video inspection, pupil detection, angle assessment, and Vision Wellness Intelligence."
    )

    st.markdown("### Upload Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.caption(
            f"Selected file: {uploaded_video.name}"
        )

        if st.button("Inspect Video", type="primary"):

            with st.spinner("Running video inspection & Vision Wellness Intelligence..."):

                user_age = (
                    calculate_age(st.session_state.profile_dob)
                    if st.session_state.profile_name
                    else None
                )

                result, error = inspect_video(
                    uploaded_video.getvalue(),
                    uploaded_video.name,
                    profile_age=user_age
                )

                if error:
                    st.error(error)
                else:
                    st.session_state.video_inspection = result
                    st.success("Video inspection and Vision Wellness Intelligence completed.")

    result = st.session_state.video_inspection

    if result is not None:

        st.divider()

        st.markdown("### Video Analysis Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Resolution",
                f"{result['width']} × {result['height']}"
            )

        with col2:
            st.metric(
                "Frame Rate",
                f"{result['fps']:.2f} FPS"
                if result["fps"] > 0
                else "Unavailable"
            )

        with col3:
            st.metric(
                "Duration",
                f"{result['duration']:.2f} sec"
            )

        col4, col5 = st.columns(2)

        with col4:
            st.metric(
                "Total Frames",
                result["frame_count"]
            )

        with col5:
            st.write("**File Name**")
            st.write(result["filename"])

        # --------------------------------------------------
        # VISION WELLNESS INTELLIGENCE & SEPARATED SCORES
        # --------------------------------------------------
        report = result.get("wellness_report")

        if report is not None:

            st.divider()
            st.markdown("### Vision Wellness Intelligence")

            score_col1, score_col2, score_col3, score_col4 = st.columns(4)

            with score_col1:
                st.metric(
                    "Recording Quality Score",
                    f"{report['recording_quality_score']} / 100",
                    help="Evaluates brightness, contrast, and sharpness independently of pupil detection."
                )
                st.caption(report["quality_tier"])

            with score_col2:
                st.metric(
                    "Pupil Detection Score",
                    f"{report['detection_score']} / 100",
                    help="Evaluates pupil lock confidence and frame detection consistency."
                )
                st.caption(f"{report['detection_tier']} ({report['detection_rate_pct']}% frames)")

            with score_col3:
                st.metric(
                    "Mean Gaze / Axis Angle",
                    f"{report['mean_angle_deg']}°",
                    delta=f"Drift: {report['angle_drift_deg']}°",
                    delta_color="off"
                )
                st.caption(f"Mean Pupil Ratio: {report['mean_pupil_ratio']}%")

            with score_col4:
                st.metric(
                    "Vision Wellness Index",
                    f"{report['wellness_index']} / 100"
                )
                st.caption(f"Pupil Variability: ±{report['pupil_variability']}%")

            st.markdown("#### Referral Guidance")

            urgency = report["referral_urgency"]
            if urgency == "high":
                st.error(f"**{report['referral_level']}** — {report['referral_summary']}")
            elif urgency == "moderate":
                st.warning(f"**{report['referral_level']}** — {report['referral_summary']}")
            elif urgency == "low":
                st.info(f"**{report['referral_level']}** — {report['referral_summary']}")
            else:
                st.success(f"**{report['referral_level']}** — {report['referral_summary']}")

            st.markdown("#### Vision Wellness Recommendations")
            for rec in report["recommendations"]:
                st.markdown(f"- {rec}")

            # Quick-save current assessment into Sessions for follow-up comparison
            st.markdown("#### Log Assessment to Follow-Up History")
            with st.form("quick_save_assessment"):
                qs_col1, qs_col2 = st.columns([2, 1])
                with qs_col1:
                    quick_session_name = st.text_input(
                        "Assessment Label",
                        value=f"Pupil Check — {Path(result['filename']).stem}"
                    )
                with qs_col2:
                    quick_session_date = st.date_input(
                        "Assessment Date",
                        value=date.today()
                    )
                save_assessment_btn = st.form_submit_button("Save Assessment to Sessions")

                if save_assessment_btn:
                    if quick_session_name.strip():
                        st.session_state.sessions.append({
                            "Session Name": quick_session_name.strip(),
                            "Date": quick_session_date.strftime("%Y-%m-%d"),
                            "Recording Quality": report["recording_quality_score"],
                            "Detection Score": report["detection_score"],
                            "Wellness Index": report["wellness_index"],
                            "Mean Angle (deg)": report["mean_angle_deg"],
                            "Mean Pupil Ratio (%)": report["mean_pupil_ratio"],
                            "Referral Guidance": report["referral_level"]
                        })
                        st.success("Assessment saved to Sessions for follow-up comparison.")
                    else:
                        st.warning("Enter an assessment label before saving.")

        st.divider()

        st.markdown("### Extracted Frames & Pupil Overlay")

        if result["frames"]:

            view_mode = st.radio(
                "Frame Display Mode",
                ["Pupil & Angle Overlay", "Original Extracted Frames"],
                horizontal=True
            )

            display_frames = (
                result["annotated_frames"]
                if view_mode == "Pupil & Angle Overlay" and result.get("annotated_frames")
                else result["frames"]
            )

            frame_columns = st.columns(3)

            for index, frame in enumerate(display_frames):

                with frame_columns[index % 3]:
                    fm = result["frame_metrics"][index] if result.get("frame_metrics") else None
                    if fm:
                        caption = (
                            f"Frame {index + 1} | Qual: {fm['recording_quality_score']} | "
                            f"Det: {fm['detection_score']} | Angle: {fm['gaze_angle_deg']}°"
                        )
                    else:
                        caption = f"Frame {index + 1}"

                    st.image(
                        frame,
                        caption=caption,
                        use_container_width=True
                    )

        else:

            st.warning("No frames could be extracted.")

    st.divider()

    st.caption(
        "Research prototype: automated pupil estimation and Vision Wellness Intelligence "
        "are provided for exploratory wellness tracking and are not a substitute for clinical diagnosis."
    )


# ==================================================
# SESSIONS
# ==================================================

elif page == "Sessions":

    st.title("Assessment Sessions")

    st.write(
        "Create, manage, and compare baseline and follow-up assessment records."
    )

    st.markdown("### New Session")

    latest_report = (
        st.session_state.video_inspection.get("wellness_report")
        if st.session_state.video_inspection
        else None
    )

    with st.form("session_form"):

        session_name = st.text_input(
            "Session Name",
            placeholder="e.g., Baseline Morning Check or Follow-Up Day 7"
        )

        session_date = st.date_input(
            "Session Date",
            value=date.today()
        )

        attach_latest = st.checkbox(
            "Attach latest Pupil Check metrics (if available)",
            value=latest_report is not None,
            disabled=latest_report is None
        )

        submitted = st.form_submit_button(
            "Save Session"
        )

        if submitted:

            if session_name.strip():

                record = {
                    "Session Name": session_name.strip(),
                    "Date": session_date.strftime("%Y-%m-%d"),
                    "Recording Quality": latest_report["recording_quality_score"] if (attach_latest and latest_report) else None,
                    "Detection Score": latest_report["detection_score"] if (attach_latest and latest_report) else None,
                    "Wellness Index": latest_report["wellness_index"] if (attach_latest and latest_report) else None,
                    "Mean Angle (deg)": latest_report["mean_angle_deg"] if (attach_latest and latest_report) else None,
                    "Mean Pupil Ratio (%)": latest_report["mean_pupil_ratio"] if (attach_latest and latest_report) else None,
                    "Referral Guidance": latest_report["referral_level"] if (attach_latest and latest_report) else "Manual Entry"
                }

                st.session_state.sessions.append(record)

                st.success("Session saved.")

            else:

                st.warning("Enter a session name.")

    st.divider()

    st.markdown("### Session History")

    if st.session_state.sessions:

        sessions_df = pd.DataFrame(
