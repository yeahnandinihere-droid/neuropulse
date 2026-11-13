
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
    "prototype exploring visual assessment and "
    "neurological signal analysis."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### DEVELOPMENT")

st.sidebar.caption("Research and Development")

st.sidebar.caption("Version 0.1")

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
# HELPER FUNCTIONS
# ==================================================

def calculate_age(dob):

    today = date.today()

    age = today.year - dob.year

    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1

    return age


def inspect_video(video_bytes, filename):

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

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = capture.get(cv2.CAP_PROP_FPS)

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = frame_count / fps if fps > 0 else 0

        frames = []

        if frame_count > 0:

            positions = np.linspace(
                0,
                frame_count - 1,
                min(6, frame_count),
                dtype=int
            )

            for position in positions:

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(position)
                )

                success, frame = capture.read()

                if success:

                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frames.append(frame)

        result = {
            "filename": filename,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "frames": frames
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
        "A visual assessment and neurological signal "
        "research platform."
    )

    st.markdown("### Workspace Overview")

    col1, col2, col3 = st.columns(3)

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
        st.metric(
            "Platform Version",
            "0.1"
        )

    st.divider()

    st.markdown("### Explore NEUROPULSE")

    left, right = st.columns(2)

    with left:

        st.subheader("Pupil Check")

        st.write(
            "Upload a video, inspect its technical properties "
            "and explore extracted frames."
        )

    with right:

        st.subheader("Assessment Sessions")

        st.write(
            "Create and review your assessment session records."
        )

    st.divider()

    st.markdown("### Current Development")

    st.write(
        "The current prototype supports video inspection, "
        "frame extraction, session recording and profile "
        "management."
    )

    st.caption(
        "Automated pupil detection and validated clinical "
        "measurements are not yet available."
    )


# ==================================================
# PUPIL CHECK
# ==================================================

elif page == "Pupil Check":

    st.title("Pupil Check")

    st.write(
        "Video inspection and visual frame exploration."
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

            with st.spinner("Processing video..."):

                result, error = inspect_video(
                    uploaded_video.getvalue(),
                    uploaded_video.name
                )

                if error:

                    st.error(error)

                else:

                    st.session_state.video_inspection = result

                    st.success("Video inspection completed.")

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

        st.divider()

        st.markdown("### Extracted Frames")

        if result["frames"]:

            frame_columns = st.columns(3)

            for index, frame in enumerate(result["frames"]):

                with frame_columns[index % 3]:

                    st.image(
                        frame,
                        caption=f"Frame {index + 1}",
                        use_container_width=True
                    )

        else:

            st.warning("No frames could be extracted.")

    st.divider()

    st.caption(
        "Research prototype: automated pupil detection, "
        "pupil diameter calculation and clinical interpretation "
        "are not currently implemented."
    )


# ==================================================
# SESSIONS
# ==================================================

elif page == "Sessions":

    st.title("Assessment Sessions")

    st.write(
        "Create and manage your assessment records."
    )

    st.markdown("### New Session")

    with st.form("session_form"):

        session_name = st.text_input(
            "Session Name",
            placeholder="Enter session name"
        )

        session_date = st.date_input(
            "Session Date",
            value=date.today()
        )

        submitted = st.form_submit_button(
            "Save Session"
        )

        if submitted:

            if session_name.strip():

                st.session_state.sessions.append({
                    "Session Name": session_name.strip(),
                    "Date": session_date.strftime("%Y-%m-%d")
                })

                st.success("Session saved.")

            else:

                st.warning("Enter a session name.")

    st.divider()

    st.markdown("### Session History")

    if st.session_state.sessions:

        sessions_df = pd.DataFrame(
            st.session_state.sessions
        )

        st.dataframe(
            sessions_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = sessions_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Export Session Records",
            data=csv_data,
            file_name="neuropulse_sessions.csv",
            mime="text/csv"
        )

    else:

        st.caption("No assessment sessions recorded yet.")


# ==================================================
# MY PROFILE
# ==================================================

elif page == "My Profile":

    st.title("My Profile")

    st.write(
        "Manage your profile information."
    )

    st.markdown("### Profile Details")

    with st.form("profile_form"):

        profile_name = st.text_input(
            "Full Name",
            value=st.session_state.profile_name
        )

        profile_dob = st.date_input(
            "Date of Birth",
            value=st.session_state.profile_dob,
            max_value=date.today()
        )

        profile_notes = st.text_area(
            "Additional Notes",
            value=st.session_state.profile_notes,
            placeholder="Enter additional notes..."
        )

        submitted = st.form_submit_button(
            "Save Profile"
        )

        if submitted:

            st.session_state.profile_name = profile_name
            st.session_state.profile_dob = profile_dob
            st.session_state.profile_notes = profile_notes

            st.success("Profile updated.")

    st.divider()

    st.markdown("### Profile Summary")

    if st.session_state.profile_name:

        st.write(
            f"**Name:** {st.session_state.profile_name}"
        )

        st.write(
            f"**Date of Birth:** "
            f"{st.session_state.profile_dob.strftime('%d %B %Y')}"
        )

        st.write(
            f"**Age:** "
            f"{calculate_age(st.session_state.profile_dob)} years"
        )

        if st.session_state.profile_notes:

            st.write("**Additional Notes:**")

            st.write(st.session_state.profile_notes)

    else:

        st.caption("Complete your profile to see your summary.")


# ==================================================
# RESEARCH
# ==================================================

elif page == "Research":

    st.title("Research and Development")

    st.write(
        "Exploring visual signals and their potential "
        "applications in neurological assessment."
    )

    st.divider()

    st.markdown("### Research Directions")

    st.subheader("01 — Pupil Assessment")

    st.write(
        "Exploration of video-based pupil assessment "
        "and image-processing methods."
    )

    st.divider()

    st.subheader("02 — Visual Signal Analysis")

    st.write(
        "Investigation of measurable visual signals "
        "and their possible relationship with neurological function."
    )

    st.divider()

    st.subheader("03 — Clinical Research Integration")

    st.write(
        "Exploring computational approaches that could "
        "support clinical research workflows."
    )

    st.divider()

    st.markdown("### Development Status")

    st.write(
        "The current version supports video inspection, "
        "sample-frame extraction, session records and "
        "profile management."
    )

    st.caption(
        "Further research and validation are required "
        "before clinical application."
    )


# ==================================================
# ABOUT
# ==================================================

elif page == "About":

    st.title("About NEUROPULSE")

    st.subheader("See the Signal. Unlock Better Care.")

    st.write(
        "NEUROPULSE is a research-oriented healthcare "
        "technology platform exploring visual assessment "
        "and neurological signal analysis."
    )

    st.divider()

    st.markdown("### Parent Company")

    st.subheader("BIOPTIQ")

    st.write("The Future of Vision.")

    st.divider()

    st.markdown("### Current Capabilities")

    st.write("Video upload and inspection")
    st.write("Video metadata extraction")
    st.write("Sample-frame visualization")
    st.write("Assessment session records")
    st.write("Profile management")

    st.divider()

    st.markdown("### Research Disclaimer")

    st.write(
        "NEUROPULSE is an early-stage research and development "
        "prototype. It is not a medical device and does not "
        "provide medical diagnosis, treatment recommendations "
        "or validated clinical measurements. It should not "
        "be used to make clinical decisions."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "NEUROPULSE  |  A BIOPTIQ PRODUCT  |  The Future of Vision."
)
