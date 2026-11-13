
from pathlib import Path
from datetime import date
import tempfile

import streamlit as st
import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NEUROPULSE | BIOPTIQ",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# --------------------------------------------------

if "profile_name" not in st.session_state:
    st.session_state.profile_name = ""

if "profile_dob" not in st.session_state:
    st.session_state.profile_dob = date.today()

if "profile_notes" not in st.session_state:
    st.session_state.profile_notes = ""

if "sessions" not in st.session_state:
    st.session_state.sessions = []

if "video_inspection" not in st.session_state:
    st.session_state.video_inspection = None

if "video_name" not in st.session_state:
    st.session_state.video_name = ""


# --------------------------------------------------
# 3. CUSTOM LIGHT THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F4F8FC;
    color: #243B53;
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #F4F8FC;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF2FA;
    border-right: 1px solid #D8E4F0;
}

section[data-testid="stSidebar"] * {
    color: #173B60 !important;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #173B60 !important;
}

/* Paragraphs */
p, li {
    color: #40566F;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE6F1 !important;
    border-radius: 13px !important;
    padding: 16px !important;
}

/* Metric text */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #173B60 !important;
}

/* General bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE6F1 !important;
    border-radius: 13px !important;
}

/* Forms */
div[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE6F1 !important;
    border-radius: 13px !important;
    padding: 18px !important;
}

/* Text inputs */
.stTextInput input,
.stDateInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #173B60 !important;
    border: 1px solid #C9D8E8 !important;
    border-radius: 8px !important;
}

/* Input labels */
.stTextInput label,
.stDateInput label,
.stTextArea label,
.stNumberInput label {
    color: #344B67 !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #A9C1D9 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background-color: #1769AA !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
}

/* Button hover */
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #12558C !important;
    color: #FFFFFF !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE6F1 !important;
    border-radius: 10px !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #173B60 !important;
    border-color: #C9D8E8 !important;
}

/* Captions */
.stCaption {
    color: #71849A !important;
}

/* Branding header */
.brand-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid #DCE6F1;
    border-radius: 14px;
    padding: 17px 23px;
    margin-bottom: 28px;
}

.product-name {
    color: #1769AA;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.product-tagline {
    color: #61758C;
    font-size: 10px;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

.parent-brand {
    text-align: right;
}

.parent-name {
    color: #173B60;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 2px;
}

.parent-caption {
    color: #61758C;
    font-size: 10px;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #8193A8;
    font-size: 12px;
    padding: 25px 0 10px 0;
}

/* Mobile branding */
@media (max-width: 600px) {
    .brand-header {
        padding: 13px;
    }

    .product-name {
        font-size: 19px;
    }

    .product-tagline {
        font-size: 8px;
    }

    .parent-name {
        font-size: 14px;
    }

    .parent-caption {
        font-size: 8px;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. BRAND HEADER
# --------------------------------------------------

st.markdown("""
<div class="brand-header">

    <div class="product-brand">
        <div class="product-name">NEUROPULSE</div>
        <div class="product-tagline">
            SEE THE SIGNAL. UNLOCK BETTER CARE.
        </div>
    </div>

    <div class="parent-brand">
        <div class="parent-name">BIOPTIQ</div>
        <div class="parent-caption">
            THE FUTURE OF VISION.
        </div>
    </div>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

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
        suffix = Path(filename).suffix

        if not suffix:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temporary_file:

            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)

        if not capture.isOpened():
            return None, "Unable to open this video file."

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

        if fps and fps > 0:
            duration = frame_count / fps
        else:
            duration = 0

        frames = []

        if frame_count > 0:

            sample_positions = np.linspace(
                0,
                max(frame_count - 1, 0),
                min(6, frame_count),
                dtype=int
            )

            for position in sample_positions:

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


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("## NEUROPULSE")

st.sidebar.caption("Clinical Visual Assessment Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Pupil Check",
        "Sessions",
        "My Profile",
        "Research",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption("BIOPTIQ | The Future of Vision.")


# --------------------------------------------------
# 7. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.title("Welcome to NEUROPULSE")

    st.write(
        "A research-oriented platform designed to explore "
        "visual assessment and neurological signal analysis."
    )

    st.divider()

    st.subheader("Platform Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Platform",
            "NEUROPULSE"
        )

    with col2:
        st.metric(
            "Current Sessions",
            len(st.session_state.sessions)
        )

    with col3:
        st.metric(
            "Video Inspections",
            1 if st.session_state.video_inspection else 0
        )

    st.divider()

    st.subheader("Explore the Platform")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("Pupil Check")

            st.write(
                "Upload a video and inspect its basic technical "
                "properties and selected frames."
            )

            if st.button(
                "Open Pupil Check",
                key="overview_pupil"
            ):
                st.session_state.selected_page = "Pupil Check"
                st.rerun()

    with col2:

        with st.container(border=True):

            st.subheader("Sessions")

            st.write(
                "Create and review your assessment session records."
            )

            if st.button(
                "Open Sessions",
                key="overview_sessions"
            ):
                st.session_state.selected_page = "Sessions"
                st.rerun()

    st.info(
        "NEUROPULSE is currently a research and development "
        "platform. Video inspection does not yet perform "
        "validated pupil detection or clinical diagnosis."
    )


# --------------------------------------------------
# 8. PUPIL CHECK PAGE
# --------------------------------------------------

elif page == "Pupil Check":

    st.title("Pupil Check")

    st.write(
        "Upload a video for basic video inspection and "
        "sample-frame visualization."
    )

    st.info(
        "Current functionality: video metadata inspection "
        "and frame extraction. Automated pupil detection "
        "and measurement are not yet implemented."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="pupil_video"
    )

    if uploaded_video is not None:

        st.write(
            f"**Selected file:** {uploaded_video.name}"
        )

        if st.button(
            "Inspect Video",
            type="primary",
            key="inspect_video"
        ):

            with st.spinner("Inspecting video..."):

                video_bytes = uploaded_video.getvalue()

                result, error = inspect_video(
                    video_bytes,
                    uploaded_video.name
                )

                if error:
                    st.error(error)

                else:
                    st.session_state.video_inspection = result
                    st.session_state.video_name = uploaded_video.name

                    st.success(
                        "Video inspection completed."
                    )

    result = st.session_state.video_inspection

    if result is not None:

        st.divider()

        st.subheader("Video Information")

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
            st.metric(
                "File",
                result["filename"]
            )

        st.divider()

        st.subheader("Extracted Sample Frames")

        if result["frames"]:

            frame_columns = st.columns(3)

            for index, frame in enumerate(result["frames"]):

                with frame_columns[index % 3]:

                    st.image(
                        frame,
                        caption=f"Sample Frame {index + 1}",
                        use_container_width=True
                    )

        else:

            st.warning(
                "No frames could be extracted from this video."
            )


# --------------------------------------------------
# 9. SESSIONS PAGE
# --------------------------------------------------

elif page == "Sessions":

    st.title("Assessment Sessions")

    st.write(
        "Create and manage your session records."
    )

    with st.form("session_form"):

        session_name = st.text_input(
            "Session Name",
            placeholder="Enter a session name"
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

                st.success(
                    "Session saved successfully."
                )

            else:

                st.warning(
                    "Please enter a session name."
                )

    st.divider()

    st.subheader("Session History")

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
            label="Download Session Records",
            data=csv_data,
            file_name="neuropulse_sessions.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No sessions have been recorded yet."
        )


# --------------------------------------------------
# 10. MY PROFILE PAGE
# --------------------------------------------------

elif page == "My Profile":

    st.title("My Profile")

    st.write(
        "Manage your personal profile information."
    )

    with st.form("profile_form"):

        profile_name = st.text_input(
            "Full Name",
            value=st.session_state.profile_name,
            placeholder="Enter your name"
        )

        profile_dob = st.date_input(
            "Date of Birth",
            value=st.session_state.profile_dob,
            max_value=date.today()
        )

        profile_notes = st.text_area(
            "Additional Notes",
            value=st.session_state.profile_notes,
            placeholder="Enter any additional notes..."
        )

        profile_submitted = st.form_submit_button(
            "Save Profile"
        )

        if profile_submitted:

            st.session_state.profile_name = profile_name
            st.session_state.profile_dob = profile_dob
            st.session_state.profile_notes = profile_notes

            st.success(
                "Profile updated successfully."
            )

    st.divider()

    st.subheader("Profile Summary")

    if st.session_state.profile_name:

        st.write(
            f"**Name:** {st.session_state.profile_name}"
        )

        st.write(
            f"**Date of Birth:** "
            f"{st.session_state.profile_dob.strftime('%d %B %Y')}"
        )

        st.write(
            f"**Age:** {calculate_age(st.session_state.profile_dob)} years"
        )

        if st.session_state.profile_notes:

            st.write("**Additional Notes:**")

            st.write(
                st.session_state.profile_notes
            )

    else:

        st.info(
            "Your profile has not been completed yet."
        )


# --------------------------------------------------
# 11. RESEARCH PAGE
# --------------------------------------------------

elif page == "Research":

    st.title("Research and Development")

    st.write(
        "NEUROPULSE is being developed as a research-oriented "
        "platform exploring visual signals and their potential "
        "role in neurological assessment."
    )

    st.divider()

    st.subheader("Research Directions")

    with st.container(border=True):

        st.markdown("### 1. Pupil Assessment")

        st.write(
            "Exploring image and video-based pupil assessment "
            "methods and their potential applications."
        )

    with st.container(border=True):

        st.markdown("### 2. Visual Signal Analysis")

        st.write(
            "Investigating measurable visual signals and "
            "their possible relationship with neurological function."
        )

    with st.container(border=True):

        st.markdown("### 3. Clinical Research Integration")

        st.write(
            "Exploring how computational methods could support "
            "future clinical research and assessment workflows."
        )

    st.divider()

    st.subheader("Development Status")

    st.write(
        "The current version supports video metadata inspection, "
        "sample-frame extraction, session recording and profile "
        "management. Research validation and further development "
        "are required before clinical use."
    )


# --------------------------------------------------
# 12. ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.title("About NEUROPULSE")

    st.subheader("See the Signal. Unlock Better Care.")

    st.write(
        "NEUROPULSE is a research-oriented healthcare technology "
        "platform being developed to explore visual assessment "
        "and neurological signal analysis."
    )

    st.divider()

    st.subheader("Developed Under")

    st.markdown(
        """
        **BIOPTIQ**

        The Future of Vision.
        """
    )

    st.divider()

    st.subheader("Current Capabilities")

    st.markdown(
        """
        - Video upload and inspection
        - Video metadata extraction
        - Sample-frame visualization
        - Assessment session records
        - User profile management
        """
    )

    st.divider()

    st.subheader("Important Disclaimer")

    st.warning(
        "NEUROPULSE is an early-stage research and development "
        "platform. It is not a medical device and does not provide "
        "medical diagnosis, treatment recommendations or validated "
        "clinical measurements. Do not use it to make clinical "
        "decisions."
    )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="app-footer">
    NEUROPULSE · A BIOPTIQ PRODUCT
    <br>
    The Future of Vision.
</div>
""", unsafe_allow_html=True)
