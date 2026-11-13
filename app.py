
from pathlib import Path
from datetime import date
import tempfile

import streamlit as st
import cv2
import numpy as np
import pandas as pd


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NEUROPULSE | BIOPTIQ",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

BRAND_IMAGE = BASE_DIR / "Bioptiq and Neuropulse Identity.webp"


# =====================================================
# 3. SESSION STATE INITIALIZATION
# =====================================================

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
    st.session_state.video_name = None


# =====================================================
# 4. CUSTOM DESIGN
# =====================================================

st.markdown(
    """
    <style>

    /* Main application background */

    .stApp {
        background-color: #F3F7FB;
        color: #18344F;
    }

    [data-testid="stHeader"] {
        background-color: #F3F7FB;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #DCE8F2;
    }

    [data-testid="stSidebar"] * {
        color: #18344F !important;
    }

    /* Headings */

    h1, h2, h3, h4 {
        color: #173B5B !important;
    }

    /* General text */

    p, label, li {
        color: #304F6C;
    }

    [data-testid="stCaptionContainer"] p {
        color: #496984;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #C8D8E7;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #173B5B !important;
    }

    /* Text inputs */

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #18344F !important;
        border: 1px solid #AFC4D8 !important;
    }

    /* Date input */

    [data-testid="stDateInput"] input {
        background-color: #FFFFFF !important;
        color: #18344F !important;
    }

    /* Buttons */

    .stButton button,
    .stDownloadButton button {
        background-color: #1B659B;
        color: #FFFFFF !important;
        border: none;
        border-radius: 9px;
        min-height: 42px;
    }

    .stButton button p,
    .stDownloadButton button p {
        color: #FFFFFF !important;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background-color: #144B75;
        border: none;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 1px solid #C8D8E7;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] * {
        color: #18344F !important;
    }

    /* Alerts */

    [data-testid="stAlert"] {
        background-color: #FFFFFF;
        color: #18344F;
        border-radius: 10px;
    }

    /* Dividers */

    hr {
        border-color: #C8D8E7;
    }

    /* Form containers */

    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #D5E1EC;
        border-radius: 14px;
        padding: 20px;
    }

    /* Data tables */

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def calculate_age(dob):

    if dob is None:
        return None

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
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)
            temporary_path = temp_file.name

        capture = cv2.VideoCapture(temporary_path)

        if not capture.isOpened():
            raise ValueError("Unable to open this video recording.")

        fps = capture.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )

        sample_frames = []

        if total_frames > 0:

            positions = np.linspace(
                0,
                total_frames - 1,
                num=min(6, total_frames),
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

                    sample_frames.append(
                        {
                            "position": int(position),
                            "image": frame
                        }
                    )

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "sample_frames": sample_frames
        }

    finally:

        if capture is not None:
            capture.release()

        if temporary_path is not None:

            Path(temporary_path).unlink(
                missing_ok=True
            )


# =====================================================
# 6. BRANDING
# =====================================================

if BRAND_IMAGE.exists():

    st.image(
        str(BRAND_IMAGE),
        width=360
    )

else:

    st.warning(
        "Original BIOPTIQ and NEUROPULSE brand image "
        "could not be found in the repository root."
    )


st.title("NEUROPULSE")

st.markdown(
    """
    **Pupil Response Dashboard**

    Explore pupil-response recordings, organize sessions
    and build a personalized physiological research profile.
    """
)

profile_age = calculate_age(
    st.session_state.profile_dob
)

if st.session_state.profile_name:

    if profile_age is not None:

        st.caption(
            f"Welcome, {st.session_state.profile_name} "
            f"· Age: {profile_age} years"
        )

    else:

        st.caption(
            f"Welcome, {st.session_state.profile_name}"
        )

else:

    st.caption("Profile information has not been added yet.")


st.divider()


# =====================================================
# 7. NAVIGATION
# =====================================================

st.sidebar.title("NEUROPULSE")

st.sidebar.caption(
    "BIOPTIQ | The Future of Vision."
)

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

st.sidebar.caption(
    "See the Signal. Unlock Better Care."
)


# =====================================================
# 8. OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Welcome to NEUROPULSE")

    st.write(
        "A dedicated digital workspace for exploring "
        "pupil-response recordings and organizing "
        "physiological research data."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Sessions",
            len(st.session_state.sessions)
        )

    with col2:

        profile_complete = bool(
            st.session_state.profile_name.strip()
        )

        st.metric(
            "Profile Status",
            "Complete" if profile_complete else "Pending"
        )

    with col3:

        st.metric(
            "Workspace",
            "Active"
        )

    st.divider()

    st.subheader("Explore Your Workspace")

    with st.container(border=True):

        st.subheader("Pupil Check")

        st.write(
            "Upload and review pupil-response video recordings "
            "and inspect their technical properties."
        )

    with st.container(border=True):

        st.subheader("Sessions")

        st.write(
            "Create, organize and export recording sessions."
        )

    with st.container(border=True):

        st.subheader("My Profile")

        st.write(
            "Manage your name, date of birth, age "
            "and additional research notes."
        )

    with st.container(border=True):

        st.subheader("Research")

        st.write(
            "Explore the scientific direction and "
            "development of NEUROPULSE."
        )


# =====================================================
# 9. PUPIL CHECK
# =====================================================

elif page == "Pupil Check":

    st.header("Pupil Response Recording")

    st.write(
        "Upload a recording to review its visual content "
        "and technical properties."
    )

    st.info(
        "Supported video formats: MP4, MOV, AVI and MKV."
    )

    uploaded_video = st.file_uploader(
        "Upload Video Recording",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        if st.session_state.video_name != uploaded_video.name:

            st.session_state.video_name = uploaded_video.name
            st.session_state.video_inspection = None

        st.subheader("Recording Preview")

        st.video(video_bytes)

        if st.button(
            "Inspect Recording",
            type="primary"
        ):

            with st.spinner(
                "Inspecting recording and extracting sample frames..."
            ):

                try:

                    result = inspect_video(
                        video_bytes,
                        uploaded_video.name
                    )

                    st.session_state.video_inspection = result

                    st.success(
                        "Recording information retrieved successfully."
                    )

                except Exception as error:

                    st.session_state.video_inspection = None

                    st.error(
                        f"Recording inspection failed: {error}"
                    )

        result = st.session_state.video_inspection

        if result is not None:

            st.divider()

            st.subheader("Recording Information")

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
                )

            with col3:

                st.metric(
                    "Duration",
                    f"{result['duration']:.2f} sec"
                )

            st.metric(
                "Total Frames",
                result["total_frames"]
            )

            st.divider()

            st.subheader("Extracted Sample Frames")

            frames = result["sample_frames"]

            if frames:

                columns = st.columns(3)

                for index, frame_data in enumerate(frames):

                    with columns[index % 3]:

                        st.image(
                            frame_data["image"],
                            caption=(
                                f"Frame {frame_data['position']}"
                            ),
                            use_container_width=True
                        )

            else:

                st.info(
                    "No readable sample frames were extracted."
                )

    else:

        st.info(
            "Upload a video recording to begin."
        )

        st.session_state.video_inspection = None
        st.session_state.video_name = None


# =====================================================
# 10. SESSIONS
# =====================================================

elif page == "Sessions":

    st.header("Recording Sessions")

    st.write(
        "Organize and review your recording sessions."
    )

    with st.form("session_form"):

        session_name = st.text_input(
            "Session Name",
            placeholder="Enter session name"
        )

        submitted = st.form_submit_button(
            "Create Session"
        )

        if submitted:

            if session_name.strip():

                st.session_state.sessions.append(
                    {
                        "Session": session_name.strip(),
                        "Date": date.today().isoformat()
                    }
                )

                st.success(
                    "Session created successfully."
                )

            else:

                st.warning(
                    "Please enter a session name."
                )

    st.divider()

    if st.session_state.sessions:

        st.subheader("Your Sessions")

        session_df = pd.DataFrame(
            st.session_state.sessions
        )

        st.dataframe(
            session_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Sessions",
            data=session_df.to_csv(index=False),
            file_name="neuropulse_sessions.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No recording sessions created yet."
        )


# =====================================================
# 11. MY PROFILE
# =====================================================

elif page == "My Profile":

    st.header("My Profile")

    st.write(
        "Manage your personal profile and research notes."
    )

    with st.form("profile_form"):

        name = st.text_input(
            "Full Name",
            value=st.session_state.profile_name,
            placeholder="Enter your full name"
        )

        dob = st.date_input(
            "Date of Birth",
            value=st.session_state.profile_dob,
            min_value=date(1900, 1, 1),
            max_value=date.today()
        )

        notes = st.text_area(
            "Additional Notes",
            value=st.session_state.profile_notes,
            placeholder=(
                "Add your personal or research notes..."
            ),
            height=150
        )

        submitted = st.form_submit_button(
            "Save Profile",
            type="primary"
        )

        if submitted:

            st.session_state.profile_name = name.strip()
            st.session_state.profile_dob = dob
            st.session_state.profile_notes = notes

            st.success(
                "Profile saved successfully."
            )

    st.divider()

    age = calculate_age(
        st.session_state.profile_dob
    )

    st.subheader("Profile Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Name",
            st.session_state.profile_name or "Not added"
        )

    with col2:

        st.metric(
            "Age",
            f"{age} years" if age is not None else "Not added"
        )

    st.subheader("Additional Notes")

    if st.session_state.profile_notes.strip():

        st.write(
            st.session_state.profile_notes
        )

    else:

        st.caption(
            "No additional notes added."
        )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:12px;">
            <h3 style="color:#173B5B;">
                NEUROPULSE | BIOPTIQ
            </h3>
            <p style="color:#496984;">
                See the Signal. Unlock Better Care.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 12. RESEARCH
# =====================================================

elif page == "Research":

    st.header("Research and Development")

    st.write(
        "NEUROPULSE explores computational approaches "
        "to pupil-response recording and physiological research."
    )

    st.subheader("Research Directions")

    with st.container(border=True):

        st.subheader("Pupil Detection and Tracking")

        st.write(
            "Exploring image-processing approaches to identify "
            "and track pupil boundaries across video frames."
        )

    with st.container(border=True):

        st.subheader("Quantitative Pupil Analysis")

        st.write(
            "Investigating pupil diameter, constriction amplitude, "
            "latency and redilation characteristics."
        )

    with st.container(border=True):

        st.subheader("Longitudinal Research Profiles")

        st.write(
            "Organizing repeated recordings to explore "
            "changes in pupil-response characteristics over time."
        )

    st.subheader("Development Focus")

    st.write(
        "Building a structured platform for video acquisition, "
        "visualization, computational analysis and future validation."
    )


# =====================================================
# 13. ABOUT
# =====================================================

elif page == "About":

    st.header("About NEUROPULSE")

    st.subheader("A BIOPTIQ Product")

    st.write(
        "NEUROPULSE is the first product under BIOPTIQ."
    )

    st.markdown(
        "**BIOPTIQ — The Future of Vision.**"
    )

    st.markdown(
        "**NEUROPULSE — See the Signal. Unlock Better Care.**"
    )

    st.divider()

    st.subheader("Our Purpose")

    st.write(
        "To explore digital and computational approaches "
        "to pupil-response recordings and physiological research."
    )

    st.subheader("Current Platform Capabilities")

    st.write(
        """
        - Video upload and preview.
        - Recording metadata inspection.
        - Sample frame extraction.
        - Recording session organization.
        - Personal research profile management.
        """
    )

    st.subheader("Research Disclaimer")

    st.info(
        "NEUROPULSE is an experimental research platform. "
        "It is not a validated medical device and is not intended "
        "for clinical diagnosis or treatment decisions."
    )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    **BIOPTIQ** | The Future of Vision.
    """
)

st.caption(
    "NEUROPULSE | See the Signal. Unlock Better Care."
)

st.caption(
    "Experimental research platform."
)
