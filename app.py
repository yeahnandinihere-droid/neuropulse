from pathlib import Path
from datetime import date 
import tempfile
import streamlit as st 
import cv2
import numpy as np
import pandas as pd 
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NEUROPULSE | BIOPTIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================
# 2. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

BRAND_IMAGE = (
    BASE_DIR
    / "Bioptiq and Neuropulse Identity.webp"
)

# ==========================================
# 3. SESSION STATE
# ==========================================

if "profile_name" not in st.session_state:
    st.session_state.profile_name = ""

if "profile_dob" not in st.session_state:
    st.session_state.profile_dob = None

if "sessions" not in st.session_state:
    st.session_state.sessions = []
  # ==========================================
# 4. CUSTOM DESIGN
# ==========================================
st.markdown(""""
<style>
stApp {
    background-color: #E7EFF7;
    color: #18344F;
}

[data-testid="stSidebar"] {
    background-color: #CADBEC;
}
h1, h2, h3 {
    color: #173B5B;
}

p, label, span {
    color: #304F6C;
}

.dashboard-banner {
    background: linear-gradient(
        120deg,
        #173B5B,
        #246B96
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.dashboard-banner h1 {
    color: white;
    font-size: 34px;
}

.dashboard-banner p {
    color: #E7F2FA;
    font-size: 16px;
}

.brand-caption {
    color: #1B659B;
    font-weight: 600;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: #D1E1EF;
    border: 1px solid #BDD3E5;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #173B5B;
}

div[data-testid="stMetricLabel"] {
    color: #365A78;
}

.stButton > button {
    background-color: #1B659B;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 8px 20px;
}

.stButton > button:hover {
    background-color: #154C75;
    color: white;
}

.stDownloadButton > button {
    background-color: #1B659B;
    color: white;
}

div[data-testid="stFileUploader"] {
    background-color: #D7E6F2;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. AGE CALCULATION
# ==========================================

def calculate_age(dob):

    if dob is None:
        return None

    today = date.today()

    age = today.year - dob.year

    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1

    return age


# ==========================================
# 6. BRANDING
# ==========================================

if BRAND_IMAGE.exists():

    st.image(
        str(BRAND_IMAGE),
        width=340
    )

else:

    st.warning(
        "Brand identity image is not available."
    )


st.markdown("""
<div class="dashboard-banner">

<h1>Pupil Response Dashboard</h1>

<p>
Explore pupil-response recordings, organize sessions
and build a personalized physiological research profile.
</p>

</div>
""", unsafe_allow_html=True)

age = calculate_age(st.session_state.profile_dob)

if age is not None:
    st.caption(f"Profile age: {age} years")
else:
    st.caption("Profile age: Not added")


# ==========================================
# 7. NAVIGATION
# ==========================================

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


# ==========================================
# 8. OVERVIEW
# ==========================================

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
        st.metric(
            "Profile Status",
            "Complete"
            if st.session_state.profile_dob
            else "Pending"
        )

    with col3:
        st.metric(
            "Workspace",
            "Active"
        )

    st.subheader("Explore Your Workspace")

    st.markdown("""
    **Pupil Check**

    Upload and review pupil-response video recordings.

    **Sessions**

    Organize and export recording sessions.

    **My Profile**

    Manage profile information and personal research records.

    **Research**

    Explore the scientific direction and development of NEUROPULSE.
    """)


# ==========================================
# 9. VIDEO INSPECTION
# ==========================================

elif page == "Pupil Check":

    st.header("Pupil Response Recording")

    st.write(
        "Upload a recording to review its visual content "
        "and technical properties."
    )

    uploaded_video = st.file_uploader(
        "Upload Video Recording",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        st.video(video_bytes)

        if st.button(
            "Inspect Recording",
            type="primary"
        ):

            temporary_path = None
            capture = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False
                ) as temp_file:

                    temp_file.write(video_bytes)
                    temporary_path = temp_file.name

                capture = cv2.VideoCapture(
                    temporary_path
                )

                if not capture.isOpened():

                    st.error(
                        "Unable to read this video."
                    )

                else:

                    fps = capture.get(
                        cv2.CAP_PROP_FPS
                    )

                    total_frames = int(
                        capture.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        )
                    )

                    width = int(
                        capture.get(
                            cv2.CAP_PROP_FRAME_WIDTH
                        )
                    )

                    height = int(
                        capture.get(
                            cv2.CAP_PROP_FRAME_HEIGHT
                        )
                    )

                    duration = (
                        total_frames / fps
                        if fps > 0
                        else 0
                    )

                    st.success(
                        "Recording information retrieved."
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Resolution",
                        f"{width} × {height}"
                    )

                    col2.metric(
                        "Frame Rate",
                        f"{fps:.2f} FPS"
                    )

                    col3.metric(
                        "Duration",
                        f"{duration:.2f} sec"
                    )

                    st.metric(
                        "Total Frames",
                        total_frames
                    )

                    if st.button(
                        "Extract Sample Frames"
                    ):

                        positions = np.linspace(
                            0,
                            max(total_frames - 1, 0),
                            min(6, total_frames),
                            dtype=int
                        )

                        frames = []

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

                        st.subheader(
                            "Recording Preview"
                        )

                        for index, frame in enumerate(frames):

                            st.image(
                                frame,
                                caption=f"Frame {index + 1}",
                                width=350
                            )

            except Exception as error:

                st.error(
                    f"Recording inspection failed: {error}"
                )

            finally:

                if capture is not None:
                    capture.release()

                if temporary_path:
                    Path(temporary_path).unlink(
                        missing_ok=True
                    )

    else:

        st.info(
            "Upload a video recording to begin."
        )


# ==========================================
# 10. SESSIONS
# ==========================================

elif page == "Sessions":

    st.header("Recording Sessions")

    st.write(
        "Organize and review your recording sessions."
    )

    session_name = st.text_input(
        "Session Name",
        placeholder="Enter session name"
    )

    if st.button("Create Session"):

        if session_name.strip():

            st.session_state.sessions.append({
                "Session": session_name,
                "Date": date.today().isoformat()
            })

            st.success(
                "Session created successfully."
            )

            st.rerun()

        else:

            st.warning(
                "Please enter a session name."
            )

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
            data=session_df.to_csv(
                index=False
            ),
            file_name="neuropulse_sessions.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No recording sessions created yet."
        )


# ==========================================
# 11. MY PROFILE
# ==========================================

elif page == "My Profile":

    st.header("My Profile")

    with st.form("profile_form"):

        name = st.text_input(
            "Full Name",
            value=st.session_state.profile_name
        )

        dob = st.date_input(
            "Date of Birth",
            value=st.session_state.profile_dob,
            min_value=date(1900, 1, 1),
            max_value=date.today()
        )

        submitted = st.form_submit_button(
            "Save Profile"
        )

        if submitted:

            st.session_state.profile_name = name
            st.session_state.profile_dob = dob

            st.success(
                "Profile saved successfully."
            )

    st.divider()

    age = calculate_age(
        st.session_state.profile_dob
    )

    st.subheader("Profile Information")

    st.write(
        f"**Name:** "
        f"{st.session_state.profile_name or 'Not added'}"
    )

    st.write(
        f"**Age:** "
        f"{str(age) + ' years' if age is not None else 'Not added'}"
    )


# ==========================================
# 12. RESEARCH
# ==========================================

elif page == "Research":

    st.header("Research and Development")

    st.write(
        "NEUROPULSE explores computational approaches "
        "to pupil-response recording and physiological research."
    )

    st.subheader("Research Directions")

    with st.container(border=True):

        st.markdown("**Pupil Detection and Tracking**")

        st.write(
            "Exploring image-processing approaches to identify "
            "and track pupil boundaries across video frames."
        )

    with st.container(border=True):

        st.markdown("**Quantitative Pupil Analysis**")

        st.write(
            "Investigating pupil diameter, constriction amplitude, "
            "latency and redilation characteristics."
        )

    with st.container(border=True):

        st.markdown("**Longitudinal Research Profiles**")

        st.write(
            "Organizing repeated recordings to explore "
            "changes in pupil-response characteristics over time."
        )

    st.subheader("Development Focus")

    st.write(
        "Building a structured platform for video acquisition, "
        "visualization, computational analysis and future validation."
    )


# ==========================================
# 13. ABOUT
# ==========================================

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

    st.subheader("Research Disclaimer")

    st.info(
        "NEUROPULSE is an experimental research platform. "
        "It is not a validated medical device and is not intended "
        "for clinical diagnosis or treatment decisions."
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.markdown(
    "**BIOPTIQ** | The Future of Vision."
)

st.caption(
    "NEUROPULSE | See the Signal. Unlock Better Care."
)

st.caption(
    "Experimental research platform."
)

            
