import json
import time
import requests
import streamlit as st
import os


# Configure page settings
st.set_page_config(page_title = "Aptino Claim Engine", page_icon = "🏥", layout = "wide")

# Retrieve the backend URL from Streamlit Secrets, environment variables, or local default
if "BACKEND_URL" in st.secrets:
    BASE_API_URL = st.secrets["BACKEND_URL"]
else:
    BASE_API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

API_URL = f"{BASE_API_URL.rstrip('/')}/analyze"

# Default test payload (PUB-001)
DEFAULT_PAYLOAD = {
    "case_id": "PUB-001",
    "policy_id": "USGIC-CSC-2017-2018",
    "policy_start_date": "2025-01-01",
    "claim_date": "2026-03-14",
    "sum_insured_inr": 500000,
    "treatment": {
        "type": "inpatient",
        "diagnosis": "Acute appendicitis",
        "procedure": "Appendectomy",
        "pre_existing": False
    },
    "expenses_inr": {
        "room": 30000,
        "doctor_fees": 30000,
        "medicines_diagnostics": 90000
    }
}


def main():
    st.title("🏥 Aptino Policy-Aware Claim Engine")
    st.markdown("Multi-Agent RAG system for analyzing health insurance claims.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Claim Input (JSON)")
        
        # ADDED: File Uploader to satisfy "select/upload" requirement
        uploaded_file = st.file_uploader("Upload a Claim JSON file (Optional)", type = ["json"])
        
        # Pre-fill text area based on upload or default
        if uploaded_file is not None:
            file_contents = uploaded_file.getvalue().decode("utf-8")
            display_value = file_contents
        else:
            display_value = json.dumps(DEFAULT_PAYLOAD, indent = 4)
            
        claim_input = st.text_area(
            "Or paste claim case data here:",
            value = display_value,
            height = 350
        )
        
        analyze_button = st.button("Analyze Claim", type = "primary", use_container_width = True)
        
    with col2:
        st.subheader("Engine Output")
        
        if analyze_button:
            try:
                payload = json.loads(claim_input)
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
                return
                
            # ADDED: Track start time
            start_time = time.time()
                
            # The Interactive Loader
            with st.status("🤖 Multi-Agent Engine is processing the claim... (This usually takes 30-40 seconds)", expanded = True) as status:
                st.write("📡 Sending claim data to FastAPI backend...")
                
                try:
                    # Make the synchronous API call
                    response = requests.post(API_URL, json = payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Calculate elapsed time
                    end_time = time.time()
                    elapsed_time = round(end_time - start_time, 2)
                    
                    # Update the status box to show the trace dynamically
                    st.write(f"✅ Response received in {elapsed_time} seconds. Parsing execution trace...")
                    for step in data.get("trace", []):
                        time.sleep(0.3)  # Small delay for visual effect
                        st.write(f"➡️ **[{step['agent']}]**: {step['action']}")
                        
                    status.update(label = f"Analysis Complete! (⏱️ {elapsed_time}s)", state = "complete", expanded = False)
                    
                except requests.exceptions.RequestException as e:
                    status.update(label = "API Request Failed", state = "error")
                    st.error(f"Error connecting to backend: {e}")
                    return
            
            # --- Render the Results ---
            st.divider()
            
            # 1. Decision Status
            decision = data.get("decision", "UNKNOWN")
            confidence = data.get("confidence", 0.0)
            
            if decision == "ADMISSIBLE":
                st.success(f"### Decision: {decision} (Confidence: {confidence})")
            elif decision == "ADMISSIBLE_WITH_LIMITS":
                st.warning(f"### Decision: {decision} (Confidence: {confidence})")
            elif decision == "NEEDS_REVIEW":
                st.info(f"### Decision: {decision} (Confidence: {confidence})")
            else:
                st.error(f"### Decision: {decision} (Confidence: {confidence})")
                
            # 2. Validation Status & Elapsed Time
            validation = data.get("validation", {})
            val_status = validation.get("status")
            
            # ADDED: Display elapsed time explicitly in the metrics
            st.caption(f"⏱️ **Total Execution Time:** {elapsed_time} seconds")
            
            if val_status == "PASS":
                st.caption("✅ **Validation Agent:** Citations strictly validated against policy chunks.")
            else:
                st.caption(f"❌ **Validation Agent:** Citation validation failed: {validation.get('unsupported_claims')}")
                
            # 3. Key Findings & Limits
            with st.expander("📊 Findings & Applicable Limits", expanded = True):
                if data.get("key_findings"):
                    st.markdown("**Key Findings:**")
                    for finding in data["key_findings"]:
                        st.markdown(f"- {finding}")
                
                if data.get("applicable_limits"):
                    st.markdown("**Applicable Limits:**")
                    for limit in data["applicable_limits"]:
                        st.markdown(f"- {limit}")
                        
                if data.get("missing_evidence"):
                    st.markdown("**Missing Evidence:**")
                    for missing in data["missing_evidence"]:
                        st.markdown(f"- {missing}")

            # 4. Policy Citations
            with st.expander("📑 Policy Evidence & Citations", expanded = True):
                for idx, cit in enumerate(data.get("citations", [])):
                    st.markdown(f"**Citation {idx + 1}:** {cit.get('claim')}")
                    st.markdown(f"*Source: {cit.get('source')} | Section: {cit.get('section')} | Page: {cit.get('page')} | ID: {cit.get('chunk_id')}*")
                    st.divider()


if __name__ == "__main__":
    main()
