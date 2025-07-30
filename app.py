# Replace the section after "Play Audio" with this corrected logic:

        # Play Audio
        st.audio(audio_path, format="audio/wav", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=False, width="stretch")

        # Edit transcript step
        st.subheader("Step 2: Review and Edit Transcript")
        transcript_edited = st.text_area("Edit Transcript and Confirm", value=transcript, height=300)
        
        # Store which button was pressed in session state
        if 'report_type' not in st.session_state:
            st.session_state.report_type = None
            
        if run_highlights:
            st.session_state.report_type = 'highlights'
        elif run_bullets:
            st.session_state.report_type = 'bullets'
        elif transcript_only:
            st.session_state.report_type = 'transcript_only'

        confirm_transcript = st.button("✅ Confirm Transcript and Generate Report")

        # Process the report after transcript confirmation
        if confirm_transcript and st.session_state.report_type:
            # Update transcript with edited version
            transcript = transcript_edited
            transcript_docx = re.sub(r'(\[\d+:\d+:\d+\.\d+\] Speaker [A-Z])', r'<br><br>\1', transcript)
            transcript_docx = '<p>' + transcript_docx.strip() + '</p>'
            
            if st.session_state.report_type == 'highlights':
                with st.spinner("Writing Highlights..."):
                    try:
                        type = "format_text_highlight_prompt"
                        bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY, type)
                    except Exception as e:
                        bullets = []
                        st.warning("Bullet extraction failed.")

                with st.spinner("Formatting Tracking Report..."):
                    try:
                        html = generate_html_report(metadata, bullets, transcript, target_name)
                        save_text_file(html, report_path)
                        docx = generate_docx_report(metadata, bullets, transcript_docx, target_name)
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        st.stop()
                        
            elif st.session_state.report_type == 'bullets':
                with st.spinner("Writing Bullets..."):
                    try:
                        type = "format_text_bullet_prompt"
                        bullets = extract_raw_bullet_data_from_text(transcript, target_name, metadata, OPENAI_API_KEY, type)
                    except Exception as e:
                        bullets = []
                        st.warning("Bullet extraction failed.")

                with st.spinner("Formatting Tracking Report..."):
                    try:
                        html = generate_html_report_bullets(metadata, bullets, transcript, target_name)
                        save_text_file(html, report_path)
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        st.stop()
                        
            elif st.session_state.report_type == 'transcript_only':
                with st.spinner("Formatting Transcript Output..."):
                    try:
                        html = f"<h2>{target_name} Transcript</h2>" + transcript
                        save_text_file(html, report_path)
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        st.stop()

            # Mark processing as complete and store results
            st.session_state["processing_done"] = True
            st.session_state["html_report"] = html
            
            if video_url:
                try:
                    with open(audio_path, "rb") as f:
                        mp3_bytes = f.read()
                    st.session_state["mp3_data"] = mp3_bytes
                except Exception as e:
                    st.warning(f"Could not prepare MP3 download: {e}")
                    st.session_state["mp3_data"] = None

        # Display results if processing is complete
        if st.session_state.get("processing_done", False) and "html_report" in st.session_state:
            st.success("✅ Analysis complete!")
            
            # Show HTML report
            st.markdown(st.session_state["html_report"], unsafe_allow_html=True)
            
            # Download buttons
            st.download_button(
                "📄 Download HTML Report",
                data=st.session_state["html_report"],
                file_name=report_path.name,
                mime="text/html"
            )

            # DOCX download (if docx was generated)
            if st.session_state.report_type in ['highlights', 'bullets']:
                docx_buffer = html2docx(st.session_state["html_report"], title="Converted Document")
                st.download_button(
                    label="Download DOCX",
                    data=docx_buffer.getvalue(),
                    file_name=f"{target_name}_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            if video_url and "mp3_data" in st.session_state:
                st.download_button(
                    "🎵 Download MP3 File",
                    data=st.session_state["mp3_data"],
                    file_name=audio_path.name,
                    mime="audio/mpeg"
                )

    # Display saved results if they exist (for when user returns to page)
    elif "html_report" in st.session_state:
        st.markdown(st.session_state["html_report"], unsafe_allow_html=True)
            
        st.download_button(
            "📄 Download HTML Report",
            data=st.session_state["html_report"],
            file_name="report.html",
            mime="text/html"
        )
        
        if "mp3_data" in st.session_state and st.session_state["mp3_data"]:
            st.download_button(
                "🎵 Download MP3 File",
                data=st.session_state["mp3_data"],
                file_name="audio.mp3",
                mime="audio/mpeg"
            )
