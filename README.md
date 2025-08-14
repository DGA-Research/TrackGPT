TRACKGPT: AUDIO/VIDEO RESEARCH TRANSCRIBER AND ANALYZER
=======================================================
A Python program that automates the tracking report process: downloading, transcribing, and writing highlights and bullets. Originally created by Shawn Patterson, the program has been adapted for DGA's preferred output and connected to Streamlit for a user-friendly interface. This ReadMe is focused on use with Streamlit. To see how to use the command line interface, check out Shawn's ReadMe: https://github.com/sh-patterson/TrackGPT-Audio. 

FEATURES
========
Core Functionality:
- Multi-Source Support: Download audio from YouTube, Vimeo, Twitter, TikTok, and 1000+ other platforms using yt-dlp
- File Upload: Support for MP3, M4A, and MP4 file uploads (up to 600MB with compression guidance)
- Smart Transcription: Uses AssemblyAI and OpenAi for speaker identification and diarization
- AI Analysis: Extract structured insights using GPT models with customizable prompts
- Multiple Report Types: Generate highlights, research bullets, or combined reports
- Interactive Editing: Review and edit transcripts and speaker labels before analysis

Additional Features:
- Speaker Detection: Automatic speaker labeling with manual editing capabilities
- Multiple Export Formats: HTML and DOCX report generation with DGA customized formatting
- Audio Preservation: Download processed audio files alongside reports
- Password Protection: Requires password to access Streamlit app


ARCHITECTURE
============
TrackGPT/
├── app.py               # Main Streamlit application with multi-step workflow
├── config.py            # Configuration management with validation
├── downloader.py        # Audio download functionality using yt-dlp
├── transcriber.py       # Audio transcription with AssemblyAI integration and OpenAI speaker labeling prompts
├── analyzer.py          # AI-powered content analysis with retry logic
├── output.py            # Report generation with HTML/DOCX formatting
├── prompts.py           # AI prompt templates for highlights and bullets
├── requirements.txt     # Python dependencies with version specifications
├── config.toml          # Streamlit configuration for uploads and performance
└── packages.txt         # System packages (FFmpeg)

Data Flow Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────────────┐
│   Source    │ -> │  Download/  │ -> │ Transcribe  │ -> │ Add Speaker Labels │
│ (URL/File)  │    │   Upload    │    │ (AssemblyAI)│    │      (OpenAI)      │
└─────────────┘    └─────────────┘    └─────────────┘    └────────────────────┘
                                                                      │
┌───────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌─────────────┐
│   Export      │ <- │   Format    │ <- │  Write Bullets /  │ <- │ User Edits  │
│(HTML/DOCX/MP3)│    │   Report    │    │    Highlights     │    │ Transcript  │
└───────────────┘    └─────────────┘    │     (OpenAI)	    │    └─────────────┘
				        └───────────────────┘ 

DETAILED WORKFLOW & USE CASES
=============================
Step-by-Step Process:

1. Authentication & Access Control
   - Password-protected interface prevents unauthorized access
   - Session management maintains state across workflow steps
   - Secure handling of API keys through Streamlit secrets

2. Input Source Selection
   Available Input Methods:
   - URL Input: Paste any video/audio URL from supported platforms
   - File Upload: Drag and drop MP3, M4A, or MP4 files
   - Direct Transcript: Paste existing transcripts for analysis only
   
   Supported Platforms (via yt-dlp):
   - YouTube (videos, playlists, live streams)
   - Vimeo (public and private videos)
   - Twitter/X (video posts)
   - TikTok (public videos)
   - Facebook (public videos)
   - Instagram (public videos)
   - LinkedIn (video posts)
   - Reddit (video posts)
   - Twitch (VODs and clips)
   - And 1000+ other platforms
  * In theory, yt-dlp can handle all these formats, but there have been frequent issues with YouTube blockers.

3. Configuration Options
   - Target Name: Person to focus analysis on
   - Report Type Options:
     * Highlights: Key moments and important statements
     * Research Bullets: Structured factual claims and statements
     * Combined: Both highlights and bullets in one report
     * Transcript Only: Clean transcript with speaker labels
   
   Optional Metadata Fields:
   - Custom title (overrides auto-detected)
   - Uploader/Channel name
   - Upload date
   - Platform/Source identification

4. Processing & Transcription
   - AssemblyAI transcription with speaker diarization
   - Formats into 30 second intervals
   - Call to OpenAI API to add speaker labels

5. Transcript Review & Editing
   - Audio playback available in Streamlit
   - Full transcript displays with timestamps and option to edit
   - Option for manual speaker name editing
  
6. AI Analysis & Report Generation
   - OpenAI GPT analysis with specialized prompts
   - Structured data extraction based on target focus
   - DGA formatting and citation generation

7. Export & Download
   - HTML reports with embedded styling
   - DOCX files compatible with Microsoft Word
   - Original audio file download


ADVANCED CONFIGURATION
======================
Environment Variables Reference:

API Configuration:
- OPENAI_API_KEY: Your OpenAI API key (required)
- ASSEMBLYAI_API_KEY: Your AssemblyAI API key (required)

Streamlit Configuration (config.toml):

[server]
maxUploadSize = 600                   # File upload limit in MB
maxMessageSize = 200                  # WebSocket message size in MB
enableCORS = false                    # CORS handling
enableXsrfProtection = true          # XSRF protection

[browser]
gatherUsageStats = false             # Disable usage statistics
showErrorDetails = false             # Hide error details in production

[logger]
level = "info"                       # Logging level (debug, info, warning, error)
enableLogging = true                 # Enable/disable logging

[theme]
primaryColor = "#1f77b4"            # Primary UI color
backgroundColor = "#ffffff"          # Background color
secondaryBackgroundColor = "#f0f2f6" # Secondary background
textColor = "#262730"               # Text color

AI Prompt Configuration:

The application uses two specialized prompt templates defined in prompts.py:

1. Highlights Prompt (format_text_highlight_prompt):
   - Extracts memorable quotes and key moments
   - Focuses on impactful statements and emotional content
   - Emphasizes specific analysis and maintaining original wording

2. Research Bullets Prompt (format_text_bullet_prompt):
   - Extracts factual claims and commitments
   - Provides structured data with citations
   - Includes speaker attribution and source information

Prompt Customization: 
To modify prompts, edit the template strings in prompts.py:
- Adjust focus areas and extraction criteria
- Modify output format and structure
- Change citation and attribution requirements
- Add domain-specific instructions


TECHNICAL DEEP-DIVE
===================
Speaker Label Format:
[HH:MM:SS] Speaker A (Optional_Name): Transcript text
[HH:MM:SS] Speaker B (Optional_Name): Transcript text

Manual Speaker Editing:
- Pattern-based speaker identification
- Bulk find-and-replace functionality
- Validation of speaker consistency
- Name mapping and standardization

Processing Limitations:

File Size Limits:
- Maximum upload: 600MB (configurable)
- Automatic compression recommendations
- Chunking threshold: 24MB for API compatibility
- Memory usage: ~2x file size during processing

Duration Limits:
- Maximum audio length: 12 hours
- Recommended: Under 3 hours for optimal performance
- Processing time: ~10% of audio duration
- Timeout settings: 30 minutes default


API INTEGRATION DETAILS
=======================
AssemblyAI Integration:

Features and Capabilities:
- Speaker Diarization: Automatic detection of who is speaking
- Speaker Labels: Identification of individual voices
- Punctuation and Capitalization: Automatic formatting
- Confidence Scores: Per-word and segment confidence ratings
- Custom Vocabulary: Domain-specific term recognition
- Content Moderation: Automatic detection of sensitive content

API Specifications:
- Base URL: https://api.assemblyai.com/v2/
- Authentication: Bearer token in Authorization header
- Rate Limits: 5 concurrent requests, 100 requests/minute
- File Size Limit: 5GB per request
- Supported Languages: English (primary), 70+ languages (beta)

Cost Structure (approximate):
- Audio Transcription: $0.00037/second (~$1.33/hour)
- Speaker Diarization: Additional $0.00009/second (~$0.32/hour)

Error Handling:
- Automatic retry for transient failures
- Exponential backoff strategy (3-30 second delays)
- Graceful degradation for partial failures
- Detailed error logging and reporting

OpenAI Integration:

Model Comparison:
GPT-4o-mini (Default):
- Cost: $0.150/1M input tokens, $0.600/1M output tokens
- Speed: ~2,000 tokens/second
- Context: 128k tokens (~100 pages)
- Quality: High for most analysis tasks
- Best for: Cost-effective research and monitoring

GPT-4-turbo:
- Cost: $10/1M input tokens, $30/1M output tokens
- Speed: ~1,000 tokens/second  
- Context: 128k tokens
- Quality: Superior reasoning and analysis
- Best for: Complex analysis, legal research, academic work

Token Usage Estimation:
- Average transcript: 100-500 tokens per minute of audio
- Analysis prompt: 2,000-3,000 tokens
- Output generation: 200-1,000 tokens per extracted point
- Total per analysis: 5,000-15,000 tokens average

Rate Limiting:
- Tier 1 (Free): 3 requests/minute, 200/day
- Tier 2 (Pay-as-go): 3,500 requests/minute
- Tier 3 (Usage $50+): 5,000 requests/minute
- Enterprise: Custom limits available


REPORT FORMAT EXAMPLES 
======================
Highlights Report Structure:

HTML Output Example:
```html
<h1>Research Report: [Target Name]</h1>
<div class="metadata">
  <h2>Source Information</h2>
  <ul>
    <li><strong>Title:</strong> [Video/Audio Title]</li>
    <li><strong>Source:</strong> [Platform/Channel]</li>
    <li><strong>Date:</strong> [Upload/Broadcast Date]</li>
    <li><strong>Duration:</strong> [Runtime]</li>
  </ul>
</div>

<div class="highlights">
  <h2>Key Highlights</h2>
  <ul>
    <li><strong>[Timestamp]:</strong> [Impactful Quote or Statement]</li>
    <li><strong>[Timestamp]:</strong> [Policy Position or Commitment]</li>
    <li><strong>[Timestamp]:</strong> [Memorable Phrase or Soundbite]</li>
  </ul>
</div>

<div class="transcript">
  <h2>Full Transcript</h2>
  <p><strong>[00:00:15] Speaker A (John Doe):</strong> Opening remarks...</p>
  <p><strong>[00:02:30] Speaker B (Jane Smith):</strong> Response...</p>
</div>
```

Research Bullets Report Structure:

Each bullet point contains structured data:
- Headline: Concise summary of the factual claim
- Speaker: Who made the statement (with confidence attribution)  
- Body: Relevant contextual passage from transcript
- Source: Original source identification
- Date: Timestamp or date reference

Sample Research Bullet:
```
*** BULLET START ***
**Headline:** John Doe announced plans to increase education funding by 15% next fiscal year.
**Speaker:** JOHN DOE
**Body:** When asked about his education priorities, John Doe stated "We're committed to increasing education funding by 15% in the next fiscal year. This represents the largest investment in our schools in over a decade."
**Source:** Channel 7 News Interview
**Date:** 2024-03-15
*** BULLET END ***
```


DEVELOPER DOCUMENTATION
=======================
Core Modules Reference:

config.py - Configuration Management:
```python
class Config:
    # Essential API Keys
    OPENAI_API_KEY: str          # Required OpenAI API key
    ASSEMBLYAI_API_KEY: str      # Required AssemblyAI key
    
    # Model Configuration  
    ANALYSIS_MODEL: str          # GPT model selection
    WHISPER_MODEL: str           # Whisper model (fallback)
    
    # Processing Settings
    AUDIO_FORMAT: str            # Output audio format
    DEFAULT_OUTPUT_DIR: str      # File output directory
    
    @classmethod
    def validate(cls) -> None:   # Configuration validation
```

downloader.py - Audio Extraction:
```python
def download_audio(url: str, output_dir: Path, base_filename: str, 
                  file_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Downloads audio from URL using yt-dlp.
    
    Args:
        url: Source URL for video/audio
        output_dir: Directory for output files
        base_filename: Base name for generated files  
        file_type: File type indicator ("AUDIO" or "VIDEO")
    
    Returns:
        Tuple of (audio_file_path, metadata_dict)
    
    Raises:
        DownloadError: If download fails
        ValidationError: If URL is invalid
    """

def find_yt_dlp_executable() -> Optional[str]:
    """Locates yt-dlp executable on system."""

def find_ffmpeg_executable() -> Optional[str]:  
    """Locates ffmpeg executable on system."""
```

transcriber.py - Audio Transcription:
```python
def transcribe_file(audio_file_path: str, openai_key: str, 
                   assemblyai_key: str, speaker: str) -> str:
    """
    Transcribes audio file using AssemblyAI.
    
    Args:
        audio_file_path: Path to audio file
        openai_key: OpenAI API key (fallback)
        assemblyai_key: AssemblyAI API key
        speaker: Target speaker name
    
    Returns:
        Formatted transcript with timestamps and speaker labels
    """

def format_timestamp(ms: int) -> str:
    """Converts milliseconds to HH:MM:SS format."""
    
# Constants
CHUNK_SIZE_LIMIT = 24 * 1024 * 1024  # 24 MB API limit
DEFAULT_OVERLAP_SECONDS = 2           # Chunk overlap
```

analyzer.py - Content Analysis:
```python
@retry(wait=wait_random_exponential(min=3, max=30),
       stop=stop_after_attempt(4),
       retry=retry_if_exception_type((APIError, RateLimitError)))
def extract_raw_data_from_text(transcript: str, target_name: str,
                              metadata: Dict[str, Any], api_key: str,
                              prompt_type: str) -> str:
    """
    Extracts insights from transcript using OpenAI GPT.
    
    Args:
        transcript: Full transcript text
        target_name: Person/entity to focus analysis on
        metadata: Video/audio metadata
        api_key: OpenAI API key
        prompt_type: "format_text_highlight_prompt" or "format_text_bullet_prompt"
    
    Returns:
        Structured analysis results with delimited format
    """
```

output.py - Report Generation:
```python
def generate_report_highlights(metadata: Dict[str, Any], bullets: str,
                             transcript: str, target_name: str,
                             output_format: str) -> str:
    """Generates highlights report in HTML or DOCX format."""

def generate_report_bullets(metadata: Dict[str, Any], bullets: str,
                           transcript: str, target_name: str,
                           output_format: str) -> str:
    """Generates research bullets report."""

def generate_report_both(metadata: Dict[str, Any], bullets: str, 
                        highlights: str, transcript: str, target_name: str,
                        output_format: str) -> str:
    """Generates combined highlights and bullets report."""

def save_text_file(content: str, file_path: Path) -> None:
    """Saves text content to file with UTF-8 encoding."""

def apply_strict_title_case_every_word(text: str) -> str:
    """Applies title case formatting to text."""
```


SECURITY & PRIVACY
==================
Data Handling Practices:

Local Processing:
- Audio files processed locally when possible
- Temporary files stored in system temp directory
- Automatic cleanup of intermediate files
- No persistent storage of sensitive content
- Session-based data management only

API Data Transmission:
- HTTPS encryption for all API communications
- No audio files transmitted to OpenAI (transcripts only)
- AssemblyAI receives audio files securely
- API keys transmitted via secure headers
- No data logging or retention by default

File Security:
- Temporary files created with restricted permissions (600)
- Secure file deletion using os.remove() 
- No backup or cache files created
- Memory buffers cleared after processing
- Cross-platform secure temporary directories

API Key Security:


Streamlit Secrets Management:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
ASSEMBLYAI_API_KEY = "..."
password = "..."
```

Access Controls:
- Password protection for application access
- Session timeout after inactivity
- No user registration or account system
- Single-tenant deployment model
- Network-level access restrictions possible


TROUBLESHOOTING
===============
API-Related Issues:

Issue: "Invalid API key" errors
Solution:
1. Verify API keys are correctly formatted
2. Check .streamlit/secrets.toml file exists and has correct syntax
3. Ensure no extra spaces or quotes around keys
4. Test API keys independently

Processing Issues:

Issue: File upload failures
Solution:
1. Check file size limits in config.toml
2. Verify file format is supported
3. Try compressing large files
4. Use direct file path instead of upload

Issues with Streamlit Interface:
Solution:
- Reboot on the Streamlit Community Cloud Page


VERSION HISTORY & BRANCHES
===========================
Main branch
- Most up to date version. 
- Corrected formatting for docx and html
- Includes ability to edit transcript and speakers
- User process broken into 3 distinct steps
- Connected to streamlit

Without_transcript_editing branch:
- Can default back to this version if issues arise with the speaker editing feature in the main branch


IDEAS FOR IMPROVEMENT
=====================
Tweaking Prompts:
- As more DGA employees try out TrackGPT, we will continue to get more feedback.
- With more feedback and testing, I'm sure the prompts can be improved.
- There also could be more advanced prompt engineering techniques to be tried out.
- The bullet prompts are largely untouched from the original, if the research
Team decides it wants it, there is definitely more time that could be invested in...

Other API Edits:
- It would also be interesting to explore how other GPT models perform and play around with temperature more.


Downloading YouTube Videos:
- The feature downloading YouTube videos has not been working great.
- There is an update to ytdlp that can be installed.
- It would also be worth it to explore other ways to get around YouTube blocks.

Transcript Labels:
- The current set up for editing speaker labels could be made more user friendly.
- The time stamps splits in 30 second increments but it splits up sentences which could be fixed.


ACKNOWLEDGMENTS & CREDITS
=========================
Core Technologies:
- OpenAI: GPT models and Whisper API for advanced AI capabilities
- AssemblyAI: State-of-the-art speech recognition and speaker diarization
- Streamlit: Elegant and powerful web application framework
- yt-dlp: Comprehensive media downloading and extraction
- FFmpeg: Industry-standard multimedia framework

Contributors:
- Original concept and development: Shawn Patterson

License Information:
This project is licensed under the MIT License, providing:
- Commercial use permissions
- Modification and distribution rights
- Private use allowance
- No warranty or liability

The MIT License ensures this tool remains accessible to researchers, journalists, and analysts while allowing commercial applications and derivatives.

---
