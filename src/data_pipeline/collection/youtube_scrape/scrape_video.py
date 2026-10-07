import yt_dlp
import pandas as pd
import os
import tempfile
import re
import shutil
import argparse
import sys
import time
import random
import csv
from datetime import datetime

# List of all fields requested
ALL_REQUESTED_FIELDS = [
    'id', 'title', 'fulltitle', 'ext', 'alt_title', 'description', 'display_id', 
    'uploader', 'uploader_id', 'uploader_url', 'license', 'creators', 'creator', 
    'timestamp', 'upload_date', 'release_timestamp', 'release_date', 'release_year', 
    'modified_timestamp', 'modified_date', 'channel', 'channel_id', 'channel_url', 
    'channel_follower_count', 'channel_is_verified', 'location', 'duration', 
    'duration_string', 'view_count', 'concurrent_view_count', 'like_count', 
    'dislike_count', 'repost_count', 'average_rating', 'comment_count', 'age_limit', 
    'live_status', 'is_live', 'was_live', 'playable_in_embed', 'availability', 
    'media_type', 'start_time', 'end_time', 'extractor', 'extractor_key', 'epoch', 
    'autonumber', 'video_autonumber', 'n_entries', 'playlist_id', 'playlist_title', 
    'playlist', 'playlist_count', 'playlist_index', 'playlist_autonumber', 
    'playlist_uploader', 'playlist_uploader_id', 'playlist_channel', 
    'playlist_channel_id', 'playlist_webpage_url', 'webpage_url', 
    'webpage_url_basename', 'webpage_url_domain', 'original_url', 'categories', 
    'tags', 'cast', 'thumbnail',
    'transcript_source'
]

# Fields that are lists and should be joined into a string
LIST_FIELDS_TO_JOIN = ['creators', 'categories', 'tags', 'cast']

def clean_transcript(raw_subtitle_content):
    """
    Cleans raw VTT/SRT subtitle content to return only the spoken text,
    de-duplicating consecutive repeated lines.
    """
    if not raw_subtitle_content:
        return ""
    
    lines = raw_subtitle_content.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, WebVTT headers, timing lines, and number lines
        if not line or "-->" in line or line.upper() == "WEBVTT" or re.fullmatch(r'\d+', line):
            continue
            
        # Remove HTML-like tags (e.g., <c>, <v>)
        line = re.sub(r'<[^>]+>', '', line)
        # Condense multiple spaces into one
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Only add the line if it's not empty AND it's not the same as the last line we added
        if line and (not cleaned_lines or cleaned_lines[-1] != line):
            cleaned_lines.append(line)
            
    return " ".join(cleaned_lines)

# --- MODIFIED: Added cookie_file_path argument ---
def download_subtitle(video_url, video_id, subtitle_langs=None, cookie_file_path=None, sleep_min=30, sleep_max=90):
    """
    Downloads subtitles for a given video and returns the raw content.
    This function's options *prefer* manual subs but will grab auto-subs.
    """
    temp_dir = tempfile.mkdtemp()
    if not subtitle_langs:
        subtitle_langs = ["en.*", "en"]
    ydl_opts_subtitles = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        # Prefer manual when available, auto as fallback
        'subtitleslangs': subtitle_langs,
        'subtitlesformat': 'vtt/srt',
        'skip_download': True,
        'outtmpl': os.path.join(temp_dir, f'{video_id}.%(ext)s'),
        'quiet': True,
        'logtderr': False,
        'sleep_interval': sleep_min,
        'max_sleep_interval': sleep_max,
    }
    
    # --- NEW: Add cookie file if provided ---
    if cookie_file_path:
        ydl_opts_subtitles['cookiefile'] = cookie_file_path

    raw_subtitle_content = ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts_subtitles) as ydl:
            ydl.download([video_url])
        # Find the downloaded file
        for file in os.listdir(temp_dir):
            if file.startswith(video_id) and file.endswith(('.en.vtt', '.en.srt', '.vtt', '.srt')):
                with open(os.path.join(temp_dir, file), 'r', encoding='utf-8') as f:
                    raw_subtitle_content = f.read()
                break
    except Exception as e:
        print(f"    -> Could not download subtitles for {video_id}: {e}", file=sys.stderr)
    finally:
        shutil.rmtree(temp_dir)
    return raw_subtitle_content

def should_rate_limit_sleep(error_message):
    if not error_message:
        return False
    msg = str(error_message).lower()
    return (
        "rate-limited" in msg or
        "try again later" in msg or
        "this content isn't available" in msg
    )


def write_summary_line(summary_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def summarize_failure(error_message):
    msg = (str(error_message) or "").lower()
    if "rate-limited" in msg or "try again later" in msg:
        return "rate_limited"
    if "this content isn't available" in msg or "unavailable" in msg:
        return "unavailable"
    if "private" in msg or "members only" in msg:
        return "restricted"
    if "sign in" in msg or "cookies" in msg:
        return "auth_or_cookies"
    return "other"


def english_keys(captions_dict):
    if not captions_dict:
        return []
    preferred = ["en", "en-orig", "en-US", "en-GB", "en-CA", "en-AU"]
    ordered = []
    for key in preferred:
        if key in captions_dict and key not in ordered:
            ordered.append(key)
    for key in captions_dict.keys():
        if key.lower().startswith("en") and key not in ordered:
            ordered.append(key)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube playlist/channel metadata.")
    parser.add_argument('--url', required=True, help="The YouTube playlist or channel URL.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the output CSV file.")
    # --- NEW: Add --cookies argument ---
    parser.add_argument('--cookies', help="Path to a cookies.txt file.")
    parser.add_argument('--sleep-min', type=int, default=30, help="Minimum sleep between video requests (seconds).")
    parser.add_argument('--sleep-max', type=int, default=90, help="Maximum sleep between video requests (seconds).")
    parser.add_argument('--rate-limit-sleep', type=int, default=3600, help="Sleep time on rate limit (seconds).")
    parser.add_argument('--max-retries', type=int, default=2, help="Retries for rate-limited videos.")
    args = parser.parse_args()
    
    # --- NEW: Store cookie path in a variable ---
    cookie_path = args.cookies
    sleep_min = max(0, args.sleep_min)
    sleep_max = max(sleep_min, args.sleep_max)
    rate_limit_sleep = max(60, args.rate_limit_sleep)
    max_retries = max(0, args.max_retries)

    # Step 1: Get a flat list of all video IDs
    playlist_ydl_opts = {
        'quiet': True, 
        'extract_flat': True, 
        'logtderr': False,
        'sleep_interval': sleep_min,
        'max_sleep_interval': sleep_max,
    }
    
    # --- NEW: Add cookie file if provided ---
    if cookie_path:
        playlist_ydl_opts['cookiefile'] = cookie_path
        
    # Ensure output dir exists early so summary logs can be written
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with yt_dlp.YoutubeDL(playlist_ydl_opts) as ydl:
            playlist_info = ydl.extract_info(args.url, download=False)
    except Exception as e:
        print(f"FATAL: Could not extract playlist info from {args.url}. Reason: {e}", file=sys.stderr)
        return

    playlist_title = playlist_info.get('title', 'untitled_playlist')
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', playlist_title)
    output_filename = os.path.join(args.output_dir, f"{safe_title}.csv")
    summary_log = os.path.join(args.output_dir, f"{safe_title}_summary.log")

    video_entries = playlist_info.get('entries', [])
    if not video_entries:
        print(f"Could not find any videos in {args.url}.")
        return
        
    print(f"Ensuring output directory exists: {args.output_dir}")

    print(f"Found {len(video_entries)} videos in '{playlist_title}'. Starting data extraction...")
    print(f"Output CSV will be saved to: {output_filename}")
    write_summary_line(summary_log, f"Playlist URL: {args.url}")
    write_summary_line(summary_log, f"Playlist title: {playlist_title}")
    write_summary_line(summary_log, f"Total videos: {len(video_entries)}")
    write_summary_line(summary_log, f"Cookies file: {cookie_path if cookie_path else 'none'}")
    write_summary_line(summary_log, f"Sleep range: {sleep_min}-{sleep_max}s, rate-limit sleep: {rate_limit_sleep}s, retries: {max_retries}")

    final_columns = ALL_REQUESTED_FIELDS + ['cleaned_subtitles']

    # Resume support: skip already-scraped video IDs
    existing_ids = set()
    write_header = True
    if os.path.exists(output_filename):
        try:
            with open(output_filename, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'id' in reader.fieldnames:
                    for row in reader:
                        vid = row.get('id')
                        if vid:
                            existing_ids.add(vid)
                    write_header = False
        except Exception as e:
            print(f"⚠️  Could not read existing CSV for resume: {e}", file=sys.stderr)
    if existing_ids:
        write_summary_line(summary_log, f"Resume enabled. Existing videos: {len(existing_ids)}")
        print(f"Resume enabled. Skipping {len(existing_ids)} already-scraped videos.")

    # Step 2: Loop through each video and extract its full metadata
    ydl_opts = {
        'quiet': True, 
        'skip_download': True, 
        'logtderr': False,
        'listsubtitles': True,
        'sleep_interval': sleep_min,
        'max_sleep_interval': sleep_max,
    }
    
    # --- NEW: Add cookie file if provided ---
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path
        
    total_processed = 0
    total_failed = 0
    total_rate_limited = 0
    total_no_subs = 0
    total_with_subs = 0
    total_cleaned_nonempty = 0
    transcript_source_counts = {
        "manual": 0,
        "auto_generated": 0,
        "none": 0,
    }
    subtitle_language_counts = {}
    failure_reasons = {
        "rate_limited": 0,
        "unavailable": 0,
        "restricted": 0,
        "auth_or_cookies": 0,
        "other": 0,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, entry in enumerate(video_entries):
            video_id = entry.get('id')
            if video_id in existing_ids:
                print(f"  ⏭️  Skipping already-scraped video: {entry.get('title', video_id)}")
                continue
            
            # Sleep for a random time to avoid rate-limit
            if i > 0:
                sleep_duration = random.randint(sleep_min, sleep_max)
                print(f"  ...sleeping for {sleep_duration}s to avoid rate-limit...")
                time.sleep(sleep_duration) 
            
            # --- FIXED: Added '://' to the URL ---
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"\nProcessing video {i+1}/{len(video_entries)}: {entry.get('title', video_id)}")

            try:
                attempt = 0
                while True:
                    try:
                        video_info = ydl.extract_info(video_url, download=False)
                        break
                    except Exception as e:
                        if should_rate_limit_sleep(e) and attempt < max_retries:
                            total_rate_limited += 1
                            attempt += 1
                            print(
                                f"  -> Rate-limited. Sleeping {rate_limit_sleep}s "
                                f"before retry {attempt}/{max_retries}..."
                            )
                            write_summary_line(
                                summary_log,
                                f"Rate-limited on video {video_id}; sleeping {rate_limit_sleep}s "
                                f"before retry {attempt}/{max_retries}"
                            )
                            time.sleep(rate_limit_sleep)
                            continue
                        raise
                
                raw_subtitles = ""
                cleaned_subtitles = ""
                transcript_source = "none"

                # Check for manual English transcripts first, then auto
                subtitle_lang = None
                manual_subs = video_info.get('subtitles', {}) or {}
                auto_subs = video_info.get('automatic_captions', {}) or {}

                manual_keys = english_keys(manual_subs)
                auto_keys = english_keys(auto_subs)

                for key in manual_keys:
                    transcript_source = "manual"
                    subtitle_lang = key
                    print(f"  -> Found manual English transcript ({key}). Downloading...")
                    raw_subtitles = download_subtitle(
                        video_url, video_id, [key, "en.*", "en"], cookie_path, sleep_min, sleep_max
                    )
                    if raw_subtitles:
                        break

                if not raw_subtitles:
                    for key in auto_keys:
                        transcript_source = "auto_generated"
                        subtitle_lang = key
                        print(f"  -> Found auto-generated English subtitles ({key}). Downloading...")
                        raw_subtitles = download_subtitle(
                            video_url, video_id, [key, "en.*", "en"], cookie_path, sleep_min, sleep_max
                        )
                        if raw_subtitles:
                            break

                if not raw_subtitles:
                    if not manual_keys and not auto_keys:
                        print("  -> No English subtitles found for this video.")
                    else:
                        print("  -> Subtitle download returned empty content.")
                    transcript_source = transcript_source if transcript_source != "none" else "none"
                    total_no_subs += 1
                
                cleaned_subtitles = clean_transcript(raw_subtitles)
                if cleaned_subtitles:
                    total_with_subs += 1
                    total_cleaned_nonempty += 1
                elif transcript_source != "none":
                    write_summary_line(
                        summary_log,
                        f"EMPTY_SUBTITLES {video_id}: source={transcript_source}, lang={subtitle_lang}"
                    )
                transcript_source_counts[transcript_source] = (
                    transcript_source_counts.get(transcript_source, 0) + 1
                )
                if subtitle_lang:
                    subtitle_language_counts[subtitle_lang] = (
                        subtitle_language_counts.get(subtitle_lang, 0) + 1
                    )

                # Collect all other metadata fields
                data_row = {}
                for field in ALL_REQUESTED_FIELDS:
                    if field == 'transcript_source': continue
                    value = video_info.get(field)
                    if field in LIST_FIELDS_TO_JOIN and isinstance(value, list):
                        data_row[field] = ', '.join(map(str, value))
                    else:
                        data_row[field] = value
                
                data_row['transcript_source'] = transcript_source
                data_row['cleaned_subtitles'] = cleaned_subtitles

                # --- LOGIC: Write this one row to the CSV ---
                df_row = pd.DataFrame([data_row], columns=final_columns)

                if write_header:
                    # If this is the first write, include the header
                    df_row.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
                    write_header = False
                else:
                    # Append without the header
                    df_row.to_csv(output_filename, index=False, header=False, mode='a', quoting=csv.QUOTE_MINIMAL)
                
                total_processed += 1
                
            except Exception as e:
                total_failed += 1
                reason_key = summarize_failure(e)
                failure_reasons[reason_key] += 1
                print(f"  -> Failed to process video {video_id}. Reason: {e}", file=sys.stderr)
                write_summary_line(summary_log, f"FAILED {video_id}: {e}")

    # Step 3: All done!
    print(f"\nData saved to {output_filename}")
    write_summary_line(summary_log, "=== Summary ===")
    write_summary_line(summary_log, f"Processed: {total_processed}")
    write_summary_line(summary_log, f"Failed: {total_failed}")
    write_summary_line(summary_log, f"With subtitles: {total_with_subs}")
    write_summary_line(summary_log, f"No subtitles: {total_no_subs}")
    write_summary_line(summary_log, f"Cleaned transcripts (non-empty): {total_cleaned_nonempty}")
    write_summary_line(summary_log, f"Transcript source counts: {transcript_source_counts}")
    write_summary_line(summary_log, f"Subtitle language counts: {subtitle_language_counts}")
    write_summary_line(summary_log, f"Rate-limited events: {total_rate_limited}")
    write_summary_line(summary_log, f"Failure reasons: {failure_reasons}")
    write_summary_line(
        summary_log,
        "Troubleshooting: If rate-limited, reduce parallelism or increase --sleep-min/--sleep-max. "
        "If auth/cookies errors, refresh cookies.txt. "
        "If unavailable/restricted, the video may be private, removed, or region-locked."
    )

if __name__ == "__main__":
    main()