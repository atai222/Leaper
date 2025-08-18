import pandas as pd
import subprocess
import os

# --- CONFIGURATION ---
MANIFEST_FILE = 'clip_manifest.csv'
OUTPUT_FOLDER = 'data/raw_videos/'

# --- SCRIPT ---

def create_clips_from_manifest():
    """
    Reads a manifest CSV file and downloads video clips for each entry using yt-dlp,
    re-encoding them to ensure compatibility.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Creating output directory: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    try:
        manifest_df = pd.read_csv(MANIFEST_FILE)
    except FileNotFoundError:
        print(f"ERROR: Manifest file not found at '{MANIFEST_FILE}'")
        return

    print(f"Found {len(manifest_df)} clips to process from '{MANIFEST_FILE}'.")

    for index, row in manifest_df.iterrows():
        output_filename = (
            f"{row['competition']}_{row['year']}_"
            f"{row['athlete']}_JUMP{row['jump_num']}.mp4"
        )
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping '{output_filename}', file already exists.")
            continue

        url = row['youtube_url']
        start_time = row['start_time']
        end_time = row['end_time']
        
        # --- MODIFIED COMMAND ---
        # Added '--recode-video mp4' to standardize the output file format.
        command = [
            'yt-dlp',
            '--download-section', f"*{start_time}-{end_time}",
            '--recode-video', 'mp4',  # <--- THIS IS THE NEW, IMPORTANT LINE
            '-f', 'bestvideo',       # We can simplify this since we are recoding
            '-o', output_path,
            url
        ]
        
        print(f"\nProcessing clip: {output_filename}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully created clip: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to download clip for {output_filename}.")
            print("yt-dlp error output:", e.stderr)
        except FileNotFoundError:
            print("ERROR: 'yt-dlp' command not found.")
            break

if __name__ == "__main__":
    create_clips_from_manifest()
    print("\n--- Clipping process complete. ---")
