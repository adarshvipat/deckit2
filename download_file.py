#!/usr/bin/env python3
"""Download a file from a given URL."""

import sys
import requests
import os
from urllib.parse import urlparse

def download_file(url, output_filename=None):
    """Download a file from the given URL and save it locally."""
    try:
        # Make a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Determine the output filename
        if output_filename is None:
            # Try to get filename from Content-Disposition header
            content_disposition = response.headers.get('Content-Disposition', '')
            if 'filename=' in content_disposition:
                output_filename = content_disposition.split('filename=')[1].strip('"\'')
            else:
                # Extract filename from URL
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename or '.' not in filename:
                    filename = 'downloaded_file.ics'  # Default for ICS files
                output_filename = filename
        
        # Download and save the file
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded: {output_filename}")
        return output_filename
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python download_file.py <URL> [output_filename]", file=sys.stderr)
        print("Example: python download_file.py https://example.com/file.ics", file=sys.stderr)
        sys.exit(1)
    
    url = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else None
    
    download_file(url, output_filename)

if __name__ == "__main__":
    main()

